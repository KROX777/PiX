"""DeepXDE/Paddle coefficient optimization for PiX symbolic equations."""

import contextlib
import io
import os
import time

os.environ["DDE_BACKEND"] = "paddle"

import deepxde as dde
import numpy as np
import paddle
import paddle_custom_device  # noqa: F401
import sympy as sp


DEFAULT_CONFIG = {
    "iterations": 2000,
    "learning_rate": 3e-3,
    "pde_loss_weight": 1.0,
    "data_loss_weight": 10.0,
    "max_points": 768,
    "hidden_layers": 3,
    "neurons": 32,
    "seed": 42,
    "device": "npu:0",
    "display_every": 0,
}


def _read_config(cfg):
    raw = {} if cfg is None else cfg.get("pinn_config", {})
    result = dict(DEFAULT_CONFIG)
    for key in result:
        if key in raw:
            result[key] = raw[key]
    if "epochs" in raw and "iterations" not in raw:
        result["iterations"] = raw["epochs"]
    result["iterations"] = int(result["iterations"])
    result["max_points"] = int(result["max_points"])
    result["hidden_layers"] = int(result["hidden_layers"])
    result["neurons"] = int(result["neurons"])
    result["seed"] = int(result["seed"])
    result["display_every"] = int(result["display_every"])
    if result["iterations"] <= 0 or result["max_points"] <= 0:
        raise ValueError("PINN iterations and max_points must be positive")
    return result


def _import_backend(device):
    if dde.backend.backend_name != "paddle":
        raise RuntimeError(
            f"DeepXDE backend must be paddle, got {dde.backend.backend_name!r}"
        )
    if str(device).startswith("npu"):
        if "npu" not in paddle.device.get_all_custom_device_type():
            raise RuntimeError("Paddle NPU custom device is not available")
    paddle.set_device(device)
    return dde, paddle


def _normalize_coordinates(grids):
    centers = []
    half_ranges = []
    normalized_grids = []
    for grid in grids:
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(f"PINN requires one-dimensional nontrivial grids, got {grid.shape}")
        center = float((grid[0] + grid[-1]) / 2.0)
        half_range = float((grid[-1] - grid[0]) / 2.0)
        if not np.isfinite(half_range) or half_range <= 0:
            raise ValueError(f"Invalid coordinate range: [{grid[0]}, {grid[-1]}]")
        centers.append(center)
        half_ranges.append(half_range)
        normalized_grids.append((grid - center) / half_range)
    mesh = np.meshgrid(*normalized_grids, indexing="ij")
    coordinates = np.column_stack([axis.reshape(-1) for axis in mesh]).astype(np.float32)
    return coordinates, np.asarray(centers), np.asarray(half_ranges)


def _normalize_fields(field_data, max_points, seed):
    field_data = np.asarray(field_data, dtype=np.float64)
    if field_data.ndim < 2:
        raise ValueError(f"PINN field data must include grid and field axes, got {field_data.shape}")
    n_fields = field_data.shape[-1]
    flattened = field_data.reshape(-1, n_fields)
    means = np.mean(flattened, axis=0)
    scales = np.std(flattened, axis=0)
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(scales)):
        raise ValueError("PINN field data contains non-finite statistics")
    if np.any(scales < 1e-8):
        raise ValueError(f"PINN field scale is degenerate: {scales}")
    normalized = ((flattened - means) / scales).astype(np.float32)
    rng = np.random.default_rng(seed)
    count = min(max_points, len(flattened))
    indices = rng.choice(len(flattened), size=count, replace=False)
    return normalized, means, scales, indices


def _residual_scales(calculator, init_params):
    residual_functions = calculator.gen_np_func(calculator.sp_equation)
    if len(residual_functions) != len(calculator.sp_equation):
        raise ValueError("PINN currently requires scalar PiX equations")
    scales = []
    for function in residual_functions:
        values = np.asarray(function(calculator.args_data, init_params), dtype=np.float64)
        scale = float(np.sqrt(np.mean(values**2)))
        scales.append(scale if np.isfinite(scale) and scale > 1e-5 else 1.0)
    return scales


class _PaddleExpressionEvaluator:
    def __init__(self, dde, paddle, calculator, coordinates, fields, coefficients,
                 centers, half_ranges):
        self.dde = dde
        self.paddle = paddle
        self.calculator = calculator
        self.coordinates = coordinates
        self.fields = fields
        self.coefficients = coefficients
        self.centers = centers
        self.half_ranges = half_ranges
        coordinate_names = calculator.spatial_vars + calculator.temporal_vars
        self.coordinate_indices = {name: i for i, name in enumerate(coordinate_names)}
        self.coordinate_symbols = {
            str(symbol): i for i, symbol in enumerate(
                list(calculator.space_axis)
                + ([calculator.t] if calculator.has_time else [])
            )
        }
        self.field_functions = {
            function.func: name for name, function in calculator.sp_field_funcs.items()
        }

    def _differentiate(self, value, variable, count):
        variable_name = str(variable)
        if variable_name not in self.coordinate_indices:
            raise ValueError(f"Unknown PINN derivative coordinate: {variable_name}")
        index = self.coordinate_indices[variable_name]
        for _ in range(int(count)):
            value = self.dde.grad.jacobian(
                value, self.coordinates, i=0, j=index
            ) / float(self.half_ranges[index])
        return value

    def evaluate(self, expression):
        if expression.is_Number:
            return float(expression)
        if expression.is_Symbol:
            name = str(expression)
            if name in self.coefficients:
                return self.coefficients[name]
            if name in self.coordinate_symbols:
                index = self.coordinate_symbols[name]
                return (
                    float(self.centers[index])
                    + float(self.half_ranges[index]) * self.coordinates[:, index:index + 1]
                )
            raise ValueError(f"Unsupported PINN symbol: {name}")
        if isinstance(expression, sp.Derivative):
            value = self.evaluate(expression.expr)
            for variable, count in expression.variable_count:
                value = self._differentiate(value, variable, count)
            return value
        if expression.func in self.field_functions:
            return self.fields[self.field_functions[expression.func]]
        if expression.is_Add:
            values = [self.evaluate(argument) for argument in expression.args]
            return sum(values[1:], values[0])
        if expression.is_Mul:
            values = [self.evaluate(argument) for argument in expression.args]
            result = values[0]
            for value in values[1:]:
                result = result * value
            return result
        if expression.is_Pow:
            exponent = expression.exp
            if not exponent.is_Number:
                raise ValueError(f"PINN requires a numeric exponent, got {exponent}")
            return self.evaluate(expression.base) ** float(exponent)

        functions = {
            sp.sin: self.paddle.sin,
            sp.cos: self.paddle.cos,
            sp.exp: self.paddle.exp,
            sp.tanh: self.paddle.tanh,
            sp.log: self.paddle.log,
            sp.Abs: self.paddle.abs,
            sp.sqrt: self.paddle.sqrt,
        }
        if expression.func in functions and len(expression.args) == 1:
            return functions[expression.func](self.evaluate(expression.args[0]))
        raise ValueError(
            f"Unsupported PINN expression node {type(expression).__name__}: {expression}"
        )


def _make_geometry(dde, dimension):
    if dimension == 1:
        return dde.geometry.Interval(-1.0, 1.0)
    return dde.geometry.Hypercube([-1.0] * dimension, [1.0] * dimension)


def optimize_with_pinn_impl(calculator, cfg, deci_list, init_params, mse_func=None):
    """Fit PiX unknown quantities with a DeepXDE inverse PINN."""
    del deci_list
    started = time.perf_counter()
    config = _read_config(cfg)
    dde, paddle = _import_backend(config["device"])

    if not calculator.sp_equation:
        calculator.get_sp_equation()
    if not calculator.sp_equation:
        raise ValueError("PINN requires at least one parsed PiX equation")

    coefficient_names = list(calculator.sp_unknown_quantities)
    init_params = np.asarray(init_params, dtype=np.float64)
    if len(coefficient_names) != len(init_params):
        raise ValueError(
            f"PINN coefficient mismatch: {len(coefficient_names)} symbols, "
            f"{len(init_params)} initial values"
        )
    if not coefficient_names:
        loss = float(mse_func(init_params)) if mse_func is not None else 0.0
        return {
            "x": init_params,
            "fun": loss,
            "pinn_loss": loss,
            "nit": 0,
            "status": "NoParameters",
            "time": time.perf_counter() - started,
        }

    data_loader = calculator.data_loader
    coordinate_names = data_loader.spatial_vars + data_loader.temporal_vars
    if len(data_loader.grids) != len(coordinate_names):
        raise ValueError(
            f"PINN grid mismatch: {len(data_loader.grids)} grids for "
            f"{len(coordinate_names)} coordinates"
        )
    if data_loader.u.shape[-1] != len(data_loader.field_vars):
        raise ValueError("PINN field axis does not match Calculator field variables")

    dde.config.set_random_seed(config["seed"])
    coordinates, centers, half_ranges = _normalize_coordinates(data_loader.grids)
    normalized_fields, field_means, field_scales, indices = _normalize_fields(
        data_loader.u, config["max_points"], config["seed"]
    )
    observation_coordinates = coordinates[indices]
    residual_scales = _residual_scales(calculator, init_params)
    coefficients = {
        name: dde.Variable(float(value))
        for name, value in zip(coefficient_names, init_params)
    }

    def pde(normalized_coordinates, normalized_outputs):
        fields = {
            name: float(field_means[index]) + float(field_scales[index])
            * normalized_outputs[:, index:index + 1]
            for index, name in enumerate(data_loader.field_vars)
        }
        evaluator = _PaddleExpressionEvaluator(
            dde, paddle, calculator, normalized_coordinates, fields, coefficients,
            centers, half_ranges,
        )
        return [
            evaluator.evaluate(expression) / scale
            for expression, scale in zip(calculator.sp_equation, residual_scales)
        ]

    observations = [
        dde.icbc.PointSetBC(
            observation_coordinates,
            normalized_fields[indices, index:index + 1],
            component=index,
        )
        for index in range(len(data_loader.field_vars))
    ]
    geometry = _make_geometry(dde, len(coordinate_names))
    data = dde.data.PDE(
        geometry,
        pde,
        observations,
        num_domain=0,
        anchors=observation_coordinates,
    )
    network = dde.nn.FNN(
        [len(coordinate_names)]
        + [config["neurons"]] * config["hidden_layers"]
        + [len(data_loader.field_vars)],
        "tanh",
        "Glorot normal",
    )
    model = dde.Model(data, network)
    loss_weights = (
        [float(config["pde_loss_weight"])] * len(calculator.sp_equation)
        + [float(config["data_loss_weight"])] * len(observations)
    )
    model.compile(
        "adam",
        lr=float(config["learning_rate"]),
        loss_weights=loss_weights,
        external_trainable_variables=list(coefficients.values()),
    )
    display_every = config["display_every"] or config["iterations"]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        loss_history, _ = model.train(
            iterations=config["iterations"], display_every=display_every
        )

    fitted = np.asarray(
        [float(coefficients[name]) for name in coefficient_names], dtype=np.float64
    )
    if np.any(~np.isfinite(fitted)):
        raise RuntimeError(f"PINN produced non-finite coefficients: {fitted}")
    final_losses = np.asarray(loss_history.loss_train[-1], dtype=np.float64)
    if np.any(~np.isfinite(final_losses)):
        raise RuntimeError(f"PINN produced non-finite training losses: {final_losses}")
    objective = float(mse_func(fitted)) if mse_func is not None else float(final_losses.sum())
    if not np.isfinite(objective):
        raise RuntimeError(f"PINN coefficients have non-finite PiX objective: {objective}")
    return {
        "x": fitted,
        "fun": objective,
        "pinn_loss": float(final_losses.sum()),
        "pinn_pde_loss": float(final_losses[:len(calculator.sp_equation)].sum()),
        "pinn_data_loss": float(final_losses[len(calculator.sp_equation):].sum()),
        "nit": config["iterations"],
        "status": "Success",
        "time": time.perf_counter() - started,
    }
