"""
Discovery algorithm implementations for PiX.

This module contains various algorithms for discovering PDE systems:
    - BFSearch: Breadth-first search with symbolic regression
    - MCTS: Monte Carlo Tree Search
    - SINDy: Sparse Identification of Nonlinear Dynamics
    - Adjoint: Adjoint-based optimization methods
"""

__all__ = [
    'BFSearch',
    'MCTS',
    'sindy',
    'adjoint',
]
