"""
PiX: PDE System Interpretable eXplorer

A user-friendly platform for interpretable PDE system discovery from observational data
and physics prior knowledge. Built on the PhysPDE framework with enhanced usability for
both physicists and AI researchers.

Main components:
    - calculator: Symbolic computation framework for physical quantities
    - hypotheses_tree: Tree structure for managing physics hypotheses
    - data_loader: Data loading and preprocessing utilities
    - methods: Implementation of various discovery algorithms (BFSearch, MCTS, etc.)
"""

__version__ = "1.0.0"

from pix.calculator import Calculator
from pix.hypotheses_tree import HypothesesTree, LightTree
from pix.data_loader import DataLoader

__all__ = ["Calculator", "HypothesesTree", "LightTree", "DataLoader"]
