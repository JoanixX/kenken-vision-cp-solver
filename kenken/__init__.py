"""KenKen solver: visión computacional + constraint programming (OR-Tools CP-SAT)."""

from .instance import Cage, Instance, InstanceError
from .model import build_model, check_solution, find_solutions, has_unique_solution, solve

__all__ = ["Cage", "Instance", "InstanceError", "build_model", "check_solution",
           "find_solutions", "has_unique_solution", "solve"]
