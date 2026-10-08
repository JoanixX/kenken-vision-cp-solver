"""KenKen solver: visión computacional + constraint programming (OR-Tools CP-SAT)."""

from .instance import Cage, Instance, InstanceError
from .model import (
    add_reified_cage_constraint,
    build_model,
    build_model_joint,
    build_model_table,
    check_solution,
    compute_allowed_tuples,
    find_solutions,
    has_unique_solution,
    solve,
    solve_joint,
    solve_table,
)

from .pipeline import PipelineResult, extract_structure, read_instance, solve_image

__all__ = [
    "Cage",
    "Instance",
    "InstanceError",
    "PipelineResult",
    "add_reified_cage_constraint",
    "build_model",
    "build_model_joint",
    "build_model_table",
    "check_solution",
    "compute_allowed_tuples",
    "extract_structure",
    "find_solutions",
    "has_unique_solution",
    "read_instance",
    "solve",
    "solve_image",
    "solve_joint",
    "solve_table",
]
