# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Actor implementations extending LLMActor with specialized capabilities.
"""

from .mathTutor import MathTutor, StudentInfo, solve_math_problem_with_sympy

__all__ = ["MathTutor", "StudentInfo", "solve_math_problem_with_sympy"]
