# Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
#
# MIT License — see LICENSE file for details.
"""Archetype-driven mathematics student."""

import textwrap

from aidu.ai.agents.math_tutor import MathTutor
from aidu.ai.agents.student import Student
from aidu.ai.llm.agent import UserInput


class MathStudent(Student):
    """A mathematics-specific student simulation."""

    student_knowledge = "* The student knows that x can represent an unknown value."
    student_missing_knowledge = textwrap.dedent("""\
        * Quadratic-equation methods not introduced by the tutor.
        * Factoring and algebraic procedures not introduced by the tutor.
        """)

    prompt_template = textwrap.dedent("""\
        Act as a mathematics student in a classroom simulation. Respond as the
        student, not as a tutor or answer key.

        [MATHEMATICS KNOWLEDGE]
        {student_knowledge}

        [MISSING MATHEMATICS KNOWLEDGE]
        {student_missing_knowledge}

        [ARCHETYPE]
        {narrative_block}

        """).strip()


class MathUserInput(UserInput):
    state_key = MathTutor.__name__
    target = MathStudent
    continuations = []
