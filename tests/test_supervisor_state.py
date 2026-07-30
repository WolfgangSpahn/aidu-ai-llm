import pytest
from pydantic import ValidationError

from aidu.ai.core.supervisor import SUPERVISOR_DIMENSIONS, SupervisorState


def test_supervisor_dimensions_are_defined_by_the_core_state():
    assert SUPERVISOR_DIMENSIONS == frozenset(SupervisorState.model_fields)


def test_supervisor_prior_is_complete_and_neutral():
    prior = SupervisorState.prior()

    assert all(
        result.fit == 0.5
        and result.reason == "No tutor response has been assessed yet."
        for result in (
            prior.factual_fit,
            prior.goal_alignment,
            prior.knowledge_alignment,
            prior.belief_alignment,
            prior.scaffolding_fit,
        )
    )


def test_supervisor_state_rejects_out_of_range_fit():
    result = {"fit": 0.8, "reason": "It fits the supplied context."}
    state = {dimension: result for dimension in SUPERVISOR_DIMENSIONS}
    state["factual_fit"] = {"fit": 1.2, "reason": "It does not fit."}

    with pytest.raises(ValidationError):
        SupervisorState.model_validate(state)
