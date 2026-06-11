from pydantic import BaseModel, Field


class StudentBelief(BaseModel):
    """
    Domain-independent estimate of the student's current
    mental and learning state.

    Values are probabilities in the range [0, 1].
    """

    engagement: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confusion: float = Field(default=0.5, ge=0.0, le=1.0)
    frustration: float = Field(default=0.5, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)

    self_explanation: float = Field(default=0.5, ge=0.0, le=1.0)
    guessing: float = Field(default=0.5, ge=0.0, le=1.0)
    help_seeking: float = Field(default=0.5, ge=0.0, le=1.0)


    def to_text(self) -> str:

        observations = []

        if self.engagement > 0.7:
            observations.append(
                "The student appears actively engaged."
            )

        if self.curiosity > 0.7:
            observations.append(
                "The student shows strong curiosity."
            )

        if self.confidence < 0.5:
            observations.append(
                "The student's confidence appears limited."
            )

        if self.confusion > 0.5:
            observations.append(
                "Some confusion is likely."
            )

        if self.frustration > 0.5:
            observations.append(
                "The student may be experiencing frustration."
            )

        if self.help_seeking > 0.6:
            observations.append(
                "The student readily seeks guidance."
            )

        if self.self_explanation < 0.5:
            observations.append(
                "The student rarely explains concepts in their own words."
            )

        if self.guessing > 0.5:
            observations.append(
                "Some responses may be based on guessing."
            )

        return " ".join(observations)
    

if __name__ == "__main__":
    belief = StudentBelief()
    belief.engagement = 0.8
    belief.confusion = 0.6
    print(belief.to_text())