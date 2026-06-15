from pydantic import BaseModel, Field


class StudentKnowledge(BaseModel):

    arithmetic: float = 0.2
    fractions: float = 0.1
    equations: float = 0.4
    functions: float = 0.5
    derivatives: float = 0.8
    integrals: float = 0.6

    # ------------------------------------------------------------------
    # Tutor-facing assessment
    # ------------------------------------------------------------------

    def to_tutor_text(self) -> str:

        observations = []

        def describe(name: str, value: float):

            if value < 0.2:
                return f"The student shows severe weaknesses in {name}."

            if value < 0.4:
                return f"The student has limited proficiency in {name}."

            if value < 0.6:
                return f"The student demonstrates partial understanding of {name}."

            if value < 0.8:
                return f"The student is generally competent in {name}."

            return f"The student demonstrates strong mastery of {name}."

        observations.extend(
            [
                describe("arithmetic", self.arithmetic),
                describe("fractions", self.fractions),
                describe("equations", self.equations),
                describe("functions", self.functions),
                describe("derivatives", self.derivatives),
                describe("integrals", self.integrals),
            ]
        )

        # Highlight problematic prerequisite gaps

        if (
            self.derivatives > 0.7
            and self.arithmetic < 0.3
        ):
            observations.append(
                "The student can often apply derivative procedures but struggles with basic arithmetic."
            )

        if (
            self.integrals > 0.7
            and self.fractions < 0.3
        ):
            observations.append(
                "The student can discuss integration concepts but frequently makes errors involving fractions."
            )

        return " ".join(observations)

    def to_student_prompt(self) -> str:

        observations = []

        weak_areas = []
        strong_areas = []

        def weak(area: str):
            weak_areas.append(area)

        def strong(area: str):
            strong_areas.append(area)

        # ------------------------------------------------------------
        # Knowledge assessment
        # ------------------------------------------------------------

        if self.arithmetic < 0.3:
            weak("basic arithmetic calculations")

        elif self.arithmetic > 0.7:
            strong("basic arithmetic calculations")

        if self.fractions < 0.3:
            weak("fractions, ratios, and division")

        elif self.fractions > 0.7:
            strong("fractions and proportional reasoning")

        if self.equations < 0.3:
            weak("solving equations")

        elif self.equations > 0.7:
            strong("solving equations")

        if self.functions < 0.3:
            weak("interpreting functions")

        elif self.functions > 0.7:
            strong("function concepts")

        if self.derivatives < 0.3:
            weak("differentiation")

        elif self.derivatives > 0.7:
            strong("derivative procedures")

        if self.integrals < 0.3:
            weak("integration")

        elif self.integrals > 0.7:
            strong("integration techniques")

        # ------------------------------------------------------------
        # Behavioral constraints
        # ------------------------------------------------------------

        if weak_areas:

            observations.append(
                "The student has not yet mastered the following topics: "
                + ", ".join(weak_areas)
                + "."
            )

            observations.append(
                "The student should not reliably apply methods from these areas without guidance."
            )

            observations.append(
                "The student may be unable to identify appropriate solution strategies involving these topics."
            )

            observations.append(
                "When these topics are required, the student may become confused, make realistic mistakes, or resort to guessing."
            )

        if strong_areas:

            observations.append(
                "The student is generally comfortable with: "
                + ", ".join(strong_areas)
                + "."
            )

        observations.append(
            "The student must behave consistently with this knowledge profile."
        )

        observations.append(
            "Do not suddenly demonstrate knowledge or procedures from weak areas."
        )

        observations.append(
            "Do not solve problems using techniques the student would not realistically know."
        )

        observations.append(
            "If prerequisite skills are weak, mistakes may occur even when discussing more advanced topics."
        )

        observations.append(
            "Responses should reflect what this student would actually know, not what an ideal student should know."
        )

        return " ".join(observations)



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


    def vector(self) -> list[float]:
        return [
            self.engagement,
            self.confidence,
            self.confusion,
            self.frustration,
            self.curiosity,
            self.self_explanation,
            self.guessing,
            self.help_seeking,
        ]

    def to_tutor_text(self) -> str:

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
    def to_student_prompt(self) -> str:

        observations = []

        # ------------------------------------------------------------
        # Engagement
        # ------------------------------------------------------------

        if self.engagement > 0.8:
            observations.append(
                "The student is highly engaged and actively participates in the discussion."
            )
        elif self.engagement < 0.3:
            observations.append(
                "The student is only weakly engaged and may respond briefly or with limited effort."
            )

        # ------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------

        if self.confidence > 0.8:
            observations.append(
                "The student tends to answer confidently and may commit strongly to conclusions."
            )
        elif self.confidence < 0.3:
            observations.append(
                "The student is uncertain and may hesitate, seek reassurance, or ask for clarification."
            )

        # ------------------------------------------------------------
        # Confusion
        # ------------------------------------------------------------

        if self.confusion > 0.8:
            observations.append(
                "The student is significantly confused and may struggle to identify productive next steps."
            )
        elif self.confusion > 0.5:
            observations.append(
                "The student may misunderstand parts of the problem or need additional guidance."
            )

        # ------------------------------------------------------------
        # Frustration
        # ------------------------------------------------------------

        if self.frustration > 0.8:
            observations.append(
                "The student is frustrated and may express discouragement or impatience."
            )
        elif self.frustration > 0.5:
            observations.append(
                "The student may become discouraged when progress is slow."
            )

        # ------------------------------------------------------------
        # Curiosity
        # ------------------------------------------------------------

        if self.curiosity > 0.8:
            observations.append(
                "The student enjoys exploring ideas and often asks conceptual questions."
            )
        elif self.curiosity > 0.5:
            observations.append(
                "The student shows interest in understanding why methods work."
            )

        # ------------------------------------------------------------
        # Self explanation
        # ------------------------------------------------------------

        if self.self_explanation > 0.8:
            observations.append(
                "The student naturally explains reasoning and intermediate thoughts."
            )
        elif self.self_explanation < 0.3:
            observations.append(
                "The student rarely explains reasoning unless explicitly asked."
            )

        # ------------------------------------------------------------
        # Guessing
        # ------------------------------------------------------------

        if self.guessing > 0.8:
            observations.append(
                "The student frequently proposes answers without fully checking them."
            )
        elif self.guessing > 0.5:
            observations.append(
                "The student may occasionally guess when unsure."
            )

        # ------------------------------------------------------------
        # Help seeking
        # ------------------------------------------------------------

        if self.help_seeking > 0.8:
            observations.append(
                "The student readily asks for hints, confirmation, or guidance."
            )
        elif self.help_seeking < 0.3:
            observations.append(
                "The student prefers working independently before asking for help."
            )

        # ------------------------------------------------------------
        # Interaction guidance
        # ------------------------------------------------------------

        observations.append(
            "Respond consistently with these characteristics while remaining realistic and natural."
        )

        observations.append(
            "Do not explicitly mention these characteristics unless they naturally emerge in the conversation."
        )

        return " ".join(observations)

if __name__ == "__main__":
    belief = StudentBelief()
    belief.engagement = 0.8
    belief.confusion = 0.6
    print(belief.to_tutor_text())
    print(belief.to_student_prompt())