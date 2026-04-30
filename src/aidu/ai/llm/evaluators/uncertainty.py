"""
Uncertainty evaluator for assessing student confidence calibration in educational dialogues.
"""

import textwrap
import logging

from ..evaluator import Evaluator

logger = logging.getLogger(__name__)


class UncertaintyEvaluator(Evaluator):
    """
    Evaluates whether student's confidence is well-calibrated in their last turn.
    
    Returns a distribution over:
    - Very well-calibrated (confident on correct, uncertain on hard)
    - Well-calibrated
    - Neutral
    - Poorly-calibrated
    - Very poorly-calibrated (overconfident on wrong answers)
    """

    likert_labels = [
        "Very well-calibrated",
        "Well-calibrated",
        "Neutral",
        "Poorly-calibrated",
        "Very poorly-calibrated"
    ]

    system_prompt = textwrap.dedent("""\
        Evaluate how well-calibrated the student's confidence is in their response.
        
        Context: {context}
        Student response: {text}
        Correct answer: {correct_answer}
        
        Assign probabilities across the Likert scale:
        - Very well-calibrated: Student's confidence perfectly matches correctness
        - Well-calibrated: Confidence mostly matches correctness with minor misalignment
        - Neutral: Confidence and correctness have mixed or unclear relationship
        - Poorly-calibrated: Noticeable misalignment between confidence and correctness
        - Very poorly-calibrated: Strong overconfidence on wrong answers
        
        Consider:
        - If answer mentions multiple options including the correct one → Lean towards well-calibrated
        - If answer uses hedging language ("maybe", "possibly", "not sure") → Appropriate caution
        - If answer is confident but wrong → Lean towards poorly-calibrated
        - If answer is certain and correct → Lean towards very well-calibrated
        
        Return a distribution with non-zero probabilities showing your confidence across all categories.
        Respond with JSON:
        {{"distribution": [<very_well>, <well>, <neutral>, <poorly>, <very_poorly>]}}
        """).strip()


def run_smoke_test():
    """Smoke test for UncertaintyEvaluator."""
    from dotenv import load_dotenv
    import os
    from ..client import LLMClient
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Missing OPENAI_API_KEY in .env"
    
    console = Console()
    client = LLMClient(api_key)
    evaluator = UncertaintyEvaluator(client)
    
    # Test cases with clear field names
    test_cases = [
        {
            "tutor_message": "Let's work through differentiation.",
            "student_message": "I think the derivative of 3x^2 is 6x.",
            "correct_answer": "The derivative of 3x^2 is 6x",
            "description": "Well-calibrated (correct and confident)"
        },
        {
            "tutor_message": "Give it a try.",
            "student_message": "I'm not sure, but maybe it could be 2x? Or possibly 6x?",
            "correct_answer": "The derivative of 3x^2 is 6x",
            "description": "Well-calibrated (uncertain with difficult problem)"
        },
        {
            "tutor_message": "How would you find the derivative?",
            "student_message": "Definitely the derivative is 2x^2, I'm 100% sure.",
            "correct_answer": "The derivative of 3x^2 is 6x",
            "description": "Poorly-calibrated (overconfident on wrong answer)"
        },
    ]
    
    console.print("\n[bold cyan]UncertaintyEvaluator Smoke Test[/bold cyan]")
    
    for idx, test_case in enumerate(test_cases, 1):
        # Create test case panel
        test_info = f"""[bold]{test_case['description']}[/bold]
        
[yellow]Tutor:[/yellow] {test_case['tutor_message']}
[cyan]Student:[/cyan] {test_case['student_message']}
[green]Correct:[/green] {test_case['correct_answer']}"""
        
        console.print(Panel(test_info, title=f"Test {idx}", expand=False))
        
        # Evaluate student right after their response, using tutor message as history
        distribution = evaluator.evaluate(
            eval_params={
                "text": test_case['student_message'],
                "context": test_case['tutor_message'],
                "correct_answer": test_case['correct_answer']
            }
        )
        
        if distribution:
            # Create distribution table
            table = Table(title="Distribution", show_header=True)
            table.add_column("Label", style="cyan")
            table.add_column("Probability", style="green")
            table.add_column("Visualization", style="magenta")
            
            labels = evaluator.likert_labels
            for label, prob in zip(labels, distribution):
                bar = "█" * int(prob * 30)
                table.add_row(label, f"{prob:5.2f}", bar)
            
            console.print(table)
        else:
            console.print("[red]❌ Evaluation failed[/red]")
        
        console.print()
    
    console.print("[bold green]✅ UncertaintyEvaluator smoke test complete![/bold green]\n")


if __name__ == "__main__":
    from rich.logging import RichHandler
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()]
    )
    
    run_smoke_test()
