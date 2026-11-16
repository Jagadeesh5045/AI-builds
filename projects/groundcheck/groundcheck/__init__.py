"""groundcheck: groundedness and citation checking for RAG answers."""
from .checker import check_answer, ClaimResult, AnswerReport
__all__ = ["check_answer", "ClaimResult", "AnswerReport"]
__version__ = "1.0.0"
