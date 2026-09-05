"""Terminal entry point for learning and manually testing the SQL-RAG workflow."""

import argparse

from app.services.llm.llm_client import create_llm_client
from app.services.sql_rag_service import SQLRagService


# This function defines the terminal arguments used to submit one SQL-RAG question.
def _parse_arguments() -> argparse.Namespace:
    """Read the question and authorized role supplied on the command line."""
    parser = argparse.ArgumentParser(description="Run one MediBot SQL-RAG question.")
    parser.add_argument("question", help="Natural-language analytical question.")
    parser.add_argument(
        "--role",
        required=True,
        help="User role; only billing_executive and admin may use SQL RAG.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print the generated SQL and read-only database rows before the answer.",
    )
    return parser.parse_args()


# This function runs the terminal command while keeping FastAPI out of the current learning phase.
def main() -> None:
    """Run SQL-RAG from a terminal and print the requested result."""
    arguments = _parse_arguments()
    service = SQLRagService(create_llm_client())
    result = service.answer_with_details(arguments.question, arguments.role)

    if arguments.show_details:
        print(f"Generated SQL:\n{result.sql}\n")
        print(f"Database rows:\n{result.rows}\n")
    print(f"Answer:\n{result.answer}")


# This guard runs main only when Python starts this module as a command.
if __name__ == "__main__":
    main()
