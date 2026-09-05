"""Unit tests for read-only MediAssist database access."""

import unittest

from app.repositories.mediassist_repository import (
    UnsafeQueryError,
    execute_read_only_query,
)


# This test class verifies both useful reads and rejected unsafe SQL.
class ExecuteReadOnlyQueryTests(unittest.TestCase):
    # This test verifies SQLite rows are returned as ordinary Python dictionaries.
    def test_returns_claim_rows_as_dictionaries(self) -> None:
        rows = execute_read_only_query(
            "SELECT claim_id, status FROM claims ORDER BY claim_id LIMIT 2"
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"claim_id", "status"})

    # This test verifies the repository permits each assignment database table.
    def test_allows_both_assignment_tables(self) -> None:
        rows = execute_read_only_query(
            "SELECT category, COUNT(*) AS ticket_count "
            "FROM maintenance_tickets GROUP BY category LIMIT 1"
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("category", rows[0])
        self.assertIn("ticket_count", rows[0])

    # This test verifies a write command is rejected before SQLite executes it.
    def test_rejects_non_select_statement(self) -> None:
        with self.assertRaisesRegex(UnsafeQueryError, "Only SELECT"):
            execute_read_only_query("DELETE FROM claims")

    # This test verifies only one SQL statement can be submitted at a time.
    def test_rejects_multiple_statements(self) -> None:
        with self.assertRaisesRegex(UnsafeQueryError, "Only one SQL statement"):
            execute_read_only_query("SELECT * FROM claims; SELECT * FROM maintenance_tickets")

    # This test verifies SQLite system tables are outside the approved data scope.
    def test_rejects_unapproved_table(self) -> None:
        with self.assertRaisesRegex(UnsafeQueryError, "unapproved table"):
            execute_read_only_query("SELECT name FROM sqlite_master")

    # This test verifies SQL comments cannot hide extra or misleading query text.
    def test_rejects_sql_comments(self) -> None:
        with self.assertRaisesRegex(UnsafeQueryError, "comments"):
            execute_read_only_query("SELECT * FROM claims -- comment")
