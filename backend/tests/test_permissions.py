"""Unit tests for SQL-RAG authorization policy."""

import unittest

from app.core.permissions import can_use_sql_rag


# This test class groups checks for the SQL-RAG role policy.
class CanUseSqlRagTests(unittest.TestCase):
    # This test confirms the billing role is permitted to use SQL RAG.
    def test_billing_executive_can_use_sql_rag(self) -> None:
        self.assertTrue(can_use_sql_rag("billing_executive"))

    # This test confirms the admin role is permitted to use SQL RAG.
    def test_admin_can_use_sql_rag(self) -> None:
        self.assertTrue(can_use_sql_rag("admin"))

    # This test confirms roles without analytical access are rejected.
    def test_non_analytical_roles_cannot_use_sql_rag(self) -> None:
        for role in ("doctor", "nurse", "technician", "unknown"):
            with self.subTest(role=role):
                self.assertFalse(can_use_sql_rag(role))
