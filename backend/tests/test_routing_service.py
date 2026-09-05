"""Unit tests for the replaceable question-routing interface."""

import unittest

from app.services.routing_service import (
    KeywordRoutingStrategy,
    RetrievalType,
    RoutingStrategy,
    route_question,
)


# This test router is a valid replacement implementation that always chooses Hybrid RAG.
class StaticRoutingStrategy(RoutingStrategy):
    """Test double demonstrating that route_question accepts a new strategy."""

    # This method satisfies the abstract interface with a predictable test result.
    def route(self, question: str, role: str) -> RetrievalType:
        return "hybrid_rag"


# This deliberately incomplete class proves an abstract method cannot be skipped.
class IncompleteRoutingStrategy(RoutingStrategy):
    """Intentionally omits route() to verify abstract-interface enforcement."""


# This class has a similar method but intentionally does not extend the interface.
class NotARoutingStrategy:
    """Has a route method but does not explicitly extend the interface."""

    # This method resembles a router but is rejected because the class does not extend it.
    def route(self, question: str, role: str) -> RetrievalType:
        return "hybrid_rag"


# This test class covers the keyword router and its Java-like interface boundary.
class KeywordRoutingStrategyTests(unittest.TestCase):
    # This setup method creates a fresh router before every test method runs.
    def setUp(self) -> None:
        self.strategy = KeywordRoutingStrategy()

    # This test checks that an allowed analytical claim question reaches SQL RAG.
    def test_routes_authorized_analytical_claim_question_to_sql_rag(self) -> None:
        route = self.strategy.route(
            "How many claims were escalated last month?",
            "billing_executive",
        )

        self.assertEqual(route, "sql_rag")

    # This test checks that an allowed analytical ticket question reaches SQL RAG.
    def test_routes_authorized_analytical_ticket_question_to_sql_rag(self) -> None:
        route = self.strategy.route(
            "Which equipment category has the most maintenance tickets?",
            "admin",
        )

        self.assertEqual(route, "sql_rag")

    # This test checks that a procedural question stays with document retrieval.
    def test_routes_procedural_claim_question_to_hybrid_rag(self) -> None:
        route = self.strategy.route(
            "How do I submit a claim?",
            "billing_executive",
        )

        self.assertEqual(route, "hybrid_rag")

    # This test checks that authorization blocks SQL RAG even for an analytical question.
    def test_routes_unauthorized_analytical_question_to_hybrid_rag(self) -> None:
        route = self.strategy.route(
            "How many claims were escalated last month?",
            "doctor",
        )

        self.assertEqual(route, "hybrid_rag")

    # This test checks that a blank question is rejected rather than silently routed.
    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.strategy.route("   ", "admin")

    # This test checks callers can provide a future replacement router through the interface.
    def test_public_function_accepts_a_replacement_strategy(self) -> None:
        route = route_question(
            "How many claims were escalated?",
            "admin",
            strategy=StaticRoutingStrategy(),
        )

        self.assertEqual(route, "hybrid_rag")

    # This test checks the phase-one router explicitly extends the abstract interface.
    def test_keyword_router_explicitly_implements_routing_interface(self) -> None:
        self.assertIsInstance(self.strategy, RoutingStrategy)

    # This test checks Python refuses to instantiate a class missing an abstract method.
    def test_cannot_instantiate_incomplete_routing_implementation(self) -> None:
        with self.assertRaises(TypeError):
            IncompleteRoutingStrategy()

    # This test checks route_question rejects a class that does not extend RoutingStrategy.
    def test_rejects_strategy_that_does_not_extend_interface(self) -> None:
        with self.assertRaisesRegex(TypeError, "extend RoutingStrategy"):
            route_question(
                "How many claims were escalated?",
                "admin",
                strategy=NotARoutingStrategy(),
            )
