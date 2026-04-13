"""
test_conflicts.py -- Layer 3: Conflict detection benchmarks.

Tests the 4-type conflict classification system:
- temporal_succession
- implicit_supersede
- opinion_change
- hard_contradiction
- no_conflict (non-conflicting pairs)
"""

import pytest

from .metrics import ClassificationMetrics, per_class_accuracy


# ===========================================================================
# Test Data -- conflict pairs
# ===========================================================================

def _make_fact(subject, predicate, obj, since="", source=""):
    """Helper to create a Fact object."""
    from engram.conflicts import Fact
    return Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        since=since,
        source=source,
    )


CONFLICT_TEST_CASES = [
    # -- temporal_succession: both have dates, new is later --
    {
        "old": ("saas-app", "uses_database", "MySQL", "2025-01"),
        "new": ("saas-app", "uses_database", "PostgreSQL", "2025-03"),
        "source_text": "We now use PostgreSQL.",
        "expected_type": "temporal_succession",
    },
    {
        "old": ("saas-app", "uses_auth", "Auth0", "2024-06"),
        "new": ("saas-app", "uses_auth", "Clerk", "2025-01"),
        "source_text": "Using Clerk now.",
        "expected_type": "temporal_succession",
    },
    {
        "old": ("saas-app", "deployed_on", "Heroku", "2024-12"),
        "new": ("saas-app", "deployed_on", "Railway", "2025-02"),
        "source_text": "Deployed to Railway.",
        "expected_type": "temporal_succession",
    },
    {
        "old": ("saas-app", "uses_framework", "Express.js", "2024-08"),
        "new": ("saas-app", "uses_framework", "FastAPI", "2025-02"),
        "source_text": "Built with FastAPI.",
        "expected_type": "temporal_succession",
    },
    {
        "old": ("frontend", "status", "alpha", "2025-01"),
        "new": ("frontend", "status", "beta", "2025-04"),
        "source_text": "Frontend is now in beta.",
        "expected_type": "temporal_succession",
    },

    # -- implicit_supersede: replacement language --
    {
        "old": ("saas-app", "uses_database", "MySQL", ""),
        "new": ("saas-app", "uses_database", "PostgreSQL", ""),
        "source_text": "We switched from MySQL to PostgreSQL.",
        "expected_type": "implicit_supersede",
    },
    {
        "old": ("saas-app", "uses_auth", "Auth0", ""),
        "new": ("saas-app", "uses_auth", "Clerk", ""),
        "source_text": "Migrated from Auth0 to Clerk for auth.",
        "expected_type": "implicit_supersede",
    },
    {
        "old": ("saas-app", "uses_database", "Elasticsearch", ""),
        "new": ("saas-app", "uses_database", "Meilisearch", ""),
        "source_text": "Replaced Elasticsearch with Meilisearch.",
        "expected_type": "implicit_supersede",
    },
    {
        "old": ("saas-app", "deployed_on", "Heroku", ""),
        "new": ("saas-app", "deployed_on", "Railway", ""),
        "source_text": "We moved from Heroku to Railway.",
        "expected_type": "implicit_supersede",
    },
    {
        "old": ("frontend", "uses_framework", "Redux", ""),
        "new": ("frontend", "uses_framework", "Zustand", ""),
        "source_text": "Dropped Redux and switched to Zustand.",
        "expected_type": "implicit_supersede",
    },
    {
        "old": ("ci", "uses_framework", "CircleCI", ""),
        "new": ("ci", "uses_framework", "GitHub Actions", ""),
        "source_text": "Migrating from CircleCI to GitHub Actions.",
        "expected_type": "implicit_supersede",
    },
    {
        "old": ("caching", "uses_framework", "Memcached", ""),
        "new": ("caching", "uses_framework", "Redis", ""),
        "source_text": "We stopped using Memcached and changed to Redis.",
        "expected_type": "implicit_supersede",
    },

    # -- opinion_change: preference predicates --
    {
        "old": ("Alex", "prefers", "JavaScript", ""),
        "new": ("Alex", "prefers", "TypeScript", ""),
        "source_text": "Alex now prefers TypeScript.",
        "expected_type": "opinion_change",
    },
    {
        "old": ("Maya", "favorite", "React", ""),
        "new": ("Maya", "favorite", "Svelte", ""),
        "source_text": "Maya changed her favorite framework to Svelte.",
        "expected_type": "opinion_change",
    },
    {
        "old": ("Jordan", "prefers", "Terraform", ""),
        "new": ("Jordan", "prefers", "Pulumi", ""),
        "source_text": "Jordan now prefers Pulumi.",
        "expected_type": "opinion_change",
    },

    # -- hard_contradiction: no dates, no replacement language, not opinion --
    {
        "old": ("saas-app", "uses_database", "MySQL", ""),
        "new": ("saas-app", "uses_database", "PostgreSQL", ""),
        "source_text": "We use PostgreSQL.",
        "expected_type": "hard_contradiction",
    },
    {
        "old": ("saas-app", "uses_auth", "Auth0", ""),
        "new": ("saas-app", "uses_auth", "Clerk", ""),
        "source_text": "The app uses Clerk for auth.",
        "expected_type": "hard_contradiction",
    },
    {
        "old": ("api", "status", "stable", ""),
        "new": ("api", "status", "broken", ""),
        "source_text": "The API is broken right now.",
        "expected_type": "hard_contradiction",
    },
    {
        "old": ("saas-app", "led_by", "Alice", ""),
        "new": ("saas-app", "led_by", "Bob", ""),
        "source_text": "Bob is leading the project.",
        "expected_type": "hard_contradiction",
    },
    {
        "old": ("backend", "managed_by", "Alex", ""),
        "new": ("backend", "managed_by", "Sarah", ""),
        "source_text": "Sarah manages the backend now.",
        "expected_type": "hard_contradiction",
    },
]


class TestConflictClassification:
    """Test conflict type classification accuracy."""

    def test_classification_accuracy(self):
        """Test that conflicts are classified correctly."""
        from engram.conflicts import classify_conflict

        predictions = []
        labels = []
        details = []

        for i, case in enumerate(CONFLICT_TEST_CASES):
            old_fact = _make_fact(*case["old"])
            new_fact = _make_fact(*case["new"])

            predicted = classify_conflict(
                old_fact=old_fact,
                new_fact=new_fact,
                source_text=case["source_text"],
            )

            predictions.append(predicted)
            labels.append(case["expected_type"])

            if predicted != case["expected_type"]:
                details.append(
                    f"  [{i}] {case['old'][0]}.{case['old'][1]}: "
                    f"expected={case['expected_type']}, got={predicted} "
                    f"(source: '{case['source_text'][:40]}...')"
                )

        # Per-class accuracy
        class_acc = per_class_accuracy(predictions, labels)
        total_correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        overall_acc = total_correct / len(predictions) if predictions else 0

        print(f"\nConflict classification:")
        print(f"  Overall accuracy: {overall_acc:.1%} ({total_correct}/{len(predictions)})")
        for cls, stats in sorted(class_acc.items()):
            print(f"  {cls}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        if details:
            print("  Misclassified:")
            for line in details:
                print(line)

        # Overall accuracy should be at least 75%
        assert overall_acc >= 0.70, (
            f"Conflict classification accuracy is {overall_acc:.1%} -- below 70% threshold"
        )

    def test_conflict_detection(self):
        """Test that check_conflict correctly identifies conflicts on exclusive predicates."""
        from engram.conflicts import check_conflict

        # Exclusive predicate: same subject+predicate, different object -> conflict
        old_fact = _make_fact("saas-app", "uses_database", "MySQL")
        new_fact = _make_fact("saas-app", "uses_database", "PostgreSQL")

        conflict = check_conflict(new_fact, [old_fact])
        assert conflict is not None, "Should detect conflict on exclusive predicate"
        assert conflict.old_fact.object == "MySQL"
        assert conflict.new_fact.object == "PostgreSQL"

    def test_non_exclusive_no_conflict(self):
        """Non-exclusive predicates should not trigger conflicts."""
        from engram.conflicts import check_conflict

        # Non-exclusive: someone can love multiple things
        old_fact = _make_fact("Alex", "loves", "chess")
        new_fact = _make_fact("Alex", "loves", "swimming")

        conflict = check_conflict(new_fact, [old_fact])
        assert conflict is None, "Non-exclusive predicate should not conflict"

    def test_same_value_no_conflict(self):
        """Same subject+predicate+object should not be a conflict (it's a duplicate)."""
        from engram.conflicts import check_conflict

        old_fact = _make_fact("saas-app", "uses_database", "PostgreSQL")
        new_fact = _make_fact("saas-app", "uses_database", "PostgreSQL")

        conflict = check_conflict(new_fact, [old_fact])
        assert conflict is None, "Same value should not be a conflict"

    def test_resolution_types(self):
        """Test that each conflict type resolves correctly."""
        from engram.conflicts import Conflict, resolve_conflict

        # temporal_succession -> new_wins
        c1 = Conflict(
            old_fact=_make_fact("app", "db", "MySQL", since="2025-01"),
            new_fact=_make_fact("app", "db", "Postgres", since="2025-03"),
            conflict_type="temporal_succession",
        )
        r1 = resolve_conflict(c1)
        assert r1["resolved"] is True
        assert r1["action"] == "expire_old"

        # implicit_supersede -> new_wins with lower confidence
        c2 = Conflict(
            old_fact=_make_fact("app", "db", "MySQL"),
            new_fact=_make_fact("app", "db", "Postgres"),
            conflict_type="implicit_supersede",
        )
        r2 = resolve_conflict(c2)
        assert r2["resolved"] is True
        assert r2["new_fact_updates"]["confidence"] == 0.85

        # hard_contradiction -> unresolved
        c3 = Conflict(
            old_fact=_make_fact("app", "db", "MySQL"),
            new_fact=_make_fact("app", "db", "Postgres"),
            conflict_type="hard_contradiction",
        )
        r3 = resolve_conflict(c3)
        assert r3["resolved"] is False
        assert r3["action"] == "defer"
