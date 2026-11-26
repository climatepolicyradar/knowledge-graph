from knowledge_graph.operations.wikibase_export.rules import (
    RULES,
    Issue,
    duplicate_english_label,
    missing_english_label,
    run_rules,
)
from tests.operations.wikibase_export.helpers import make_item, make_snapshot


def test_missing_english_label():
    snapshot = make_snapshot(make_item("Q1"), make_item("Q2", label="a"))

    assert missing_english_label(snapshot) == [
        Issue(entity_id="Q1", message="has no English label")
    ]


def test_duplicate_english_label_ignores_case_and_whitespace():
    snapshot = make_snapshot(
        make_item("Q1", label="Solar power"),
        make_item("Q2", label=" solar POWER "),
        make_item("Q3", label="Wind power"),
    )

    issues = duplicate_english_label(snapshot)

    assert [(i.entity_id, i.message) for i in issues] == [
        ("Q1", "shares its English label with Q2"),
        ("Q2", "shares its English label with Q1"),
    ]


def test_run_rules_isolates_a_raising_rule():
    def broken(snapshot):
        raise RuntimeError("boom")

    issues, failed = run_rules(
        make_snapshot(make_item("Q1")), rules=[broken, missing_english_label]
    )

    assert failed == ["broken"]
    assert [i.rule for i in issues] == ["missing_english_label"]


def test_run_rules_names_issues_after_their_rule():
    def unnamed(snapshot):
        return [Issue(entity_id="Q1", message="m")]

    issues, _ = run_rules(make_snapshot(), rules=[unnamed])

    assert issues == [
        Issue(rule="unnamed", severity="warning", entity_id="Q1", message="m")
    ]


def test_all_starting_rules_are_warnings():
    snapshot = make_snapshot(
        make_item("Q1"), make_item("Q2", label="a"), make_item("Q3", label="a")
    )

    issues, failed = run_rules(snapshot)

    assert failed == []
    assert {i.rule for i in issues} == {rule.__name__ for rule in RULES}
    assert {i.severity for i in issues} == {"warning"}
