from knowledge_graph.operations.wikibase_export.rules import (
    RULES,
    Issue,
    dangling_reference,
    duplicate_english_label,
    missing_english_label,
    run_rules,
    unmirrored_subconcept,
)
from tests.operations.wikibase_export.helpers import (
    item_claim,
    make_item,
    make_snapshot,
)


def test_missing_english_label():
    snapshot = make_snapshot(make_item("Q1"), make_item("Q2", label="a"))

    assert missing_english_label(snapshot) == [
        Issue(
            rule="missing_english_label",
            severity="warning",
            entity_id="Q1",
            message="has no English label",
        )
    ]


def test_dangling_reference_distinguishes_redirect_from_missing():
    a = make_item("Q1", label="a", description="d", claims=item_claim("P3", "Q2", "Q3"))

    issues = dangling_reference(make_snapshot(a, redirects={"Q2": "Q9"}))

    assert {
        (i.entity_id, "redirect" in i.message, "missing" in i.message) for i in issues
    } == {
        ("Q1", True, False),
        ("Q1", False, True),
    }


def test_dangling_reference_accepts_existing_targets():
    a = make_item("Q1", claims=item_claim("P3", "Q2"))
    b = make_item("Q2")

    assert dangling_reference(make_snapshot(a, b)) == []


def test_unmirrored_subconcept_flags_both_directions():
    child = make_item("Q1", claims=item_claim("P2", "Q2"))
    parent = make_item("Q2")
    lonely_parent = make_item("Q3", claims=item_claim("P1", "Q4"))
    orphan = make_item("Q4")

    issues = unmirrored_subconcept(make_snapshot(child, parent, lonely_parent, orphan))

    assert [(i.entity_id, i.message) for i in issues] == [
        ("Q1", "is a subconcept of Q2, but Q2 does not list it as a subconcept"),
        ("Q3", "has subconcept Q4, but Q4 does not list it as a parent"),
    ]


def test_unmirrored_subconcept_accepts_mirrored_pair():
    child = make_item("Q1", claims=item_claim("P2", "Q2"))
    parent = make_item("Q2", claims=item_claim("P1", "Q1"))

    assert unmirrored_subconcept(make_snapshot(child, parent)) == []


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


def test_rules_ignore_statements_without_a_value():
    item = make_item(
        "Q1",
        label="a",
        description="b",
        claims={
            "P2": [
                {
                    "mainsnak": {
                        "snaktype": "somevalue",
                        "property": "P2",
                        "datatype": "wikibase-item",
                    }
                }
            ],
            "P3": [
                {
                    "mainsnak": {
                        "snaktype": "novalue",
                        "property": "P3",
                        "datatype": "wikibase-item",
                    }
                }
            ],
        },
    )

    issues, failed = run_rules(make_snapshot(item))

    assert issues == []
    assert failed == []


def test_run_rules_isolates_a_raising_rule():
    def broken(snapshot):
        raise RuntimeError("boom")

    issues, failed = run_rules(
        make_snapshot(make_item("Q1")), rules=[broken, missing_english_label]
    )

    assert failed == ["broken"]
    assert [i.rule for i in issues] == ["missing_english_label"]


def test_all_starting_rules_are_warnings():
    snapshot = make_snapshot(make_item("Q1", claims=item_claim("P2", "Q404")))

    issues, failed = run_rules(snapshot)

    assert [rule.__name__ for rule in RULES] == [
        "missing_english_label",
        "dangling_reference",
        "unmirrored_subconcept",
        "duplicate_english_label",
    ]
    assert failed == []
    assert issues and {i.severity for i in issues} == {"warning"}
