from knowledge_graph.operations.wikibase_export.rules import (
    RULES,
    Issue,
    dangling_reference,
    duplicate_english_label,
    missing_english_label,
    run_rules,
    unmirrored_subconcept,
)
from knowledge_graph.wikibase import WikibaseSession
from tests.operations.wikibase_export.helpers import (
    item_claim,
    make_item,
    make_snapshot,
)

SUBCONCEPT_OF = WikibaseSession.subconcept_of_property_id
HAS_SUBCONCEPT = WikibaseSession.has_subconcept_property_id


def test_missing_english_label():
    snapshot = make_snapshot(make_item("Q1"), make_item("Q2", label="a"))

    assert missing_english_label(snapshot) == [
        Issue(entity_id="Q1", message="has no English label")
    ]


def test_dangling_reference_distinguishes_redirect_from_missing():
    item = make_item("Q1", claims=item_claim("P3", "Q2", "Q3"))

    issues = dangling_reference(make_snapshot(item, redirects={"Q2": "Q9"}))

    assert [(i.entity_id, i.message) for i in issues] == [
        ("Q1", "points at Q2, which is a redirect to Q9"),
        ("Q1", "points at Q3, which is missing"),
    ]


def test_dangling_reference_reports_each_target_once():
    item = make_item("Q1", claims=item_claim("P3", "Q2") | item_claim("P4", "Q2"))

    assert len(dangling_reference(make_snapshot(item))) == 1


def test_dangling_reference_accepts_existing_targets():
    a = make_item("Q1", claims=item_claim("P3", "Q2"))
    b = make_item("Q2")

    assert dangling_reference(make_snapshot(a, b)) == []


def test_unmirrored_subconcept_flags_both_directions():
    child = make_item("Q1", claims=item_claim(SUBCONCEPT_OF, "Q2"))
    parent = make_item("Q2")
    lonely_parent = make_item("Q3", claims=item_claim(HAS_SUBCONCEPT, "Q4"))
    orphan = make_item("Q4")

    issues = unmirrored_subconcept(make_snapshot(child, parent, lonely_parent, orphan))

    assert [(i.entity_id, i.message) for i in issues] == [
        ("Q1", "is a subconcept of Q2, but Q2 does not list it as a subconcept"),
        ("Q3", "has subconcept Q4, but Q4 does not list it as a parent"),
    ]


def test_unmirrored_subconcept_accepts_mirrored_pair():
    child = make_item("Q1", claims=item_claim(SUBCONCEPT_OF, "Q2"))
    parent = make_item("Q2", claims=item_claim(HAS_SUBCONCEPT, "Q1"))

    assert unmirrored_subconcept(make_snapshot(child, parent)) == []


def test_unmirrored_subconcept_leaves_missing_targets_to_dangling_reference():
    child = make_item("Q1", claims=item_claim(SUBCONCEPT_OF, "Q404"))
    parent = make_item("Q2", claims=item_claim(HAS_SUBCONCEPT, "Q405"))

    assert unmirrored_subconcept(make_snapshot(child, parent)) == []


def test_unmirrored_subconcept_follows_merged_parents():
    # Merging Q2 into Q5 moves its subconcept link but leaves Q1 pointing at Q2
    child = make_item("Q1", claims=item_claim(SUBCONCEPT_OF, "Q2"))
    merged_into = make_item("Q5", claims=item_claim(HAS_SUBCONCEPT, "Q1"))

    snapshot = make_snapshot(child, merged_into, redirects={"Q2": "Q5"})

    assert unmirrored_subconcept(snapshot) == []


def test_unmirrored_subconcept_follows_merged_children():
    # Merging Q2 into Q5 moves its parent link but leaves Q3 listing Q2
    merged_into = make_item("Q5", claims=item_claim(SUBCONCEPT_OF, "Q3"))
    parent = make_item("Q3", claims=item_claim(HAS_SUBCONCEPT, "Q2"))

    snapshot = make_snapshot(merged_into, parent, redirects={"Q2": "Q5"})

    assert unmirrored_subconcept(snapshot) == []


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


def test_rules_ignore_statements_without_an_item_target():
    item = make_item(
        "Q1",
        label="a",
        claims={
            "P2": [
                {"mainsnak": {"snaktype": "somevalue", "datatype": "wikibase-item"}}
            ],
            "P3": [{"mainsnak": {"snaktype": "novalue", "datatype": "wikibase-item"}}],
            "P4": [
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datatype": "string",
                        "datavalue": {"value": "a definition", "type": "string"},
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


def test_run_rules_names_issues_after_their_rule():
    def unnamed(snapshot):
        return [Issue(entity_id="Q1", message="m")]

    issues, _ = run_rules(make_snapshot(), rules=[unnamed])

    assert issues == [
        Issue(rule="unnamed", severity="warning", entity_id="Q1", message="m")
    ]


def test_all_starting_rules_are_warnings():
    snapshot = make_snapshot(
        make_item("Q1", claims=item_claim(SUBCONCEPT_OF, "Q2", "Q404")),
        make_item("Q2", label="a"),
        make_item("Q3", label="a"),
    )

    issues, failed = run_rules(snapshot)

    assert failed == []
    assert {i.rule for i in issues} == {rule.__name__ for rule in RULES}
    assert {i.severity for i in issues} == {"warning"}
