from knowledge_graph.operations.wikibase_export.report import (
    diff_issues,
    summary_text,
    thread_texts,
)
from knowledge_graph.operations.wikibase_export.rules import Issue

A = Issue(rule="rule_a", severity="warning", entity_id="Q1", message="a")
B = Issue(rule="rule_a", severity="warning", entity_id="Q2", message="b")
C = Issue(rule="rule_c", severity="warning", entity_id="Q3", message="c")


def test_diff_first_run_everything_is_new():
    diff = diff_issues(None, [A, B])

    assert diff.new == [A, B]
    assert diff.resolved == []
    assert diff.total == 2
    assert diff.changed


def test_diff_unchanged():
    assert not diff_issues([A, B], [B, A]).changed


def test_diff_new_and_resolved():
    diff = diff_issues([A, B], [B, C])

    assert diff.new == [C]
    assert diff.resolved == [A]
    assert diff.total == 2
    assert diff.changed


def test_summary_counts_new_resolved_and_total():
    text = summary_text(diff_issues([A, B], [B, C]), "s3://b/p", advanced=True)

    assert "1 new" in text
    assert "1 resolved" in text
    assert "2 in total" in text
    assert "rule_c: 1" in text
    assert "s3://b/p" in text


def test_summary_says_when_downstream_is_held_back():
    diff = diff_issues(None, [A])

    assert "earlier snapshot" in summary_text(diff, "s3://b/p", advanced=False)
    assert "earlier snapshot" not in summary_text(diff, "s3://b/p", advanced=True)


def test_thread_texts_one_per_rule_with_new_issues():
    texts = thread_texts(diff_issues([B], [A, B, C]), "s3://b/p")

    assert len(texts) == 2
    assert "rule_a" in texts[0] and "• Q1: a" in texts[0] and "Q2" not in texts[0]
    assert "rule_c" in texts[1] and "• Q3: c" in texts[1]


def test_thread_text_caps_at_20_per_rule():
    issues = [
        Issue(rule="r", severity="warning", entity_id=f"Q{i:03}", message="m")
        for i in range(25)
    ]

    (text,) = thread_texts(diff_issues(None, issues), "s3://b/p")

    assert text.count("• ") == 20
    assert "Q019" in text and "Q020" not in text
    assert "5 more" in text and "s3://b/p" in text
