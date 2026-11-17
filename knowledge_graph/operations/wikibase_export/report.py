from collections import Counter, defaultdict

from pydantic import BaseModel

from knowledge_graph.operations.wikibase_export.rules import Issue

MAX_ISSUES_PER_RULE = 20


class IssueDiff(BaseModel):
    """How the issues in a snapshot differ from those last reported."""

    new: list[Issue]
    resolved: list[Issue]
    total: int

    @property
    def changed(self) -> bool:
        """Whether there is anything worth reporting."""
        return bool(self.new or self.resolved)


def diff_issues(previous: list[Issue] | None, current: list[Issue]) -> IssueDiff:
    """Compare a snapshot's issues with the last reported ones, if any."""
    previous_set = set(previous or [])
    current_set = set(current)
    return IssueDiff(
        new=[issue for issue in current if issue not in previous_set],
        resolved=[issue for issue in previous or [] if issue not in current_set],
        total=len(current),
    )


def summary_text(diff: IssueDiff, snapshot_uri: str, advanced: bool) -> str:
    """The headline message for a run whose issues changed."""
    lines = [
        "*Concept store export*: "
        f"{len(diff.new)} new, {len(diff.resolved)} resolved, "
        f"{diff.total} in total"
    ]
    for rule, count in sorted(Counter(issue.rule for issue in diff.new).items()):
        lines.append(f"• {rule}: {count} new")
    if not advanced:
        lines.append(
            "Downstream is still reading an earlier snapshot: "
            "the current one is pinned, or this export has errors."
        )
    lines.append(f"Snapshot: {snapshot_uri}")
    return "\n".join(lines)


def thread_texts(diff: IssueDiff, snapshot_uri: str) -> list[str]:
    """One message per rule listing its new issues, capped to keep Slack readable."""
    by_rule: dict[str, list[Issue]] = defaultdict(list)
    for issue in diff.new:
        by_rule[issue.rule].append(issue)

    texts = []
    for rule, issues in sorted(by_rule.items()):
        issues = sorted(issues, key=lambda issue: issue.entity_id)
        lines = [f"*{rule}*"]
        lines.extend(
            f"• {issue.entity_id}: {issue.message}"
            for issue in issues[:MAX_ISSUES_PER_RULE]
        )
        if len(issues) > MAX_ISSUES_PER_RULE:
            lines.append(
                f"and {len(issues) - MAX_ISSUES_PER_RULE} more in "
                f"{snapshot_uri}/issues.jsonl"
            )
        texts.append("\n".join(lines))
    return texts
