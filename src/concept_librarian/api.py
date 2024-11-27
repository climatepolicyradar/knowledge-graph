from collections import Counter, defaultdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.concept_librarian.checks import get_concept_store_issues

app = FastAPI()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/concepts", response_class=HTMLResponse)
def generate_html_report(request: Request, response_class=HTMLResponse):
    """
    Generate an HTML report from the list of issues.

    Args:
        issues: List of ConceptStoreIssue objects
        output_dir: Optional directory to write report to. Defaults to data/processed/concept_librarian/

    Returns:
        Path to the generated HTML report
    """
    issues = get_concept_store_issues()

    issues_by_type = defaultdict(list)
    for issue in issues:
        issues_by_type[issue.issue_type].append(issue)

    type_counts = Counter(issue.issue_type for issue in issues)

    return templates.TemplateResponse(
        "report.html.j2",
        {
            "total_issues": len(issues),
            "type_counts": type_counts,
            "issues_by_type": issues_by_type,
        },
    )

    # # Setup output directory
    # if output_dir is None:
    #     output_dir = Path(__file__).parent.parent / "data/processed/concept_librarian"

    # output_dir.mkdir(parents=True, exist_ok=True)

    # # Write report
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # output_path = output_dir / f"librarian_report_{timestr}.html"
    # output_path.write_text(html_content)

    # return output_path
