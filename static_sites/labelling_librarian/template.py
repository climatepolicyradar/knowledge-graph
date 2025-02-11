from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from static_sites.labelling_librarian.checks import LabellingIssue

# Get the directory where this file lives and load the templates
current_dir = Path(__file__).parent.resolve()
env = Environment(loader=FileSystemLoader(current_dir / "templates"))


def create_index_page(issues: list[LabellingIssue]) -> str:
    """Create an HTML report of all issues found using the Jinja template"""

    return env.get_template("index.html").render(
        issues=issues,
    )
