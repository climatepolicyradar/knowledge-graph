from pathlib import Path

from jinja2 import Environment, FileSystemLoader


# Get the directory where this file lives and load the templates
current_dir = Path(__file__).parent.resolve()
env = Environment(loader=FileSystemLoader(current_dir / "templates"))


def create_index_page(dataset_names: list[str]) -> str:
    """Create an HTML report of all issues found using the Jinja template"""

    return env.get_template("index.html").render(
        dataset_names=dataset_names,
    )
