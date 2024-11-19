"""
Translate parsed documents to English

Takes a directory full of parsed documents, and uses the google cloud translate API to
convert text to English.

The script assumes that the parsed document objects are stored in `data/interim/output`.
The translated documents will be saved in `data/interim/translated`.

You'll need a local file called `google-credentials.json` in the root directory with
a set of Google Cloud API credentials.
"""

import json

from cpr_sdk.models import BaseDocument
from cpr_sdk.parser_models import BaseParserOutput
from navigator_document_parser.translator.translate import translate_parser_output
from rich.console import Console
from rich.progress import track

from scripts.config import interim_data_dir

console = Console()

target_language = "en"

n_translated = 0
parser_output_dir = interim_data_dir / "output"
translated_dir = interim_data_dir / "translated"
translated_dir.mkdir(parents=True, exist_ok=True)
paths = list(parser_output_dir.rglob("*.json"))
console.print(
    f"🤓 Translating {len(paths)} documents in {parser_output_dir} into English"
)

for path in track(
    paths, description="Translating documents", transient=True, console=console
):
    with open(path, encoding="utf-8") as f:
        parser_output_data = json.load(f)
    parser_output = BaseParserOutput(**parser_output_data)
    document = BaseDocument.from_parser_output(parser_output)

    output_path = translated_dir / path.relative_to(parser_output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        continue

    if target_language not in document.languages:
        console.log(f"Translating {path}")
        translated_parser_output = translate_parser_output(
            parser_output, target_language
        )
        n_translated += 1
    else:
        translated_parser_output = parser_output

    output_path.write_text(translated_parser_output.model_dump_json())

console.print(f"✅ Translated {n_translated}/{len(paths)}", style="green")
