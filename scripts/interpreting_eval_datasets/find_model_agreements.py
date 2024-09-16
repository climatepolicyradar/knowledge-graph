from pathlib import Path

from rich.console import Console

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig

console = Console(highlight=False)


def labelled_passage_to_html(labelled_passage: LabelledPassage) -> str:
    sorted_spans = sorted(labelled_passage.spans, key=lambda x: x.start_index)
    html = f'<div class="passage">{labelled_passage.text}</div>'
    highlights = []

    for span in sorted_spans:
        label_type = (
            "model"
            if any("Classifier" in labeller for labeller in span.labellers)
            else "human"
        )
        highlights.append(
            f'<span class="highlight {label_type}" style="left: {span.start_index}ch; width: {span.end_index - span.start_index}ch;"></span>'
        )

    return f'<div class="passage-container">{html}{"".join(highlights)}</div>'


config_path = Path("data/config/sectors.yaml")
console.log(f"⚙️ Loading config from {config_path}")
config = SamplingConfig.load(config_path)
console.log("✅ Config loaded")

for wikibase_id in config.wikibase_ids:
    console.log(f"🤔 Processing {wikibase_id}")
    labelled_passages_dir = processed_data_dir / "labelled_passages" / wikibase_id

    labelled_passages_path = labelled_passages_dir / "agreements.json"
    human_labelled_passages = [
        LabelledPassage.model_validate_json(line)
        for line in labelled_passages_path.read_text(encoding="utf-8").splitlines()
    ]
    classifier = Classifier.load(classifier_dir / wikibase_id)

    for labelled_passage in human_labelled_passages:
        model_labels = classifier.predict(labelled_passage.text)
        labelled_passage.spans.extend(model_labels)

    output_path = labelled_passages_dir / wikibase_id / "labels.html"
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="initial-scale=1.0">
        <title>Labelled Passages - {wikibase_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            li {{
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 10px;
                padding: 15px;
                overflow-x: auto;
            }}
            .passage-container {{
                position: relative;
                font-family: monospace;
                white-space: nowrap;
                overflow-x: visible;
            }}
            .passage {{
                position: relative;
                z-index: 1;
                display: inline-block;
            }}
            .highlight {{
                position: absolute;
                height: 1.2em;
                top: 0;
                z-index: 0;
                opacity: 0.3;
            }}
            .highlight.model {{
                background-color: #ff9999;
            }}
            .highlight.human {{
                background-color: #99ff99;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                margin-bottom: 20px;
            }}
            .legend-item {{
                margin: 0 10px;
                display: flex;
                align-items: center;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                margin-right: 5px;
                border: 1px solid #333;
                opacity: 0.3;
            }}
        </style>
    </head>
    <body>
        <h1>Labelled Passages - {wikibase_id}</h1>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #99ff99;"></div>
                <span>Human Label</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ff9999;"></div>
                <span>Model Label</span>
            </div>
        </div>
        <ul>
            {"".join(f"<li>{labelled_passage_to_html(labelled_passage)}</li>" for labelled_passage in human_labelled_passages)}
        </ul>
    </body>
    </html>
    """
    html = " ".join(html.split())
    output_path.write_text(html, encoding="utf-8")
    console.log(f"📝 Wrote labels to {output_path}")

    console.log("✅ Done")
