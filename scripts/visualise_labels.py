import os
from typing import Annotated

import typer
from rich.console import Console

from scripts.config import concept_dir, processed_data_dir
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import merge_overlapping_spans

console = Console()
app = typer.Typer()


def labelled_passage_to_html(
    labelled_passage: LabelledPassage, labeller_to_color: dict[str, str]
) -> str:
    """
    Convert a LabelledPassage to an HTML string

    Spans will be highlighted in different colors based on the labeller.

    :param LabelledPassage labelled_passage: The LabelledPassage to convert to HTML
    :param dict[str, str] labeller_to_color: A dictionary mapping labeller names to
    their corresponding colors
    :return str: An HTML string representing the LabelledPassage, with <span> tags
    around each span
    """
    sorted_spans = sorted(labelled_passage.spans, key=lambda x: x.start_index)
    highlights = []
    for span in sorted_spans:
        for labeller in span.labellers:
            highlights.append(
                f'<span class="highlight" style="left: {span.start_index}ch; width: {span.end_index - span.start_index}ch; background-color: {labeller_to_color[labeller]};"></span>'
            )

    return f'<div class="passage-container"><div class="passage">{labelled_passage.text}</div>{"".join(highlights)}</div>'


def visualise_labelled_passages_as_html(
    concept: Concept,
    labelled_passages: list[LabelledPassage],
    title: str = "Labelled Passages",
) -> str:
    """
    Turn a list of labelled passages into an HTML string which visualises the labels

    The labelled passages will be displayed in a list, with spans highlighted in
    different colors based on the labellers.

    :param list[LabelledPassage] labelled_passages: The labelled passages to visualise
    """
    wikibase_url = f"{os.environ.get('WIKIBASE_URL')}/wiki/Item:{concept.wikibase_id}"

    all_labeller_names = set(
        labeller
        for labelled_passage in labelled_passages
        for span in labelled_passage.spans
        for labeller in span.labellers
    )
    colors = [
        f"hsl({i * 360 // len(all_labeller_names)}, 70%, 50%)"
        for i in range(len(all_labeller_names))
    ]
    labeller_to_color = {
        labeller: color for labeller, color in zip(all_labeller_names, colors)
    }

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="initial-scale=1.0">
        <title>{title} - {concept.preferred_label}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0 auto;
                padding: 20px;
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
            .legend {{
                display: flex;
                margin-bottom: 20px;
                border-top: 1px solid #ccc;
                padding-top: 10px;
                border-bottom: 1px solid #ccc;
                padding-bottom: 10px;
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
    <div>
        <h1>{title} - <a href="{wikibase_url}">{concept}</a></h1>
        {f"<p>{concept.description}</p>" or ""}
    </div>
        <div class="legend">
            <b>Labellers:</b>
            {
                "".join(
                    (
                        f'<div class="legend-item"><div class="legend-color" style="background-color: {color};"></div>'
                        f'<span>{labeller.capitalize()}</span></div>' 
                    )
                for labeller, color in labeller_to_color.items())       
            }
        </div>
        <ul>
            {"".join(f"<li>{i+1}. {labelled_passage_to_html(labelled_passage, labeller_to_color)}</li>" for i, labelled_passage in enumerate(labelled_passages))}
        </ul>
    </body>
    </html>
    """
    return " ".join(html.split())


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to train",
            parser=WikibaseID,
        ),
    ],
):
    concept = Concept.load(concept_dir / f"{wikibase_id}.json")

    console.log(f"Visualising labels for {concept}")

    predictions_dir = processed_data_dir / "predictions"
    console.log(f"Loading predictions from {predictions_dir}")
    try:
        with open(predictions_dir / f"{wikibase_id}.jsonl", "r", encoding="utf-8") as f:
            predictions = [LabelledPassage.model_validate_json(line) for line in f]
        n_annotations = sum([len(entry.spans) for entry in predictions])
        console.log(
            f"Loaded {len(predictions)} positively predicted passages with "
            f"{n_annotations} individual spans"
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No sampled passages found for {wikibase_id}. Please run"
            f"  just sample {wikibase_id}"
        ) from e

    visualisations_dir = processed_data_dir / "visualisations" / wikibase_id
    visualisations_dir.mkdir(parents=True, exist_ok=True)

    console.log("🤝 Visualising inter-annotator agreement")
    html = visualise_labelled_passages_as_html(
        concept=concept,
        labelled_passages=concept.labelled_passages,
        title="Inter-annotator agreement",
    )
    output_path = visualisations_dir / "inter_annotator_agreement.html"
    output_path.write_text(html, encoding="utf-8")
    console.log(f"📄 Saved inter-annotator agreement visualisation to {output_path}")

    console.log("🥇 Creating gold standard labelled passages")
    gold_standard_passages: list[LabelledPassage] = []
    for labelled_passage in concept.labelled_passages:
        merged_spans = merge_overlapping_spans(
            # if there's any overlap between spans, merge them
            spans=labelled_passage.spans,
            jaccard_threshold=0,
        )
        gold_standard_passages.append(
            labelled_passage.model_copy(update={"spans": merged_spans}, deep=True)
        )
    n_annotations = sum([len(entry.spans) for entry in gold_standard_passages])
    console.log(
        f"Created {len(gold_standard_passages)} gold standard passages with "
        f"{n_annotations} individual spans"
    )
    console.log("🤩 Visualising gold standard labels' agreement with model predictions")

    output_path = visualisations_dir / "model_vs_gold_standard.html"
    predictions_and_gold_standard_labels = [
        predicted_passage.model_copy(
            update={"spans": predicted_passage.spans + gold_standard_passage.spans},
            deep=True,
        )
        for predicted_passage, gold_standard_passage in zip(
            predictions, gold_standard_passages
        )
    ]
    html = visualise_labelled_passages_as_html(
        concept=concept,
        labelled_passages=predictions_and_gold_standard_labels,
        title="Model predictions vs Gold-standard labels",
    )
    output_path.write_text(html, encoding="utf-8")
    console.log(f"📄 Saved model comparison visualisation to {output_path}")

    console.log("💯 visualising all model predictions")
    output_path = visualisations_dir / "predictions.html"
    html = visualise_labelled_passages_as_html(
        concept=concept, labelled_passages=predictions, title="All model predictions"
    )
    output_path.write_text(html, encoding="utf-8")
    console.log(f"📄 Saved prediction visualisation to {output_path}")


if __name__ == "__main__":
    app()
