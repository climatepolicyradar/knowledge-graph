import os
import random
import re
from typing import Optional

from anthropic import Anthropic
from lxml import etree

from src.classifier import Classifier
from src.concept import Concept
from src.identifiers import generate_identifier
from src.labelled_passage import LabelledPassage
from src.span import Span
from src.wikibase import WikibaseSession


class LLMClassifier(Classifier):
    """A classifier that uses an LLM to predict the presence of a concept in a text."""

    def __init__(
        self,
        concept: Concept,
        model: str = "claude-3-5-haiku-20241022",
        wikibase: Optional[WikibaseSession] = None,
    ):
        self.concept = concept
        self.concept_description = concept.to_markdown(wikibase=wikibase)
        self.wikibase = wikibase
        default_examples = """<input>
        <passage id="a8h3hcxj">I worked on a horse farm when I was a kid. I was in charge of feeding the ponies.</passage>
        <passage id="ykq2kk5a">I love horses. I have a horse named Bessie.</passage>
        <passage id="fh7akadd">I have a pet dog.</passage>
        </input>

        <output>
        <passage id="a8h3hcxj">I worked on a <concept>horse</concept> farm when I was a kid. I was in charge of feeding the <concept>ponies</concept>.</passage>
        <passage id="ykq2kk5a">I love <concept>horses</concept>. I have a <concept>horse</concept> named Bessie.</passage>
        <passage id="fh7akadd">I have a pet dog.</passage>
        </output>
        """

        self.examples = (
            self._create_labelled_examples(self.concept.labelled_passages)
            if self.concept.labelled_passages
            else default_examples
        )

        self.prompt_template = """
        You are a specialist climate policy analyst, tasked with identifying mentions of the "{PREFERRED_LABEL}" concept in climate policy documents. Your task is to read passages from real climate policy documents and mark up any references to the concept of "{PREFERRED_LABEL}" with specific XML tags.

        You will be paid $0.10 for each individual mention of the concept which you annotate correctly, but will be docked $0.05 for each mention of the concept which you miss or annotate incorrectly. Please take your time and be accurate in your judgements. Individual mistakes can add up to be costly, so make sure you tag every mention of the concept that you're sure of!

        First, carefully review the following description of the concept of "{PREFERRED_LABEL}":

        <concept_description>
        {CONCEPT_DESCRIPTION}
        </concept_description>

        You will be given a set of passages from real climate policy documents. Your task is to identify any instances of "{PREFERRED_LABEL}" in these passages and mark them with <concept> tags.
        
        Instructions:
        1. Read through each passage carefully, thinking about the concept of "{PREFERRED_LABEL}" as described in the concept description.
        2. Identify any mentions of the concept of "{PREFERRED_LABEL}", including direct references and related terms, as given in the concept description.
        3. In your response, you should reproduce the input passages exactly, surrounding each identified mention of the concept within each passage with a set of <concept> tags.
        4. If a passage contains multiple instances of the concept, you should surround each one with a <concept> tag. This is important - you will be docked $0.05 for each mention of the concept which you miss or annotate incorrectly, so make sure you tag everything you're sure of!
        5. If a passage does not contain any instances of the concept, you should reproduce the original passage in the output exactly as it was given to you, without any <concept> tags.
        6. If an entire passage refers to the concept without mentioning it by name in any specific tokens, you should surround the entire passage with a <concept> tag.
        7. Your output must maintain the input structure of the passages, using the same unique identifiers as attached to each passage in the input.
        8. To help your classification, you may use a preceding <concept_identification> section to explain any reasoning which helps you to identify the concept in each passage. It's a scratchpad - the thoughts here will not be used by any of the subsequent pipeline steps, but should help to make your process more coherent. For example, for each passage ID you might include a sentence or two, discussing each of the plausible terms within the passage and justifying your final choice for why each candidate term is/isn't a valid instance of the concept.
        9. CRITICAL: The text in each output <passage> must be EXACTLY identical to the input text, character-for-character, with the only difference being the addition of <concept> tags. This includes punctuation, whitespace, newlines, etc! Double check and re-read the input text, being extra careful to ensure that you haven't added or removed any characters apart from the <concept> tags.
        
        Your response must follow this exact structure:
        <response>
        <concept_identification>
        [Space for reasoning which guides your final output <passage> objects in the <output> tag]
        </concept_identification>
        <output>
        [Your marked-up passages here]
        </output>
        </response>

        Here are some example input/output tags:
        <examples>
        {EXAMPLES}
        </examples>

        Here are the real passages you need to analyse:

        <input>
        {INPUT_PASSAGES}
        </input>

        Begin your analysis now, ensuring your response follows the exact XML structure specified above. Remember, the text in each output <passage> must be EXACTLY identical to the input text, character-for-character, with the only difference being the addition of <concept> tags.
        """

        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _create_labelled_examples(
        self, labelled_passages: list[LabelledPassage]
    ) -> str:
        """Use concept's gold-standard labelled passages to create INPUT and OUTPUT examples for the LLM."""
        positive_examples = [
            labelled_passage
            for labelled_passage in labelled_passages
            if len(labelled_passage.spans) > 0
        ]
        negative_examples = [
            labelled_passage
            for labelled_passage in labelled_passages
            if len(labelled_passage.spans) == 0
        ]
        sampled_examples = random.sample(
            positive_examples, min(2, len(positive_examples))
        ) + random.sample(negative_examples, min(1, len(negative_examples)))

        input_xml_lines = []
        output_xml_lines = []
        for passage in sampled_examples:
            identifier = generate_identifier(passage.text)
            input_xml_lines.append(
                f'<passage id="{identifier}">{passage.text}</passage>'
            )
            output_text = (
                passage.get_highlighted_text()
                .replace("[cyan]", "<concept>")
                .replace("[/cyan]", "</concept>")
            )
            output_xml_lines.append(
                f'<passage id="{identifier}">{output_text}</passage>'
            )
        input_xml = "\n".join(input_xml_lines)
        output_xml = "\n".join(output_xml_lines)
        examples = f"""<input>
        {input_xml}
        </input>

        <output>
        {output_xml}
        </output>
        """

        return examples

    def _get_spans_from_string(self, text: str) -> list[Span]:
        """Get the spans from a string that has been marked up with concept markers."""
        spans = []
        # keep track of the number of characters we've added with concept markers
        offset = 0
        for match in re.finditer(r"<concept>(.*?)</concept>", text):
            start_index = match.start() - (offset * len("<concept></concept>"))
            end_index = start_index + len(match.group(1))
            offset += 1
            spans.append(
                Span(
                    text=text,
                    start_index=start_index,
                    end_index=end_index,
                    concept_id=self.concept.wikibase_id,
                    labellers=[str(self)],
                )
            )
        return spans

    def _sanitise_text_for_xml(self, text: str) -> str:
        """
        Sanitise a text by replacing bad XML characters with filler strings and normalize text.

        XML is very annoying... It's great at adding structure to LLM output, but if
        we give an XML parser any text which contains characters like ["&", "<", ">",
        '"', "'"], it'll throw a xmlParseEntityRef error. Typically, we would just
        escape these characters (eg replacing "&" with "&amp;"). However, these
        escaped versions would change the length of the text, and because we need to
        keep track of the indices of the concept tags that have been added by the LLM,
        we can't take this approach! The escaping and un-escaping introduces so many
        opportunities for the spans to fall out of alignment with the original text.

        Instead, we replace all of the bad characters with a filler character,
        thereby maintaining the original length while allowing the XML to be parsed.
        It's an ugly solution but it works. The only thing we need to be careful about
        is that we construct our final Span objects using the original texts which were
        fed as _input_ to the LLM, rather than taking the sanitised output of the LLM as
        our source of truth.

        Additionally, this method normalizes text by handling common discrepancies like:
        - Converting various whitespace characters (newlines, tabs) to spaces
        - Normalizing Unicode characters like curly quotes, em/en dashes, and ellipses
        while preserving the text length.
        """
        # First handle XML special characters
        bad_xml_strings = ["&", "<", ">", '"', "'"]
        xml_translation = str.maketrans(
            {string: "_" * len(string) for string in bad_xml_strings}
        )
        text = text.translate(xml_translation)

        # Then normalize common Unicode discrepancies and whitespace variations
        normalize_translation = str.maketrans(
            {
                " ": " ",
                "\n": " ",
                "\t": " ",
                "…": "...",
                "'": "'",
                "–": "-",
                "—": "-",
                "’": "'",
                "‘": "'",
                "“": '"',
                "”": '"',
            }
        )
        return text.translate(normalize_translation)

    def generate_prompt(self, text_dicts: dict[str, dict[str, str]]) -> str:
        """Generate the prompt for the LLM."""
        input_xml = "\n\t".join(
            f'<passage id="{identifier}">{text_dict["sanitised"]}</passage>'
            for identifier, text_dict in text_dicts.items()
        )

        prompt = self.prompt_template.format(
            PREFERRED_LABEL=self.concept.preferred_label,
            CONCEPT_DESCRIPTION=self.concept_description,
            EXAMPLES=self.examples,
            INPUT_PASSAGES=input_xml,
        )

        # remove all superfluous whitespace from the lines of the prompt
        prompt = "\n".join(line.strip() for line in prompt.split("\n"))

        return prompt

    def generate_text_dicts(self, texts: list[str]) -> dict[str, dict[str, str]]:
        """Generate the text dictionary for a batch of texts."""
        return {
            generate_identifier(text): {
                "original": text,
                "sanitised": self._sanitise_text_for_xml(text),
            }
            for text in texts
        }

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """Predict the presence of the concept in a batch of texts."""

        text_dicts = self.generate_text_dicts(texts)
        prompt = self.generate_prompt(text_dicts)

        llm_response = (
            self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            .content[0]
            .text
        )

        try:
            root = etree.fromstring(llm_response)
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML response from LLM: {e}")

        # store the output spans in a dictionary indexed by the identifier, so that we can
        # return them in the same order as the input texts
        output_spans_dict: dict[str, list[Span]] = {}

        for identifier, text_dict in text_dicts.items():
            passage = root.xpath(f'.//passage[@id="{identifier}"]')
            if not passage:
                raise ValueError(f"Identifier not found in output: {identifier}")

            marked_up_output_text = re.sub(
                r"^<passage[^>]*>|</passage>$",
                "",
                etree.tostring(passage[0], encoding="unicode", method="xml"),
            )
            output_text_without_tags = re.sub(
                r"<concept>|</concept>", "", marked_up_output_text
            )

            # Extra normalisation of both strings before comparison. Allows for some
            # sloppiness in reproduction as long as the indexes are very likely to be
            # the same
            sanitized_input = " ".join(text_dict["sanitised"].split()).lower()
            sanitized_output = " ".join(output_text_without_tags.split()).lower()

            if sanitized_input != sanitized_output:
                raise ValueError(
                    f"Output text does not match input text, even after normalization.\n"
                    f"Identifier: {identifier}\n"
                    f"Input text: '{sanitized_input}'\n"
                    f"Output text: '{sanitized_output}'"
                )

            output_spans_dict[identifier] = [
                span.model_copy(
                    # we need to use the original text here, not the sanitised copy!
                    update={"text": text_dict["original"]}
                )
                for span in self._get_spans_from_string(marked_up_output_text)
            ]

        # Return spans in the same order as input texts
        return [output_spans_dict[generate_identifier(text)] for text in texts]

    def predict(self, text: str) -> list[Span]:
        """Predict the presence of the concept in a single text."""
        # exactly the same process as for a batch of texts, but with a single input text
        # and a single list of output spans
        return self.predict_batch(texts=[text])[0]
