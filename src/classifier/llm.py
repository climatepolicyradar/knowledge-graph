import os
import random
import re
from datetime import datetime
from typing import Callable, Optional

from anthropic import Anthropic
from lxml import etree

from src.classifier import Classifier
from src.concept import Concept
from src.identifiers import generate_identifier
from src.labelled_passage import LabelledPassage
from src.span import Span
from src.wikibase import WikibaseSession


class LLMValidationError(Exception):
    """Base exception for LLM validation errors."""


class XMLStructureError(LLMValidationError):
    """Raised when the XML structure is invalid."""


class LLMResponseStructureError(LLMValidationError):
    """Raised when the response structure is invalid (missing sections or invalid format)."""


class PassageTextMismatchError(LLMValidationError):
    """Raised when the passage text doesn't match the expected text."""

    def __init__(self, passage_id: str, expected: str, found: str):
        self.passage_id = passage_id
        self.expected = expected
        self.found = found
        super().__init__(
            f"Text mismatch in passage {passage_id}:\n"
            f"Expected: {expected}\n"
            f"Found: {found}"
        )


class PassageNotFoundError(LLMValidationError):
    """Raised when a passage is not found in the response."""

    def __init__(self, passage_id: str):
        self.passage_id = passage_id
        super().__init__(f"Passage {passage_id} not found in response")


class TextPassage:
    """A passage of text with both original and sanitised versions."""

    def __init__(self, text: str, sanitiser: Callable[[str], str]):
        self.original = text
        self.sanitised = sanitiser(text)
        self.identifier = generate_identifier(text)

    @property
    def xml_element(self) -> str:
        """Get the XML representation of this passage."""
        return f'<passage id="{self.identifier}">{self.sanitised}</passage>'


class LLMClassifierResponse:
    """Structured container for LLM response data."""

    def __init__(self, reasoning: str, passages: dict[str, etree.Element]):
        self.reasoning = reasoning
        self.passages = passages

    def get_passage(self, passage_id: str) -> Optional[etree.Element]:
        """Get a passage by ID, returning None if not found."""
        return self.passages.get(passage_id)


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
        8. To help your classification, you may use a preceding <reasoning> section to explain any reasoning which helps you to identify the concept in each passage. It's a scratchpad - the thoughts here will not be used by any of the subsequent pipeline steps, but should help to make your process more coherent. For example, for each passage ID you might include a sentence or two, discussing each of the plausible terms within the passage and justifying your final choice for why each candidate term is/isn't a valid instance of the concept.
        9. CRITICAL: The text in each output <passage> must be EXACTLY identical to the input text, character-for-character, with the only difference being the addition of <concept> tags. This includes punctuation, whitespace, newlines, etc! Double check and re-read the input text, being extra careful to ensure that you haven't added or removed any characters apart from the <concept> tags.
        
        Your response must follow this exact structure:
        <response>
        <reasoning>
        [Space for reasoning which guides your final output <passage> objects in the <output> tag]
        </reasoning>
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
        """Use concept's gold-standard labelled passages to create examples for the LLM"""
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
            output_text = passage.get_highlighted_text(
                start_pattern="<concept>", end_pattern="</concept>"
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
        n_spans_found = 0
        for match in re.finditer(r"<concept>(.*?)</concept>", text):
            offset = n_spans_found * len("<concept></concept>")
            start_index = match.start() - offset
            end_index = start_index + len(match.group(1))
            n_spans_found += 1
            spans.append(
                Span(
                    text=text,
                    start_index=start_index,
                    end_index=end_index,
                    concept_id=self.concept.wikibase_id,
                    labellers=[str(self)],
                    timestamps=[datetime.now()],
                )
            )
        return spans

    def _sanitise_text_for_xml(self, text: str) -> str:
        """
        Sanitise text by replacing bad characters with filler strings

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
        - Normalizing Unicode characters like quotes, em/en dashes, and ellipses while
        preserving the text length.
        """
        # First handle XML special characters
        bad_xml_strings = ["&", "<", ">", '"', "'"]
        xml_translation = str.maketrans(
            {
                # account for future additions to the list of bad XML strings which might
                # be a few characters long
                string: "_" * len(string)
                for string in bad_xml_strings
            }
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

    def generate_prompt(self, passages: list[TextPassage]) -> str:
        """Generate the prompt for the LLM."""
        input_xml = "\n\t".join(passage.xml_element for passage in passages)

        prompt = self.prompt_template.format(
            PREFERRED_LABEL=self.concept.preferred_label,
            CONCEPT_DESCRIPTION=self.concept_description,
            EXAMPLES=self.examples,
            INPUT_PASSAGES=input_xml,
        )

        # remove all superfluous whitespace from the lines of the prompt
        prompt = "\n".join(line.strip() for line in prompt.split("\n"))

        return prompt

    def _get_llm_response(self, prompt: str) -> str:
        """Get raw response from LLM for the given prompt."""
        return (
            self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
            )
            .content[0]
            .text
        )

    def _remove_concept_tags(self, text: str) -> str:
        """Remove all concept tags from a string."""
        return re.sub(r"<concept>|</concept>", "", text)

    def _remove_passage_tags(self, text: str) -> str:
        """Remove all passage tags from a string."""
        return re.sub(r"^<passage[^>]*>|</passage>$", "", text)

    def _validate_passage_text(
        self, passage_element: etree.Element, expected_text: str, passage_id: str
    ) -> None:
        """Validate that passage text matches expected text (ignoring concept tags)."""
        passage_text = etree.tostring(passage_element, encoding="unicode", method="xml")
        parsed_passage_text = self._remove_passage_tags(passage_text)
        parsed_passage_text = self._remove_concept_tags(parsed_passage_text)

        sanitised_input = " ".join(expected_text.split()).lower()
        sanitised_output = " ".join(parsed_passage_text.split()).lower()

        if sanitised_input != sanitised_output:
            raise PassageTextMismatchError(
                passage_id=passage_id,
                expected=expected_text,
                found=parsed_passage_text,
            )

    def _extract_spans_from_passage(
        self,
        passage_elements: list[etree.Element],
        text_passage: TextPassage,
    ) -> list[Span]:
        """Extract concept spans from a marked-up passage after validation."""
        if not passage_elements:
            raise PassageNotFoundError(text_passage.identifier)

        passage = passage_elements[0]
        self._validate_passage_text(
            passage_element=passage,
            expected_text=text_passage.sanitised,
            passage_id=text_passage.identifier,
        )

        passage_text = etree.tostring(passage, encoding="unicode", method="xml")
        marked_up_text = self._remove_passage_tags(passage_text)

        return [
            span.model_copy(update={"text": text_passage.original})
            for span in self._get_spans_from_string(marked_up_text)
        ]

    def _parse_llm_response(self, llm_response: str) -> LLMClassifierResponse:
        """Parse the complete LLM response into a structured format"""
        try:
            root = etree.fromstring(llm_response)
        except etree.XMLSyntaxError as e:
            raise XMLStructureError(f"Invalid XML structure: {e}")

        reasoning_elements = root.xpath(".//reasoning")
        if not reasoning_elements:
            raise LLMResponseStructureError("Response is missing a reasoning section")
        reasoning = reasoning_elements[0].text or ""

        # Extract passages
        output_elements = root.xpath(".//output")
        if not output_elements:
            raise LLMResponseStructureError("Response is missing an output section")

        try:
            passages = {
                passage.get("id"): passage
                for passage in output_elements[0].xpath(".//passage[@id]")
            }
        except Exception as e:
            raise LLMResponseStructureError(f"Failed to parse output: {e}")

        return LLMClassifierResponse(reasoning=reasoning, passages=passages)

    def predict_batch(self, texts: list[str], n_retries: int = 3) -> list[list[Span]]:
        """Predict the presence of the concept in a batch of texts."""
        passages = [
            TextPassage(text=text, sanitiser=self._sanitise_text_for_xml)
            for text in texts
        ]

        try:
            prompt = self.generate_prompt(passages)
            llm_response = self._get_llm_response(prompt)
            parsed_response = self._parse_llm_response(llm_response)

            # Process each passage and collect results
            passage_id_to_predicted_spans: dict[str, list[Span]] = {}
            for passage in passages:
                llm_passage = parsed_response.get_passage(passage.identifier)
                if llm_passage is None:
                    raise PassageNotFoundError(passage.identifier)

                try:
                    passage_id_to_predicted_spans[passage.identifier] = (
                        self._extract_spans_from_passage(
                            passage_elements=[llm_passage],
                            text_passage=passage,
                        )
                    )
                except PassageTextMismatchError as e:
                    # Include reasoning in error context if available
                    error_msg = f"Text mismatch in passage {passage.identifier}"
                    if parsed_response.reasoning:
                        error_msg += f"\nLLM reasoning: {parsed_response.reasoning}"
                    raise ValueError(error_msg) from e

            # Return results in same order as input texts
            return [
                passage_id_to_predicted_spans[passage.identifier]
                for passage in passages
            ]

        except XMLStructureError as e:
            raise ValueError("Invalid XML structure in LLM response") from e
        except LLMResponseStructureError as e:
            raise ValueError("Invalid response structure") from e
        except Exception as e:
            raise ValueError("Failed to process LLM predictions") from e

    def predict(self, text: str) -> list[Span]:
        """Predict the presence of the concept in a single text."""
        return self.predict_batch(texts=[text])[0]
