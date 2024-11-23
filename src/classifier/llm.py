import json
import os
import random
import re
from typing import Optional

from anthropic import Anthropic

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
        self.wikibase = wikibase
        default_examples = """<input>
        
        {
            "a8h3hcxj": "I worked on a horse farm when I was a kid. I was in charge of feeding the ponies.",
            "ykq2kk5a": "I love horses. I have a horse named Bessie.",
            "fh7akadd": "I have a pet dog."
        }
        
        </input>

        <output>
        
        {
            "a8h3hcxj": "I worked on a <CONCEPT>horse</CONCEPT> farm when I was a kid. I was in charge of feeding the <CONCEPT>ponies</CONCEPT>.",
            "ykq2kk5a": "I love <CONCEPT>horses</CONCEPT>. I have a <CONCEPT>horse</CONCEPT> named Bessie.",
            "fh7akadd": "I have a pet dog."
        }
        
        </output>
        """

        self.examples = (
            self._create_labelled_examples(self.concept.labelled_passages)
            if self.concept.labelled_passages
            else default_examples
        )

        self.prompt_template = """
        You are an AI assistant specialized in analyzing climate policy documents to identify mentions of the "{PREFERRED_LABEL}" concept. Your task is to process input passages and mark any references to the concept of "{PREFERRED_LABEL}" with XML tags.

        First, carefully review the following description of the concept of "{PREFERRED_LABEL}":

        <concept_description>
        {CONCEPT_DESCRIPTION}
        </concept_description>

        Now, you will be given a set of passages from climate policy documents in JSON format. Your task is to identify any instances of "{PREFERRED_LABEL}" in these passages and mark them with <CONCEPT> tags.

        <input_passages>
        {INPUT_PASSAGES}
        </input_passages>

        Instructions:
        1. Read through each passage carefully.
        2. Identify any mentions of the concept of "{PREFERRED_LABEL}", including direct references and related terms as described in the concept description.
        3. Surround the identified text with <CONCEPT> tags.
        4. If a passage contains multiple instances of the concept, mark all of them.
        5. If a passage does not contain any instances of the concept, include it in the output exactly as it was given to you.
        6. Maintain the original JSON structure in your output, using the same identifiers (keys) as in the input.
        7. CRITICAL: The output text must be EXACTLY identical to the input text, character-for-character, with the only difference being the addition of <CONCEPT> tags. This means:
           - No changing of spacing or punctuation
           - No fixing of typos or grammatical errors
           - No changing of capitalization
           - No removal or addition of any characters whatsoever
           - No reformatting or restructuring of any kind, including spacing or line breaks
        8. Apart from the <CONCEPT> tags, you must not modify the text in any way, no matter how small the change might seem.

        Before providing your final output, wrap your thought process for each passage in <concept_identification> tags. This process should include:
        1. Listing key terms and phrases from the concept description.
        2. For each passage, identifying potential matches and explaining why they fit or don't fit the concept.
        3. Counting the number of identified concepts in each passage.

        This will help ensure a thorough and accurate identification of the concept.

        Output Format:
        Present your final output in JSON format, matching the structure of the input but with added <CONCEPT> tags where appropriate. Each key-value pair should be on a new line for readability.

        Example output structure:
        {EXAMPLE_OUTPUT}

        Begin your concept identification process now, followed by the final output in the specified JSON format. Apart from what's contained within the <concept_identification> tags and the final JSON output, there should be no other text in your response.
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

        unlabelled = {
            generate_identifier(labelled_passage.text): labelled_passage.text
            for labelled_passage in sampled_examples
        }
        labelled = {
            generate_identifier(labelled_passage.text): (
                labelled_passage.get_highlighted_text()
                .replace("[cyan]", "<CONCEPT>")
                .replace("[/cyan]", "</CONCEPT>")
            )
            for labelled_passage in sampled_examples
        }

        return f"""<input>

        {json.dumps(unlabelled, indent=2)}

        </input>

        <output>

        {json.dumps(labelled, indent=2)}

        </output>
        """

    def _get_spans_from_string(self, text: str) -> list[tuple[int, int]]:
        """Get the spans from a string that has been marked up with concept markers."""
        spans = []
        # keep track of the number of characters we've added with concept markers
        offset = 0
        for match in re.finditer(r"<CONCEPT>(.*?)</CONCEPT>", text):
            start_index = match.start() - (offset * len("<CONCEPT></CONCEPT>"))
            end_index = start_index + len(match.group(1))
            offset += 1
            spans.append((start_index, end_index))
        return spans

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """Predict the presence of the concept in a batch of texts."""
        input_passages = {generate_identifier(text): text for text in texts}
        prompt = self.prompt_template.format(
            PREFERRED_LABEL=self.concept.preferred_label,
            CONCEPT_DESCRIPTION=self.concept.to_markdown(wikibase=self.wikibase),
            EXAMPLE_OUTPUT=self.examples,
            INPUT_PASSAGES=json.dumps(input_passages, indent=2),
        )

        # remove all superfluous whitespace from the lines of the prompt
        prompt = "\n".join(line.strip() for line in prompt.split("\n"))

        llm_response = (
            self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            .content[0]
            .text
        )

        # remove <concept_identification> tags from the response so that we're just left
        # with the JSON output
        stripped_response = re.sub(
            r"<concept_identification>.*?</concept_identification>",
            "",
            llm_response,
            flags=re.DOTALL,  # DOTALL ensures we can match tags across multi-line strings
        ).strip()
        try:
            json_output = json.loads(stripped_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON output: {llm_response}")

        output_spans = []
        for identifier, text in input_passages.items():
            # check that the identifier is present in the output
            if identifier not in json_output:
                raise ValueError(f"Identifier not found in output: {identifier}")
            output_text = json_output[identifier]
            # ensure that the output text is exactly the same as the input text, apart
            # from the added concept markers
            if output_text.replace("<CONCEPT>", "").replace("</CONCEPT>", "") != text:
                raise ValueError(
                    f"Output text does not match input text.\n"
                    f"Identifier: {identifier}\n"
                    f"Input text: {text}\n"
                    f"Output text: {output_text}"
                )

            output_spans.append(
                [
                    Span(
                        text=text,
                        concept=self.concept,
                        start_index=start_index,
                        end_index=end_index,
                    )
                    for start_index, end_index in self._get_spans_from_string(
                        output_text
                    )
                ]
            )
        return output_spans

    def predict(self, text: str) -> list[Span]:
        """Predict the presence of the concept in a single text."""
        # exactly the same process as for a batch of texts, but with a single input text
        # and a single list of output spans
        return self.predict_batch(texts=[text])[0]
