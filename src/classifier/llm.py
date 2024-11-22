import json
import os
import random
import re

from anthropic import Anthropic

from src.classifier import Classifier
from src.concept import Concept
from src.identifiers import generate_identifier
from src.labelled_passage import LabelledPassage
from src.span import Span


class LLMClassifier(Classifier):
    """A classifier that uses an LLM to predict the presence of a concept in a text."""

    def __init__(self, concept: Concept, model: str = "claude-3-5-haiku-20241022"):
        self.concept = concept
        default_examples = """<INPUT>
        
        {
            "a8h3hcxj": "I worked on a horse farm when I was a kid. I was in charge of feeding the ponies.",
            "ykq2kk5a": "I love horses. I have a horse named Bessie.",
            "fh7akadd": "I have a pet dog."
        }
        
        </INPUT>

        <OUTPUT>
        
        {
            "a8h3hcxj": "I worked on a <CONCEPT>horse</CONCEPT> farm when I was a kid. I was in charge of feeding the <CONCEPT>ponies</CONCEPT>.",
            "ykq2kk5a": "I love <CONCEPT>horses</CONCEPT>. I have a <CONCEPT>horse</CONCEPT> named Bessie.",
            "fh7akadd": "I have a pet dog."
        }
        
        </OUTPUT>
        """

        examples = (
            self._create_labelled_examples(self.concept.labelled_passages)
            if self.concept.labelled_passages
            else default_examples
        )

        self.prompt = f"""We're trying to identify instances of the concept "{self.concept.preferred_label}" in climate policy documents. You will be given a set of passages of text from climate policy documents, which may contain instances of the concept. Your task is to identify any instances of the concept in the text passages.
        
        Passages may contain multiple instances of the concept, and you should identify all of them if they are present. If a passage does not contain any instances of the concept, you should still include it in the output, exactly as it was given  to you. Concepts may be referred to by different names in different passages, so you should not make assumptions about what the concept might be called.

        You should mark instances of the concept by surrounding the relevant text in XML tags like <CONCEPT>relevant text</CONCEPT>.

        The input passages will be given to you in json format, and you should output your answer in the same format. Here is an example of the input and output format, based on the concept of "{self.concept.preferred_label if self.concept.labelled_passages else "horse"}":

        {examples}

        The identifiers (keys) for each passage are unique, and those in the output should correspond exactly to those in the input.
        
        In this case, you will need to identify spans which refer to the concept of "{self.concept.preferred_label}" based on the following description. You should use this description to identify the concept in the input passages, even if the concept is referred to indirectly, or by a different name in some of thepassages.

        Here is the complete description of the concept, based on our knowledge of the climate policy domain:

        <CONCEPT_DESCRIPTION>

        {self.concept.to_markdown()}
        
        </CONCEPT_DESCRIPTION>

        Here is the input text for this task:

        <INPUT>

        __INPUT_PASSAGES__
        
        </INPUT>

        Begin generating the response now, presenting your output in the specified JSON format. Do not include any additional content, preamble, or explanation in your response."""

        # remove all superfluous whitespace from the lines of the prompt
        self.prompt = "\n".join(line.strip() for line in self.prompt.split("\n"))

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
        sampled_examples = random.sample(positive_examples, 2) + random.sample(
            negative_examples, 1
        )

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

        return f"""<INPUT>

        {json.dumps(unlabelled, indent=2)}

        </INPUT>

        <OUTPUT>

        {json.dumps(labelled, indent=2)}

        </OUTPUT>
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
        prompt = self.prompt.replace(
            "__INPUT_PASSAGES__", json.dumps(input_passages, indent=2)
        )
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
            json_output = json.loads(llm_response)
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
