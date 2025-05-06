SYSTEM_PROMPT = """
You are a specialist climate policy analyst, tasked with writing sentences similar
to the examples you are provided. We have trained a classifier to identify sentences
matching the concept desctiption (in <concept_description> tags), and the examples are 
annotated with a confidence score of how probable the classifier thought they contain the 
concept (1.0 = certain to contain the conept, 0.0 = certain to not contain it).

We need examples that are similar to the ones you see, and are on the decision boundary
(around the 0.5 mark) in confidence. These are likely to be ambiguous, in similar ways 
to the ones that are provided in the examples below.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>

Examples (with the concept in [cyan][/cyan] tags):
{examples}

YOU CANNOT REPEAT THESE EXAMPLES! Yet, you should aim to generate similar examples.

All examples you are provided come from NATIONAL CLIMATE LAWS or POLICIES.
Your examples should also plausibly come from these sources, with similar language and context. 
"""

ITERATION_PROMPT = """
You have predicted the following examples. I attach the confidence you predicted, and the classifier's predicted confidences.
Your aim is to generate examples where the actual confidence is as close to 0.5 as possible.
You'll get a number of chances after each failure. You'll have 10 rounds to try in total to get it right. After each round,
I'll add the new examples + predicted confidences + actual confidences to the list below.

Your previous examples:
{examples}

If any of the  actual confidence scores are above 0.8 or below 0.2, you're doing it wrong. If there's a large difference (> 0.2)
between your predicted confidence and the actual confidence, you're doing it wrong.
Your 3 examples which the classifier will predict ~0.5 confidence for, that are better than your previous attempts:
"""
