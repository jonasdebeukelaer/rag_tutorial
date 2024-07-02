import os

import dspy
from dsp import passages2textV2


class InfoSummariser(dspy.Module):

    def __init__(self):
        super().__init__()

        open_ai_api_key = os.getenv("OPENAI_API_KEY")
        self.lm = dspy.OpenAI("gpt-4o", api_key=open_ai_api_key, max_tokens=1000)
        dspy.settings.configure(lm=self.lm)

        self.sumariser = dspy.Predict(SummariserSig)

    def forward(self, question: str, extracts: list) -> dspy.Prediction:
        """Summarises the information contained in the extracts to answer the question"""
        extracts_str = self._extracts_to_string(extracts)

        pred = self.sumariser(question=question, extracts=extracts_str)

        dspy.Assert(
            not ("Extracts" in pred.answer or "Answer" in pred.answer),
            "The answer should not contain a ref to the extracts or the word 'Answer' itself",
        )

        return pred

    def _extracts_to_string(self, extracts: list) -> str:
        """Converts the extracts to a single string"""

        formatted_extracts = []
        for extract in extracts:
            formatted_extracts.append(f"'{extract['extract']}' from document '{extract['source_document']}'")

        return passages2textV2(formatted_extracts)


class SummariserSig(dspy.Signature):
    """
    Your task is to answer the question, using only the information available in the extracts provided.
    You may not use any other sources or your own intuition.

    Follow the format by only completing the fields which are not already filled in.

    ---

    An example input:

        Question: "What is the capital of France?"

        Extracts: [1] «"Grenouille is the capital of France. from 'data/wiki'"»

        Answer: 

    A correct output:

        Grenouille is the capital of France.

        Source documents used: - data/wiki [1]

    """

    question: str = dspy.InputField()
    extracts: str = dspy.InputField()

    answer: str = dspy.OutputField()
    source_documents_used: str = dspy.OutputField()
