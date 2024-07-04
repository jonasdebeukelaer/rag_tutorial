import os

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.evaluate import metrics

CHROMA_COLLECTION_NAME = "books"
CHROMA_PATH = "chroma"
os.environ["DSP_CACHEBOOL"] = "false"


class RAG(dspy.Module):
    """
    This class is a DSPy module which gets a question and returns an answer based
    on the context it can find in the chroma db.
    """

    def __init__(self, model_name="gpt-4o", number_of_extracts: int = 3):
        super().__init__()

        # init LM
        open_ai_api_key = os.getenv("OPENAI_API_KEY")
        self.lm = dspy.OpenAI(model_name, api_key=open_ai_api_key, max_tokens=1000)

        # init retriever
        retriever_model = ChromadbRM(
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=OpenAIEmbeddingFunction(api_key=open_ai_api_key),
        )

        # configure dspy defaults
        dspy.settings.configure(lm=self.lm, rm=retriever_model)

        # init modules
        self.retriever = dspy.Retrieve(
            k=number_of_extracts
        )  # could define custom retriver which also returns the source document data
        self.sumariser = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        """Summarises the information contained in the extracts to answer the question"""

        context = self.retriever(question).passages
        pred = self.sumariser(question=question, context=context)

        # TODO: bring back (caussing issues during metric checking?)
        # dspy.Assert(
        #     not ("Extracts" in pred.answer or "Answer" in pred.answer),
        #     "The answer should not contain a ref to the context or the word 'Answer' itself",
        # )

        return pred
