import os
import argparse

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from llm import InfoSummariser

CHROMA_PATH = "chroma"
os.environ["DSP_CACHEBOOL"] = "false"


def main(question: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    summariser = InfoSummariser()

    print(f"Question: {question}")
    extracts = search(db, question)

    if extracts is None or len(extracts) == 0:
        print("Could not find the answer :(")
        return

    print("Sending question to answer to LLM...\n")
    summariser_results = summariser(question, extracts)

    print(f"Answer: {summariser_results.answer}")
    print(f"Sources used: {summariser_results.source_documents_used}")


def search(db, query_text: str, extracts_to_return: int = 3) -> list:
    print("Searching vector db for relevant extracts...")
    results = db.similarity_search_with_relevance_scores(query_text, k=extracts_to_return)

    result_set = []

    if len(results) == 0 or results[0][1] < 0.7:
        print("No good matches found in the provided documents.")
        return result_set

    print(f"Top {extracts_to_return} relevant extracts:\n")

    i = 0
    for doc, score in results:
        i += 1
        print(f"document: {i} | score: {score} | source: {doc.metadata['source']}\n", doc.page_content, "\n")
        result_set.append(
            {
                "extract": doc.page_content,
                "extract_relevance_score": score,
                "source_document": doc.metadata["source"],
            }
        )

    print("finished search\n\n")
    return result_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="The question we want answered",
        default="what is the most shocking thing Alice sees?",
    )
    args = parser.parse_args()

    main(args.question)
