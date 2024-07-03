import argparse

from llm_interface import RAG


def main(question: str):
    print(f"Question: {question}\n")
    rag_results = RAG()(question)
    print(f"Answer: {rag_results.answer}")
    print("Sources used: unkown (since now using DSPy built in retriever)")


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
