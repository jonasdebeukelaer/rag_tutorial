from typing import List
import random

import dspy
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

from llm_interface import RAG

gpt4 = dspy.OpenAI(model="gpt-4", max_tokens=1000)


def main():
    print("Load data...")
    dataset = DataLoader().from_csv("data/testing_data.csv", input_keys=("question",))
    
    random.shuffle(dataset)
    

    train_dev_ratio = 0.3
    split = int(train_dev_ratio * len(dataset))
    trainset = dataset[:split]
    devset = dataset[split:]

    print("\nInit program...")
    program = RAG(model_name="gpt-3.5-turbo-0125")

    print("\nCreating optimised program...")
    opti_program = optimise(program, trainset)


    print("\nUnoptimised program evaluation:")
    evaluator(program, devset)

    print("\nOptimised program evalulation:")
    evaluator(opti_program, devset)


class Assess(dspy.Signature):
    """
    Compare the two statements and return a similarity score.
    """

    statement_1: str = dspy.InputField()
    statement_2: str = dspy.InputField()
    similarity = dspy.OutputField(desc="The similarity score, a number in range [0, 1]")


def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None):
    with dspy.context(lm=gpt4):
        assessed = dspy.Predict(Assess)(statement_1=gold.answer, statement_2=pred.answer)

    if trace is None:
        return float(assessed.similarity)

    return float(assessed.similarity) > 0.8


def evaluator(program: dspy.Module, devset: List[dspy.Example]):
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=True)

    # Launch evaluation.
    result = evaluate(program, metric=metric)
    print(result)


def optimise(program: dspy.Module, trainset: List[dspy.Example]) -> dspy.Module:

    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
    # The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.

    optimiser = BootstrapFewShot(
        metric=metric,
        metric_threshold=0.7,
        max_bootstrapped_demos=4,
        max_labeled_demos=20,
        max_rounds=10,
    )
    return optimiser.compile(program, trainset=trainset)


if __name__ == "__main__":
    main()
