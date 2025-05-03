import typing as t

import numpy as np
import pandas as pd
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    BaseEvaluator,
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

EVAL_DICT_MAPPER: dict[str, BaseEvaluator] = {
    'faithfullness': FaithfulnessEvaluator,
    'correctness': CorrectnessEvaluator,
    'relevancy': RelevancyEvaluator,
    'context': ContextRelevancyEvaluator,
    'answer': AnswerRelevancyEvaluator,
}


def get_eval_results(key: str, eval_results: dict[str, t.Any]) -> t.Any:
    """helper function to get the evaluation results for the given key, gives a the percentage score.

    Args:
        key (str): evaluation metric to be aggregated.
        eval_results (dict[str, t.Any]): dictionary of evaluation results for all metrics.

    Returns:
        t.Any: float score for the given metric.
    """
    results = eval_results[key]

    if key.startswith('correct'):
        scores = [result.score for result in results]
        score = np.mean(scores) / 5.0 * 100  # since the score is between 0-5
    else:
        scores = [result.passing for result in results]
        score = np.mean(scores) * 100

    # print the score for the given metric.
    print(f'Score for {key}: {score}%')
    return score


def get_correctness_eval_df(eval_results: dict[str, t.Any], ground_truth: list[str]) -> pd.DataFrame:
    """helper function to export question, ground truth, response and score for the correctness evaluator.

    Args:
        eval_results (dict[str, t.Any]): dictionary of evaluation results for all metrics.
        ground_truth (list[str]): list of ground truth answers for the questions.

    Returns:
        pd.DataFrame: dataframe with all relevant information from the correctness evaluator.
    """
    data_stream = []
    for result, gt in zip(eval_results['correctness'], ground_truth):
        data_stream.append(
            {
                'question': result.query,
                'ground_truth': gt,
                'response': result.response,
                'score': result.score,
            }
        )

    return pd.DataFrame(data_stream)
