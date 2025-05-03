import asyncio
from argparse import ArgumentParser
from dataclasses import dataclass

import pandas as pd

from humana_take_home.agents.research import ResearchAgent
from humana_take_home.evaluators.evaluator import ResponseEvaluator
from humana_take_home.evaluators.utils import get_correctness_eval_df, get_eval_results


async def main(path_to_csv_file: str, use_correctness: bool, export_results_path: str | None = None) -> None:
    """runner function to evaluate the chatbot using the csv file containing evaluation data.

    Args:
        path_to_csv_file (str): path to the csv file containing the evaluation data.
        use_correctness (bool): if correctness evaluator should be used.
        export_results_path (str | None, optional): path to export correctness evaluation results dataframe.
        Defaults to None.
    """

    # read the csv file and convert it to dict.
    data = pd.read_csv(path_to_csv_file).to_dict(orient='list')

    # define evaluator and agent to be evaluated.
    evaluator = ResponseEvaluator(
        use_correctness=use_correctness,
    )
    agent = ResearchAgent.from_local_storage(similarity_top_k=3)

    # run evaluation on test data and chat engine.
    eval_results = await evaluator.evaluate(data=data, chat_engine=agent.get_chat_engine())

    # print the evaluation scores for all metrics.
    for key in ['relevancy', 'faithfullness']:
        get_eval_results(key, eval_results)

    if use_correctness:
        get_eval_results('correctness', eval_results)

    # export the correctness evaluation results to csv file.
    if export_results_path is not None:
        df = get_correctness_eval_df(eval_results, data['ground_truth'])
        df.to_csv(export_results_path, index=False)


if __name__ == '__main__':

    @dataclass
    class CommandLine:
        path_to_csv_file: str
        use_correctness: bool
        export_results_path: str | None = None

    argparser = ArgumentParser('evaluate trained chatbot')
    argparser.add_argument(
        '--path_to_csv_file',
        help='path to the csv file containing the evaluation data',
        required=True,
    )
    argparser.add_argument(
        '--use_correctness',
        help='use correctness evaluator',
        action='store_true',
    )
    argparser.add_argument(
        '--export_results_path',
        help='path to export the evaluation results',
        required=False,
    )

    args = CommandLine(**vars(argparser.parse_args()))

    # run main function
    asyncio.run(
        main(
            path_to_csv_file=args.path_to_csv_file,
            use_correctness=args.use_correctness,
            export_results_path=args.export_results_path,
        )
    )
