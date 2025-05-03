import typing as t

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.evaluation import BatchEvalRunner
from tqdm import tqdm

from humana_take_home.utils import get_default_ollama_llm

from .utils import EVAL_DICT_MAPPER


class ResponseEvaluator:
    """Evaluator to evaluate the response from the chatbot using different metrics.
    The evaluator uses the following metrics:
      - Relevancy
      - Faithfulness
      - Correctness
    """

    def __init__(
        self,
        use_correctness: bool = True,
    ) -> None:
        # define the evaluators to use for the evaluation.
        self.llm = get_default_ollama_llm(temperature=0.0)
        self.evaluators = {
            'relevancy': EVAL_DICT_MAPPER['relevancy'](llm=self.llm),
            'faithfullness': EVAL_DICT_MAPPER['faithfullness'](llm=self.llm),
        }
        if use_correctness:
            self.evaluators['correctness'] = EVAL_DICT_MAPPER['correctness'](llm=self.llm)

        # define the batch eval runner to use for the evaluation.
        self.batch_eval_runner = BatchEvalRunner(
            evaluators=self.evaluators,
            workers=4,  # workers to use for evaluation, could be changed to higer number for faster evaluation.
            show_progress=True,
        )

    async def evaluate(self, data: dict[str, list[str]], chat_engine: BaseChatEngine) -> t.Any:
        """a method to evaluate the responses from the chatbot using different metrics.

        Args:
            data (dict[str, list[str]]): query and ground truth data to evaluate the responses.
            chat_engine (BaseChatEngine): chat engine to be evaluated.

        Returns:
            t.Any: dict of evaluation results for all metrics.
        """

        # grab responses from the chat engine before evaluation.
        # this is to avoid the overhead of calling the chat engine for each evaluation.
        responses = await self.__get_responses(
            questions=data['questions'],
            chat_engine=chat_engine,
        )

        eval_results = await self.batch_eval_runner.aevaluate_responses(
            queries=data['questions'],
            responses=responses,
            references=data.get('ground_truth', None),
        )

        return eval_results

    async def __get_responses(
        self,
        questions: list[str],
        chat_engine: BaseChatEngine,
    ) -> t.Any:
        """method to get the responses from chat engine for the given questions.

        Args:
            questions (list[str]): list of questions to generate responses for.
            chat_engine (BaseChatEngine): chat engine to be used for generating responses.

        Returns:
            t.Any: list of responses for the input questions.
        """
        responses = []
        for _, question in tqdm(enumerate(questions), total=len(questions), desc='Generating responses'):
            response = await chat_engine.achat(
                message=question,
                chat_history=None,
            )
            chat_engine.reset()
            responses.append(response)
        return responses
