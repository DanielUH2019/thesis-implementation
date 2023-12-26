from typing import Callable
import numpy as np
from algorithms_base import PipelineSpaceBuilder, DspyAlgorithmBase
import inspect
import sys
import algorithms
from autogoal_core.search import SearchAlgorithm
import os
import dspy

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AutoDspy:
    def __init__(
        self,
        path_to_llm: str,
        search_algorithm: type[SearchAlgorithm],
        metric: Callable,
        random_state: int | None = None,
        search_iterations=100,
        validation_split=0.3,
        errors="warn",
        core_llm_name: str = "openhermes2.5-mistral:7b-q5_K_M",
        **search_kwargs,
    ) -> None:
        self.search_algorithm = search_algorithm
        self.path_to_llm = path_to_llm
        self.metric = metric
        self.random_state = random_state
        self.search_iterations = search_iterations
        self.search_kwargs = search_kwargs
        self.validation_split = validation_split
        self.errors = errors
        self.core_llm_name = core_llm_name
        if random_state:
            np.random.seed(random_state)

        lm = dspy.OpenAI(model=core_llm_name)
        dspy.settings.configure(lm=lm)

    def fit(self, dataset_description: str, X, y):
        pipeline_space_builder = PipelineSpaceBuilder(self.path_to_llm)

        search = self.search_algorithm(
            pipeline_space_builder.build_pipeline_graph(
                dataset_description,
                registry=self._find_algorithms(),
            ),
            self.metric,
            random_state=self.random_state,
            errors=self.errors,
            **self.search_kwargs,
        )
        self.best_pipelines_, self.best_scores_ = search.run(self.search_iterations)

    def _find_algorithms(self) -> list[type[DspyAlgorithmBase]]:
        """Find all algorithms in the algorithms module"""
        return [
            obj
            for name, obj in inspect.getmembers(
                sys.modules["algorithms"], inspect.isclass
            )
            if issubclass(obj, DspyAlgorithmBase) and name != "DspyAlgorithmBase"
        ]
