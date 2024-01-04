from typing import Callable
import numpy as np
from algorithms_base import DspyPipeline, PipelineSpaceBuilder, DspyAlgorithmBase
import inspect
import sys
import algorithms
from autogoal_core.search import SearchAlgorithm
import os
import dspy
from dspy.evaluate.evaluate import Evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AutoDspy:
    def __init__(
        self,
        lm,
        path_to_llm: str,
        search_algorithm: type[SearchAlgorithm],
        metric: Callable,
        evaluator,
        random_state: int | None = None,
        search_iterations=100,
        errors="warn",
        core_llm_name: str = "openhermes2.5-mistral:7b-q5_K_M",
        **search_kwargs,
    ) -> None:
        self.search_algorithm = search_algorithm
        self.path_to_llm = path_to_llm
        self.metric = metric
        self.evaluator = evaluator
        self.random_state = random_state
        self.search_iterations = search_iterations
        self.search_kwargs = search_kwargs
        self.errors = errors
        self.core_llm_name = core_llm_name
        if random_state:
            np.random.seed(random_state)

        # dspy.settings.configure(lm=lm)

    def fit(self, dataset_description: str, trainset, examples_descriptions: dict[str, str], **kwargs):
        pipeline_space_builder = PipelineSpaceBuilder(self.path_to_llm, examples_descriptions)
        search = self.search_algorithm(
            pipeline_space_builder.build_pipeline_graph(
                dataset_description,
                registry=self._find_algorithms(),
            ),
            self.make_fitness_function(self.evaluator, self.metric, trainset),
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
    
    def make_fitness_function(self, evaluator, metric, trainset):

        def fitness_function(pipeline: DspyPipeline):
            compiled_program = pipeline.run(trainset=trainset, metric=metric)
            return (evaluator(compiled_program, metric= metric),)
        
        return fitness_function
