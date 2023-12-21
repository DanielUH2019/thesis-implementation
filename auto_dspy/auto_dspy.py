import numpy as np
from algorithms_base import PipelineSpaceBuilder, DspyAlgorithmBase
import inspect
import sys
import algorithms
from autogoal_core.search import SearchAlgorithm
from autogoal_core.utils.metrics import (
    unsupervised_fitness_fn_moo,
    supervised_fitness_fn_moo,
    accuracy,
)
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AutoDspy:
    def __init__(
        self,
        dataset_description: str,
        path_to_llm: str,
        search_algorithm: type[SearchAlgorithm],
        random_state: int | None = None,
        search_iterations=100,
        objectives=None,
        validation_split=0.3,
        errors="warn",
        cross_validation="median",
        cross_validation_steps=3,
        **search_kwargs,
    ) -> None:
        self.dataset_description = dataset_description
        self.search_algorithm = search_algorithm
        self.path_to_llm = path_to_llm
        self.random_state = random_state
        self.search_iterations = search_iterations
        self.search_kwargs = search_kwargs
        self.objectives = objectives or accuracy
        self.validation_split = validation_split
        self.errors = errors
        self.cross_validation = cross_validation
        self.cross_validation_steps = cross_validation_steps
        if random_state:
            np.random.seed(random_state)

        if not type(self.objectives) is type(tuple) and not type(
            self.objectives
        ) is type(list):
            self.objectives = (self.objectives,)

    def fit(self, X, y):
        pipeline_space_builder = PipelineSpaceBuilder(self.path_to_llm)

        search = self.search_algorithm(
            pipeline_space_builder.build_pipeline_graph(
                self.dataset_description,
                registry=self._find_algorithms(),
            ),
            self.make_fitness_fn(X, y),
            random_state=self.random_state,
            errors=self.errors,
            **self.search_kwargs,
        )
        self.best_pipelines_, self.best_scores_ = search.run(self.search_iterations)

        self.fit_pipeline(X, y)

    def _check_fitted(self):
        if not hasattr(self, "best_pipelines_"):
            raise TypeError(
                "This operation cannot be performed on an unfitted AutoML instance. Call `fit` first."
            )

    def _find_algorithms(self):
        """Find all algorithms in the algorithms module"""
        return [
            obj
            for name, obj in inspect.getmembers(
                sys.modules["algorithms"], inspect.isclass
            )
            if issubclass(obj, DspyAlgorithmBase) and name != "DspyAlgorithmBase"
        ]

    def fit_pipeline(self, X, y):
        self._check_fitted()

        for pipeline in self.best_pipelines_:
            pipeline.send("train")
            pipeline.run(X, y)
            pipeline.send("eval")

    def make_fitness_fn(self, X, y=None):
        """
        Create a fitness function to evaluate pipelines.
        """
        if y is not None:
            y = np.asarray(y)

        inner_fitness_fn = (
            unsupervised_fitness_fn_moo(self.objectives)
            if y is None
            else supervised_fitness_fn_moo(self.objectives)
        )

        def fitness_fn(pipeline):
            return inner_fitness_fn(
                pipeline,
                X,
                y,
                self.validation_split,
                self.cross_validation_steps,
                self.cross_validation,
            )

        return fitness_fn
