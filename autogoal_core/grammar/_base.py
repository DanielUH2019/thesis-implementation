from autogoal_core.sampling import Sampler
from typing import Optional

class Grammar:
    def __init__(self, start):
        self._start = start

    def sample(self, *, max_iterations: int = 100, sampler: Optional[Sampler] = None):
        if sampler is None:
            sampler = Sampler()

        return self._sample(
            symbol=self._start, max_iterations=max_iterations, sampler=sampler
        )

    def __call__(self, sampler: Optional[Sampler] = None):
        return self.sample(sampler=sampler)

    def _sample(self, symbol, max_iterations, sampler):
        raise NotImplementedError()
