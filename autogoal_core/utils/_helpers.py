# # `autogoal_core.utils._helpers`

"""
This module contains the `optimize` function that allows to apply black-box hyper-parameter
search to an arbitrary Python code.
"""

import inspect
import textwrap
import functools

from typing import Callable

from autogoal_core.search import PESearch
from autogoal_core.grammar import generate_cfg


Kb = 1024
Mb = 1024 * Kb
Gb = 1024 * Mb

Sec = 1
Min = 60 * Sec
Hour = 60 * Min

MAX_REPR_DEPTH = 10

_repr_depth = [0]


def nice_repr(cls):
    """
    A decorator that adds a nice `repr(.)` to any decorated class.

    Decorate a class with `@nice_repr` to automatically generate a `__repr__()`
    method that prints the class name along with any parameters defined in the
    constructor which can be found in `dir(self)`.

    ##### Examples

    All of the parameters that you want to be printed in `repr(.)` should
    be either stored in the instance or accesible by name (e.g., as a property).

    ```python
    >>> @nice_repr
    ... class MyType:
    ...     def __init__(self, a, b, c):
    ...         self.a = a
    ...         self._b = b
    ...         self._c = c
    ...
    ...     @property
    ...     def b(self):
    ...         return self._b
    ...
    >>> x = MyType(42, b='hello', c='world')
    >>> x
    MyType(a=42, b="hello")

    ```

    It works nicely with nested objects, if all of them are `@nice_repr` decorated.

    ```python
    >>> @nice_repr
    ... class A:
    ...     def __init__(self, inner):
    ...         self.inner = inner
    >>> @nice_repr
    ... class B:
    ...     def __init__(self, value):
    ...         self.value = value
    >>> A([B(i) for i in range(10)])
    A(
        inner=[
            B(value=0),
            B(value=1),
            B(value=2),
            B(value=3),
            B(value=4),
            B(value=5),
            B(value=6),
            B(value=7),
            B(value=8),
            B(value=9),
        ]
    )

    ```

    It works with cyclic object graphs as well:

    ```python
    >>> @nice_repr
    ... class A:
    ...     def __init__(self, a:A=None):
    ...         self.a = self
    >>> A()
    A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(...))))))))))))

    ```

    !!! note
        Change `autogoal_core.utils.MAX_REPR_DEPTH` to increase the depth level of recursive `repr`.

    """

    def repr_method(self):
        init_signature = inspect.signature(self.__init__)
        exclude_param_names = set(["self"])

        if _repr_depth[0] > MAX_REPR_DEPTH:
            return f"{self.__class__.__name__}(...)"

        _repr_depth[0] += 1

        parameter_names = [
            name
            for name in init_signature.parameters
            if name not in exclude_param_names
        ]
        parameter_values = [getattr(self, param, None) for param in parameter_names]

        if hasattr(self, "__nice_repr_hook__"):
            self.__nice_repr_hook__(parameter_names, parameter_values)

        args = ", ".join(
            f"{name}={repr(value)}"
            for name, value in zip(parameter_names, parameter_values)
            if value is not None
        )
        fr = f"{self.__class__.__name__}({args})"

        _repr_depth[0] -= 1

        try:
            import black

            return black.format_str(fr, mode=black.FileMode()).strip()
        except:
            return fr

    cls.__repr__ = repr_method
    return cls


def flatten(y):
    """
    Recursively flattens a list.

    ##### Examples

    ```python
    >>> flatten([[1],[2,[3]],4])
    [1, 2, 3, 4]

    ```
    """
    if isinstance(y, list):
        return [z for x in y for z in flatten(x)]
    else:
        return [y]


def factory(func_or_type, *args, **kwargs):
    def call():
        return func_or_type(*args, **kwargs)

    return call


# ## Black-box optimization

# The following function defines a black-box optimization that can be applied to any function.


def optimize(
    fn,
    search_strategy=PESearch,
    generations=100,
    pop_size=10,
    allow_duplicates=False,
    logger=None,
    **kwargs,
):
    """
    A general-purpose optimization function.

    Simply define any function `fn` with suitable parameter annotations
    and apply `optimize`.

    **Parameters**:

    * `search_strategy`: customize the search strategy. By default a `PESearch` will be performed.
    * `generations`: max number of generations to run.
    * `logger`: instance of `Logger` (or list) to pass to the search strategy.
    * `**kwargs`: additional keyword arguments passed to the search strategy constructor.
    """

    params_func = _make_params_func(fn)

    @functools.wraps(fn)
    def eval_func(kwargs):
        return fn(**kwargs)

    grammar = generate_cfg(params_func)

    search = search_strategy(
        grammar,
        eval_func,
        pop_size=pop_size,
        allow_duplicates=allow_duplicates,
        **kwargs,
    )
    best, best_fn = search.run(generations, logger=logger)

    return best, best_fn


# ### Implementation details

# To make `optimize` work we need to define both a grammar and a callable function
# to pass to the search algorithm


class _ParamsDict(dict):
    pass


def _make_params_func(fn: Callable):
    signature = inspect.signature(fn)

    func_name = f"{fn.__name__}_params"
    args_names = signature.parameters.keys()

    def annotation_repr(ann):
        if inspect.isclass(ann) or inspect.isfunction(ann):
            return ann.__name__

        return repr(ann)

    args_line = ",\n                ".join(f"{k}={k}" for k in args_names)
    params_line = ", ".join(
        f"{arg.name}:{annotation_repr(arg.annotation)}"
        for arg in signature.parameters.values()
    )

    func_code = textwrap.dedent(
        f"""
        def {func_name}({params_line}):
            return _ParamsDict(
                {args_line}
            )"""
    )

    globals_dict = dict(fn.__globals__)
    globals_dict["_ParamsDict"] = _ParamsDict
    locals_dict = {}
    exec(func_code, globals_dict, locals_dict)
    return locals_dict[func_name]
