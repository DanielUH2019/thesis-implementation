import abc
from typing import Any, Optional, Tuple
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature
from autogoal_core.kb._algorithm import (
    Algorithm,
    PipelineNode,
    PipelineSpace,
    make_seq_algorithm,
)
import networkx as nx
from autogoal_core.utils import nice_repr
from autogoal_core.grammar import Graph, GraphSpace, generate_cfg, Union, Symbol
import guidance


class AlgorithmSignature(Signature):
    def inputs_fields_to_maintain(self) -> dict[str, InputField]:
        return {}


class DspyAlgorithmBase(Algorithm):
    """Represents an abstract dspy algorithm with a run method."""

    @classmethod
    @abc.abstractmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        pass

    @classmethod
    def input_types(cls) -> Tuple[type, ...]:
        """Returns an ordered list of the expected semantic input types of the `run` method."""
        return tuple(
            [v.annotation for v in cls.get_signature().input_fields().values()]
        )

    @classmethod
    def input_args(cls) -> Tuple[str, ...]:
        """Returns an ordered tuple of the names of the arguments in the `run` method."""
        return tuple(cls.get_signature().input_fields().keys())

    @classmethod
    def output_type(cls) -> Tuple[type, ...]:
        """Returns an ordered list of the expected semantic output type of the `run` method."""
        return tuple(
            [v.annotation for v in cls.get_signature().output_fields().values()]
            + [
                v.annotation
                for v in cls.get_signature().inputs_fieldds_to_output().values()
            ]
        )

    @abc.abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """Executes the algorithm."""
        pass

    @classmethod
    def is_compatible_with(cls, other_signature: Signature) -> bool:
        outputs = cls.get_signature().output_fields()
        inputs = other_signature.input_fields()
        if len(outputs) != len(inputs):
            return False

        for o_key in outputs:
            for i_key in inputs:
                if (
                    o_key != i_key
                    or outputs[o_key].annotation != inputs[i_key].annotation
                ):
                    return False

        return True


def build_input_args(
    algorithm: DspyAlgorithmBase, values: dict[str, Any]
) -> dict[str, Any]:
    """Buils the correct input mapping for `algorithm` using the provided `values` mapping types to objects."""
    inputs = {
        k: v for k, v in algorithm.get_signature().kwargs if isinstance(v, InputField)
    }
    result = {k: v for k, v in values if k in inputs}
    assert len(inputs) == result  # Cannot find compatible input value
    return result


@nice_repr
class Pipeline:
    """Represents a sequence of algorithms.

    Each algorithm must have a `run` method declaring it's input and output type.
    The pipeline instance also receives the input and output types.
    """

    def __init__(
        self, algorithms: list[DspyAlgorithmBase], input_types: list[AlgorithmSignature]
    ) -> None:
        self.algorithms = algorithms
        self.input_types = input_types

    def run(self, instruction):
        memory: dict[str, Any] = {}

        memory["instruction"] = instruction
        final_output: Optional[Any] = None

        for i, algorithm in enumerate(self.algorithms):
            args = build_input_args(algorithm, memory)
            output = algorithm.run(**args)
            for k, v in output:
                memory[k] = v

            inputs_to_maintain = set(
                algorithm.get_signature().inputs_fields_to_maintain()
            )
            inputs_to_delete = set(algorithm.input_args()) - inputs_to_maintain
            for key in inputs_to_delete:
                del memory[key]

            if i == len(self.algorithms) - 1:
                final_output = output

        return final_output


class PipelineSpaceBuilder:
    def __init__(self, path_to_llm) -> None:
        self.model = guidance.models.LlamaCpp(path_to_llm)

    def find_initial_valid_nodes(
        self, input_instruction: str, algorithms_pool: set[DspyAlgorithmBase]
    ) -> list[PipelineNode]:
        """Find the nodes of the graph that are valid to stablish an edge from the start node representing the input instruction using an llm that infers what signatures are compatible to the start node."""

        initial_valid_nodes = []
        context = """\
        A signature is a declarative specification of input/output behavior of a DSPy module.\

        Instead of investing effort into how to get your LM to do a sub-task, signatures enable you to inform DSPy what the sub-task is. Later, the DSPy compiler will figure out how to build a complex prompt for your large LM (or finetune your small LM) specifically for your signature, on your data, and within your pipeline.\

        A signature consists of three simple elements:\

        A minimal description of the sub-task the LM is supposed to solve.\
        A description of one or more input fields (e.g., input question) that will we will give to the LM.\
        A description of one or more output fields (e.g., the question's answer) that we will expect from the LM.\

        """
        for a in algorithms_pool:
            current_signature = a.get_signature()
            instruction = f"""\
                Given the instruction: {input_instruction}\
                can the following signature be used: {current_signature}
            """
            lm = (
                context
                + instruction
                + f"Answer: {guidance.select(['yes', 'no'], name='answer')}"
            )

            if lm["answer"] == "no":
                continue

            initial_valid_nodes.append(
                PipelineNode(
                    algorithm=a,
                    input_types=a.get_signature().input_fields(),
                    output_types=a.get_signature().output_fields(),
                    registry=algorithms_pool,
                )
            )

        return initial_valid_nodes

    def build_pipeline_graph(
        self,
        input_instruction: str,
        output_type: type,
        registry: list[DspyAlgorithmBase],
        max_list_depth: int = 3,
    ) -> PipelineSpace:
        """Build a graph of algorithms.

        Every node in the graph corresponds to a <autogoal.grammar.ContextFreeGrammar> that
        generates an instance of a class with a `run` method.

        Each `run` method must declare input and output types in the form:

            def run(self, a: type_1, b: type_2, ...) -> type_n:
                # ...
        """

        # if not isinstance(input_types, (list, tuple)):
        #     input_types = [input_types]

        # We start by enlarging the registry with all Seq[...] algorithms

        pool = set(registry)

        for algorithm in registry:
            for _ in range(max_list_depth):
                algorithm = make_seq_algorithm(algorithm)
                pool.add(algorithm)

        # For building the graph, we'll keep at each node the guaranteed output types

        # We start by collecting all the possible input nodes,
        # those that can process a subset of the input_types
        initial_valid_nodes: list[PipelineNode] = self.find_initial_valid_nodes(
            input_instruction, pool
        )

        G = Graph()

        for node in initial_valid_nodes:
            G.add_edge(GraphSpace.Start, node)

        # We'll make a BFS exploration of the pipeline space.
        # For every open node we will add to the graph every node to which it can connect.
        closed_nodes = set()

        while initial_valid_nodes:
            node = initial_valid_nodes.pop(0)

            # These are the types that are available at this node
            guaranteed_types = node.output_types

            # The node's output type
            node_output_type = node.algorithm.output_type()

            # Here are all the algorithms that could be added new at this point in the graph
            for algorithm in pool:
                if not algorithm.is_compatible_with(guaranteed_types):
                    continue

                # We never want to apply the same exact algorithm twice
                if algorithm == node.algorithm:
                    continue

                # And we never want an algorithm that doesn't provide a novel output type...
                # if (
                #     algorithm.output_type() in guaranteed_types
                #     and
                #     # ... unless it is an idempotent algorithm
                #     tuple([algorithm.output_type()]) != algorithm.input_types()
                # ):
                #     continue

                # BUG: this validation ensures no redundant nodes are added.
                #      The downside is that it prevents pipelines that need two algorithms
                #      to generate the input of another one.

                # And we do not want to ignore the last node's output type
                is_using_last_output = False
                for input_type in algorithm.input_types():
                    if issubclass(node_output_type, input_type):
                        is_using_last_output = True
                        break
                if not is_using_last_output:
                    continue

                p = PipelineNode(
                    algorithm=algorithm,
                    input_types=guaranteed_types,
                    output_types=guaranteed_types | set([algorithm.output_type()]),
                    registry=registry,
                )

                G.add_edge(node, p)

                if p not in closed_nodes and p not in initial_valid_nodes:
                    initial_valid_nodes.append(p)

            # Now we check to see if this node is a possible output
            if issubclass(node.algorithm.output_type(), output_type):
                G.add_edge(node, GraphSpace.End)

            closed_nodes.add(node)

        # Remove all nodes that are not connected to the end node
        try:
            reachable_from_end = set(
                nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End)
            )
            unreachable_nodes = set(G.nodes) - reachable_from_end
            G.remove_nodes_from(unreachable_nodes)
        except KeyError:
            raise TypeError("No pipelines can be found!")

        return PipelineSpace(G, input_types=input_types)
