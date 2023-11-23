import abc
from typing import Any, Optional, Tuple
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.teleprompt import Teleprompter
import dspy
from autogoal_core.kb._algorithm import (
    Algorithm,
    PipelineNode,
    make_seq_algorithm,
)
import networkx as nx
from autogoal_core.utils import nice_repr
from autogoal_core.grammar import Graph, GraphSpace, generate_cfg, Union, Symbol
import guidance
from pydantic import BaseModel, ValidationError, create_model
from json_schema_to_grammar import SchemaConverter
from llama_cpp import Llama, LlamaGrammar


class AlgorithmSignature(Signature):
    @classmethod
    def inputs_fields_to_maintain(cls) -> dict[str, InputField]:
        return {}


class DspyAlgorithmBase(Algorithm):
    """Represents an abstract dspy algorithm with a run method."""

    @classmethod
    def is_teleprompter(cls) -> bool:
        return False

    @classmethod
    @abc.abstractmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        pass

    @classmethod
    def input_types(cls) -> Tuple[type, ...]:
        """Returns an ordered list of the expected semantic input types of the `run` method."""
        signature = cls.get_signature()
        return tuple(
            [
                signature.kwargs[key].annotation
                for key in signature.kwargs
                if isinstance(signature.kwargs[key], InputField)
            ]
        )

    @classmethod
    def input_args(cls) -> Tuple[str, ...]:
        """Returns an ordered tuple of the names of the arguments in the `run` method."""
        names = [name for name in cls.get_signature().kwargs]
        return tuple(names)

    @classmethod
    def output_type(cls) -> Tuple[type, ...]:
        """Returns an ordered list of the expected semantic output type of the `run` method."""
        signature = cls.get_signature()
        return tuple(
            [
                signature.kwargs[key].annotation
                for key in signature.kwargs
                if isinstance(signature.kwargs[key], OutputField)
            ]
            + [v.annotation for v in signature.inputs_fields_to_maintain().values()]
        )

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> dict[str, Any]:
        """Executes the algorithm."""
        pass

    @classmethod
    def is_compatible_with(cls, other: type["DspyAlgorithmBase"]) -> bool:
        if cls.is_teleprompter() or other.is_teleprompter():
            return True

        outputs = {
            k: v for k, v in cls.get_signature().kwargs if isinstance(v, OutputField)
        }
        inputs = {
            k: v for k, v in other.get_signature().kwargs if isinstance(v, InputField)
        }

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


class DspyPipelineSpace(GraphSpace):
    def __init__(self, graph: Graph, path_to_llm: str):
        super().__init__(graph, initializer=self._initialize)
        self.path_to_llm = path_to_llm

    def _initialize(self, item: PipelineNode, sampler):
        return item.sample(sampler)

    def nodes(self) -> set[type[Algorithm]]:
        """Returns a list of all algorithms (types) that exist in the graph."""
        return set(
            node.algorithm
            for node in self.graph.nodes
            if isinstance(node, PipelineNode)
        )

    def sample(self, *args, **kwargs):
        path = super().sample(*args, **kwargs)
        return DspyPipeline(path, path_to_llm=self.path_to_llm)


@nice_repr
class DspyPipeline:
    """Represents a sequence of algorithms.

    Each algorithm must have a `run` method declaring it's input and output type.
    """

    def __init__(
        self,
        algorithms: list[DspyAlgorithmBase],
        path_to_llm: str,
    ) -> None:
        self.algorithms = algorithms
        self.path_to_llm = path_to_llm

    def _get_init(self):
        """Obtains the definition for the __init__ method of the dspy module that is been building"""

        def init(
            self,
            algorithms: list[DspyAlgorithmBase],
            path_to_llm: str,
        ):
            self.algorithms = algorithms
            self.path_to_llm = path_to_llm

        return init

    def _get__extract_arguments_to_call_initial_algorithm(self):
        def _extract_arguments_to_call_initial_algorithm(
            self, input_instruction: str, algorithm: DspyAlgorithmBase
        ) -> dict[str, Any]:
            """Extract arguments related to an AlgorithmSignature from an instruction in natural language using an llm"""

            llm = Llama(self.path_to_llm, n_gpu_layers=-1)
            instruction = f"""
            Given the instruction: {input_instruction}
            And the schema: {algorithm.get_signature().kwargs}
            Extract information from the instruction that match the schema
            """

            pydantic_model, grammar_text = self._generate_grammar_from_signature(
                algorithm.get_signature()
            )
            grammar = LlamaGrammar.from_string(grammar_text)

            response = llm(instruction, grammar=grammar)
            json_response = response["choices"][0]["text"]

            model = pydantic_model.model_validate_json(json_response)
            return model.model_dump()

        return _extract_arguments_to_call_initial_algorithm

    def _get_generate_grammar_from_signature(self):
        def _generate_grammar_from_signature(
            self, signature: AlgorithmSignature
        ) -> tuple[type[BaseModel], str]:
            fields = {
                k: (v.annotation, ...)
                for k, v in signature.kwargs
                if isinstance(v, InputField) or isinstance(v, OutputField)
            }
            DynamicSignatureModel = create_model("DynamicSignatureModel", **fields)
            converter = SchemaConverter({})
            converter.visit(DynamicSignatureModel.model_json_schema(), '')
            grammar_text = converter.format_grammar()
            return DynamicSignatureModel, grammar_text

        return _generate_grammar_from_signature

    def _get_forward(self):
        """Obtains the definition for the forward method of the dspy module that is been building"""

        def forward(self, input_instruction: str):
            memory: dict[str, Any] = {}
            memory["instruction"] = input_instruction
            final_output: Optional[Any] = None
            for i, algorithm in enumerate(self.algorithms):
                if i == 0:
                    args = self._extract_arguments_to_call_initial_algorithm(
                        input_instruction, algorithm
                    )
                else:
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

        return forward

    def run(self, trainset):
        teleprompter = self.algorithms.pop()
        assert teleprompter.is_teleprompter()
        dspy_module = type(
            "GeneratedDspyModule",
            (dspy.Module,),
            {
                "__init__": self._get_init(),
                "_generate_grammar_from_signature": self._get_generate_grammar_from_signature(),
                "_extract_arguments_to_call_initial_algorithm": self._get__extract_arguments_to_call_initial_algorithm(),
                "forward": self._get_forward(),
            },
        )

        compiled_program = teleprompter.run(dspy_module(), trainset=trainset)
        return compiled_program


class PipelineSpaceBuilder:
    def __init__(self, path_to_llm: str) -> None:
        self.path_to_llm = path_to_llm

    def find_initial_valid_nodes(
        self,
        dataset_description: str,
        algorithms_pool: set[DspyAlgorithmBase],
    ) -> list[PipelineNode]:
        """Find the nodes of the graph that are valid to stablish an edge from
        the start node(representing a high level description of the dataset that is going to be used),
        using an llm that infers what signatures are compatible to the start node.
        """
        model = guidance.models.LlamaCpp(self.path_to_llm)

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
                Given the dataset_description: {dataset_description}\
                can the following signature be used directly?: {repr(current_signature)}
            """
            lm = (
                model
                + context
                + instruction
                + f"Answer: {guidance.select(['yes', 'no'], name='answer')}"
            )

            if lm["answer"] == "no":
                continue

            initial_valid_nodes.append(
                PipelineNode(
                    algorithm=a,
                    input_types=a.input_types(),
                    output_types=a.output_type(),
                    registry=algorithms_pool,
                )
            )

        return initial_valid_nodes

    def _extract_arguments_to_call_algorithm(self):
        pass

    def build_pipeline_graph(
        self,
        dataset_description: str,
        output_type: type,
        registry: list[DspyAlgorithmBase],
        max_list_depth: int = 3,
    ) -> DspyPipelineSpace:
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

        teleprompters_in_registry = [x for x in pool if x.is_teleprompter()]
        assert len(teleprompters_in_registry) == 1
        teleprompter = teleprompters_in_registry.pop()
        pool.remove(teleprompter)
        # This is the last step of the pipeline, after sampling this node we can stop.
        # Every node in the graph will have an edge to this node and this node will have an edge to the end node
        teleprompter_node = PipelineNode(
            teleprompter, input_types=None, output_types=None, registry=registry
        )

        for algorithm in registry:
            for _ in range(max_list_depth):
                algorithm = make_seq_algorithm(algorithm)
                pool.add(algorithm)

        # For building the graph, we'll keep at each node the guaranteed output types

        # We start by collecting all the possible input nodes,
        # those that can process a subset of the input_types
        initial_valid_nodes: list[PipelineNode] = self.find_initial_valid_nodes(
            dataset_description, pool
        )

        G = Graph()

        for node in initial_valid_nodes:
            G.add_edge(GraphSpace.Start, node)

        # We'll make a BFS exploration of the pipeline space.
        # For every open node we will add to the graph every node to which it can connect.
        closed_nodes = set()

        while initial_valid_nodes:
            node = initial_valid_nodes.pop(0)
            G.add_edge(node, teleprompter_node)
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

        G.add_edge(teleprompter_node, GraphSpace.End)

        # Remove all nodes that are not connected to the end node
        try:
            reachable_from_end = set(
                nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End)
            )
            unreachable_nodes = set(G.nodes) - reachable_from_end
            G.remove_nodes_from(unreachable_nodes)
        except KeyError:
            raise TypeError("No pipelines can be found!")

        return DspyPipelineSpace(G, path_to_llm=self.path_to_llm)
