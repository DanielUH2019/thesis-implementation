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
from memory import Memory, MemoryDataModel, ValueStoreObjectModel


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
    def input_types(cls) -> list[InputField]:
        """Returns a list of the expected {InputField} of the `run` method."""
        signature = cls.get_signature()
        return [
            signature.kwargs[i]
            for i in signature.kwargs
            if isinstance(signature.kwargs[i], InputField)
        ]

    @classmethod
    def input_args(cls) -> Tuple[str, ...]:
        """Returns an ordered tuple of the names of the arguments in the `run` method."""
        signature = cls.get_signature()
        names = [
            name
            for name in signature.kwargs
            if isinstance(signature.kwargs[name], InputField)
        ]
        return tuple(names)

    @classmethod
    def output_type(cls) -> list[OutputField]:
        """Returns a list of the expected {OutputField} of the `run` method."""
        signature = cls.get_signature()
        return [
            signature.kwargs[key].annotation
            for key in signature.kwargs
            if isinstance(signature.kwargs[key], OutputField)
        ] + [v for v in signature.inputs_fields_to_maintain().values()]

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> dict[str, Any]:
        """Executes the algorithm."""
        pass

    @classmethod
    def is_compatible_with(
        cls, memory: Memory, similarity_score=0.6, modify_memory=True
    ) -> bool:
        """
        Determines if the current algorithm can be called using the data stored in the memory,
        i.e., based on the descriptions of the required input for this algorithm,
        search in the memory data that semantically matches that descriptions

        if modify_memory is True and the algorithm is compatible it will replace the corresponding values from memory
        """

        inputs = cls.input_types()
        matches_id = []
        for i in inputs:
            results = memory.query(str(i.desc), limit=1)
            if results[0].score >= similarity_score:
                matches_id.append(results[0].id)
            else:
                return False

        if not modify_memory:
            return True

        for id in matches_id:
            memory.delete_from_db(id)

        for o in cls.output_type():
            memory.insert(o.desc)

        return True


class DspyPipelineSpace(GraphSpace):
    def __init__(self, graph: Graph, path_to_llm: str):
        super().__init__(graph, initializer=self._initialize)
        self.path_to_llm = path_to_llm

    def _initialize(self, item: PipelineNode, sampler):
        return item.sample(sampler)

    def nodes(self) -> set[DspyAlgorithmBase]:
        """Returns a list of all algorithms (types) that exist in the graph."""
        return set(
            node.algorithm
            for node in self.graph.nodes
            if isinstance(node, DspyPipelineNode)
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

    def run(self, trainset):
        teleprompter = self.algorithms.pop()
        assert teleprompter.is_teleprompter()
        dspy_module_generator = DspyModuleGenerator(self.algorithms, self.path_to_llm)

        compiled_program = teleprompter.run(dspy_module_generator(), trainset=trainset)
        return compiled_program


class DspyPipelineNode(PipelineNode):
    def __init__(self, algorithm: DspyAlgorithmBase, registry=None) -> None:
        self.algorithm = algorithm
        self.input_types = algorithm.input_types()
        self.output_types = algorithm.output_type()
        self.grammar = generate_cfg(self.algorithm, registry=registry)

        def __eq__(self, o: "DspyPipelineNode") -> bool:
            return isinstance(o, DspyPipelineNode) and o.algorithm == self.algorithm


class DspyModuleGenerator(dspy.Module):
    def __init__(
        self,
        algorithms: list[DspyAlgorithmBase],
        path_to_llm: str,
    ) -> None:
        self.algorithms = algorithms
        self.path_to_llm = path_to_llm

    def forward(self, input_instruction: str):
        memory = Memory()
        memory.insert_to_value_store(
            [
                ValueStoreObjectModel(
                    key_name="instruction",
                    description="input_instruction",
                    value=input_instruction,
                )
            ]
        )
        final_output: Optional[Any] = None
        for i, algorithm in enumerate(self.algorithms):
            if i == 0:
                args = self._extract_arguments_to_call_initial_algorithm(
                    input_instruction, algorithm
                )
            else:
                args = self._build_input_args(algorithm, memory)
            output = algorithm.run(**args)
            inputs_to_maintain = set(
                algorithm.get_signature().inputs_fields_to_maintain()
            )
            inputs_to_delete = set(algorithm.input_args()) - inputs_to_maintain
            for key in inputs_to_delete:
                memory.delete_from_store(key)
            for k, v, desc in output:
                memory.insert_to_value_store(
                    [ValueStoreObjectModel(key_name=k, value=v, description=desc)]
                )
            if i == len(self.algorithms) - 1:
                final_output = output
        return final_output

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
        converter.visit(DynamicSignatureModel.model_json_schema(), "")
        grammar_text = converter.format_grammar()
        return DynamicSignatureModel, grammar_text

    def _build_input_args(
        self, algorithm: DspyAlgorithmBase, memory: Memory
    ) -> dict[str, Any]:
        """Buils the correct input mapping for `algorithm` using the provided `values` mapping types to objects."""
        required_input_keys = set(
            [
                k
                for k, v in algorithm.get_signature().kwargs
                if isinstance(v, InputField)
            ]
        )

        avaliable_keys = set(
            [key for key in memory.value_store if key in required_input_keys]
        )
        unmatched_keys = required_input_keys - avaliable_keys
        result = {k: memory.value_store[k] for k in avaliable_keys}
        if len(unmatched_keys) > 0:
            most_similar_keys = [
                memory.retrieve_stored_value(k, v.desc)
                for k, v in algorithm.get_signature().kwargs
                if k in unmatched_keys
            ]
            result.update(most_similar_keys)

        if len(result) != len(required_input_keys):
            raise ValueError(
                f"Could not find enough arguments to call {algorithm.get_signature()}"
            )
        return result


class PipelineSpaceBuilder:
    def __init__(self, path_to_llm: str) -> None:
        self.path_to_llm = path_to_llm

    def find_initial_valid_nodes(
        self,
        dataset_description: str,
        algorithms_pool: set[DspyAlgorithmBase],
    ) -> list[DspyPipelineNode]:
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
                DspyPipelineNode(
                    algorithm=a,
                    registry=algorithms_pool,
                )
            )

        return initial_valid_nodes

    def _extract_arguments_to_call_algorithm(self):
        pass

    def build_pipeline_graph(
        self,
        dataset_description: str,
        registry: list[DspyAlgorithmBase],
    ) -> DspyPipelineSpace:
        """Build a graph of algorithms.

        Every node in the graph corresponds to a <autogoal.grammar.ContextFreeGrammar> that
        generates an instance of a class with a `run` method.

        """

        pool = set(registry)

        teleprompters_in_registry = [x for x in pool if x.is_teleprompter()]
        assert len(teleprompters_in_registry) == 1
        teleprompter = teleprompters_in_registry.pop()
        pool.remove(teleprompter)
        # This is the last step of the pipeline, after sampling this node we can stop.
        # Every node in the graph will have an edge to this node and this node will have an edge to the end node
        teleprompter_node = DspyPipelineNode(teleprompter, registry=registry)

        # We start by collecting all the possible input nodes,
        # those that are found compatible with the dataset description
        initial_valid_nodes: list[DspyPipelineNode] = self.find_initial_valid_nodes(
            dataset_description, pool
        )

        G = Graph()

        for node in initial_valid_nodes:
            G.add_edge(GraphSpace.Start, node)

        # We'll make a DFS exploration of the pipeline space from every initial valid node.
        # For every open node we will add to the graph every node to which it can connect.

        self._dfs(G, initial_valid_nodes, pool, teleprompter_node, registry)

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

    def _dfs(
        self,
        G,
        initial_valid_nodes: list[DspyPipelineNode],
        pool: set[DspyAlgorithmBase],
        teleprompter_node: DspyPipelineNode,
        registry: list[DspyAlgorithmBase],
    ):
        for start_node in initial_valid_nodes:
            stack = [start_node]
            closed_nodes = set()

            # Memory to be used for simulating algorithms execution and stablish compatibility
            memory = Memory()

            while stack:
                node = stack.pop()
                G.add_edge(node, teleprompter_node)

                # Here are all the algorithms that could be added new at this point in the graph
                for algorithm in pool:
                    if not algorithm.is_compatible_with(memory):
                        continue

                    # We never want to apply the same exact algorithm twice
                    if algorithm == node.algorithm:
                        continue

                    p = DspyPipelineNode(
                        algorithm=algorithm,
                        registry=registry,
                    )

                    G.add_edge(node, p)

                    if p not in closed_nodes and p not in initial_valid_nodes:
                        initial_valid_nodes.append(p)

                # TODO Find a way to check if the last node would be sufficient to generate a valid solution
                # for a problem based on the dataset description, maybe with a dspy program
                G.add_edge(node, teleprompter_node)

                closed_nodes.add(node)
