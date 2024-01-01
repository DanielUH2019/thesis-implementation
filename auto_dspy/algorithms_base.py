import abc
from ast import arg
from typing import Any, Callable, Optional, Tuple
from uuid import uuid4
from dspy.signatures.field import InputField, OutputField, Field
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
import warnings
import outlines
from prompts import (
    instruction_to_extract_arguments_to_json,
    instruction_to_extract_arguments_to_json,
)


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
    def output_type(cls) -> tuple[dict[str, OutputField], dict[str, InputField]]:
        """Returns a list of the expected {OutputField} of the `run` method."""
        signature = cls.get_signature()

        outputs = {
            k: v for k, v in signature.kwargs.items() if isinstance(v, OutputField)
        }
        return outputs, signature.inputs_fields_to_maintain()

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Executes the algorithm."""
        pass

    @classmethod
    def is_compatible_with(
        cls, memory: Memory, similarity_score=0.6
    ) -> tuple[bool, dict[str, InputField]]:
        """
        Determines if the current algorithm can be called using the data stored in the memory,
        i.e., based on the descriptions of the required input for this algorithm,
        search in the memory data that semantically matches that descriptions

        if modify_memory is True and the algorithm is compatible it will replace the corresponding values from memory
        """

        inputs = cls.input_types()
        matches = {}
        for i in inputs:
            results = memory.query(str(i.desc), limit=1)
            # print(f"len de resultados query {results}")
            if (
                len(results["distances"][0]) == 0
                or results["distances"][0][0] < similarity_score
            ):
                return False, {}

            matches[results["ids"][0][0]] = i
        return True, matches


@nice_repr
class DspyPipelineNode(PipelineNode):
    def __init__(self, algorithm: type[DspyAlgorithmBase], registry=None) -> None:
        self.algorithm = algorithm
        # print(f"algorithm {algorithm}")
        self.input_types = algorithm.input_types()
        self.output_types = algorithm.output_type()
        self.grammar = generate_cfg(self.algorithm, registry=registry)

        def __eq__(self, o: "DspyPipelineNode") -> bool:
            return isinstance(o, DspyPipelineNode) and o.algorithm == self.algorithm


@nice_repr
class DspyPipelineSpace(GraphSpace):
    def __init__(
        self,
        graph: Graph,
        path_to_llm: str,
        examples_description: dict[str, dict[str, str]],
    ):
        super().__init__(graph, initializer=self._initialize)
        self.path_to_llm = path_to_llm
        self.examples_description = examples_description

    def _initialize(self, item: DspyPipelineNode, sampler):
        return item.sample(sampler)

    def nodes(self) -> set[type[DspyAlgorithmBase]]:
        """Returns a list of all algorithms (types) that exist in the graph."""
        return set(
            node.algorithm
            for node in self.graph.nodes
            if isinstance(node, DspyPipelineNode)
        )

    def sample(self, *args, **kwargs):
        path = super().sample(*args, **kwargs)
        return DspyPipeline(
            path,
            path_to_llm=self.path_to_llm,
            examples_description=self.examples_description,
        )


# @nice_repr
# class DspyPipeline:
#     """Represents a sequence of algorithms.

#     Each algorithm must have a `run` method declaring it's input and output type.
#     """

#     def __init__(
#         self,
#         algorithms: list[DspyAlgorithmBase],
#         path_to_llm: str,
#         examples_description: dict[str, dict[str, str]],
#     ) -> None:
#         self.algorithms = algorithms
#         self.path_to_llm = path_to_llm
#         self.examples_description = examples_description

#     def run(self, trainset, metric: Callable):
#         teleprompter = self.algorithms.pop()
#         assert teleprompter.is_teleprompter()
#         dspy_module_generator = DspyModuleGenerator(
#             self.algorithms, self.path_to_llm, self.examples_description
#         )

#         compiled_program = teleprompter.run(
#             metric=metric,
#             dspy_module=dspy_module_generator,
#             trainset=trainset,
#         )
#         return compiled_program


@nice_repr
class DspyPipeline:
    """Represents a sequence of algorithms.

    Each algorithm must have a `run` method declaring it's input and output type.
    """

    def __init__(
        self,
        algorithms: list[DspyAlgorithmBase],
        path_to_llm: str,
        examples_description: dict[str, dict[str, str]],
    ) -> None:
        self.algorithms = algorithms
        self.path_to_llm = path_to_llm
        self.examples_description = examples_description

    def run(self, trainset, metric: Callable):
        teleprompter = self.algorithms.pop()
        assert teleprompter.is_teleprompter()
        dspy_module_generator = DspyModuleGenerator(
            self.algorithms, self.path_to_llm, self.examples_description
        )

        compiled_program = teleprompter.run(
            metric=metric,
            dspy_module=dspy_module_generator,
            trainset=trainset,
        )
        return compiled_program


@nice_repr
class DspyModuleGenerator(dspy.Module):
    def __init__(
        self,
        algorithms: list[DspyAlgorithmBase],
        path_to_llm: str,
        examples_description: dict[str, dict[str, str]],
    ) -> None:
        super().__init__()
        self.algorithms = algorithms
        self.path_to_llm = path_to_llm
        self.examples_description = examples_description

    def forward(self, **kwargs):
        memory = Memory(collection_name=str(uuid4()))
        for k, v in kwargs.items():
            memory.insert_to_value_store(
                [
                    ValueStoreObjectModel(
                        key_name=k,
                        description=self.examples_description["inputs"][k],
                        value=v,
                    )
                ]
            )
        print("debo haber insertado", kwargs)
        final_output: Optional[Any] = None
        print("alll data", memory.get_all_data())
        for i, algorithm in enumerate(self.algorithms):
            # if i == 0:
            #     args = self._extract_arguments_to_call_initial_algorithm(
            #         algorithm=algorithm, **kwargs
            #     )
            # else:
            print(f"antes de correr {memory.get_all_data()}")
            args = self._build_input_args(algorithm, memory)
            print(f"builded args {args}")
            output_values = list(algorithm.run(**args))
            # inputs_to_maintain = set(
            #     algorithm.get_signature().inputs_fields_to_maintain().keys()
            # )

            # inputs_to_delete = set(algorithm.input_args()) - inputs_to_maintain
            output_types, _ = algorithm.output_type()
            # for k, v in algorithm.get_signature().kwargs.items():
            #     if isinstance(v, InputField) and k in inputs_to_delete:
            #         memory.delete_from_store([v.desc])
            counter = 0
            for k, v in output_types.items():
                memory.insert_to_value_store(
                    [
                        ValueStoreObjectModel(
                            key_name=k, value=output_values[counter], description=v.desc
                        )
                    ]
                )
                counter += 1

            if i == len(self.algorithms) - 1:
                final_output = output_values
        output_key = list(self.examples_description["outputs"].keys())[0]
        prediction_kwargs = {output_key: final_output[0]}
        prediction_to_return = dspy.Prediction(**prediction_kwargs)
        return prediction_to_return

    # def _extract_arguments_to_call_initial_algorithm(
    #     self, algorithm: DspyAlgorithmBase, **kwargs
    # ) -> dict[str, Any]:
    #     """Extract arguments related to an AlgorithmSignature from an instruction in natural language using an llm"""

    #     # llm = Llama(self.path_to_llm, n_gpu_layers=-1)
    #     signature = algorithm.get_signature()
    #     fields = {
    #         k: (v.annotation, ...)
    #         for k, v in algorithm.get_signature().kwargs.items()
    #         if isinstance(v, InputField)
    #     }
    #     DynamicSignatureModel = create_model("DynamicSignatureModel", **fields)

    #     llm = outlines.models.transformers(self.path_to_llm, device="cuda")
    #     generator = outlines.generate.json(llm, DynamicSignatureModel)
    #     # instruction = f"""
    #     #     Given the inputs: {kwargs}
    #     #     And the schema: {algorithm.get_signature().kwargs}
    #     #     Extract information from the instruction that match the schema
    #     #     """
    #     kwargs_with_descriptions = {k: v.desc for k, v in signature.kwargs.items()}
    #     required_arguments = {
    #         "arguments": [signature.kwargs.keys()],
    #         "descriptions": kwargs_with_descriptions,
    #     }
    #     # pydantic_model, grammar_text = self._generate_grammar_from_signature(
    #     #     algorithm.get_signature()
    #     # )
    #     # grammar = LlamaGrammar.from_string(grammar_text)

    #     # response = llm(instruction, grammar=grammar, temperature=0.0)
    #     prompt = instruction_to_extract_arguments_to_json(kwargs, required_arguments)
    #     json_response = generator(prompt)
    #     # json_response = response["choices"][0]["text"]
    #     # print(f"schema {DynamicSignatureModel}")
    #     assert isinstance(json_response, DynamicSignatureModel)
    #     # print(f"generated json {json_response} with type: {type(json_response)}")
    #     # model = DynamicSignatureModel.model_validate_json(json_response)
    #     return json_response.model_dump()

    # def _generate_grammar_from_signature(
    #     self, signature: type[AlgorithmSignature]
    # ) -> tuple[type[BaseModel], str]:

    #     sequence = generator("Give me a character description")
    #     converter = SchemaConverter({})
    #     converter.visit(DynamicSignatureModel.model_json_schema(), "")
    #     grammar_text = converter.format_grammar()
    #     return DynamicSignatureModel, grammar_text

    def _build_input_args(
        self, algorithm: DspyAlgorithmBase, memory: Memory
    ) -> dict[str, Any]:
        """Buils the correct input mapping for `algorithm` using the provided `values` mapping types to objects."""
        required_input_keys = set(
            [
                k
                for k, v in algorithm.get_signature().kwargs.items()
                if isinstance(v, InputField)
            ]
        )

        # avaliable_keys = set(
        #     [key for key in memory.value_store if key in required_input_keys]
        # )
        # unmatched_keys = required_input_keys - avaliable_keys
        # result = {k: memory.value_store[k] for k in avaliable_keys}
        inputs_to_maintain = set(
                algorithm.get_signature().inputs_fields_to_maintain().keys()
            )

        inputs_to_delete = set(algorithm.input_args()) - inputs_to_maintain
        result = {}
        # if len(unmatched_keys) > 0:
        # most_similar = []
        for k, v in algorithm.get_signature().kwargs.items():
            if k in required_input_keys:
                id, _, value = memory.retrieve_stored_value(v.desc)
                result[k] = value
                # if k in inputs_to_delete:
                #     memory.delete_from_store([id])
                    
        # most_similar = [
        #     (k, memory.retrieve_stored_value(v.desc)[1])
        #     for k, v in algorithm.get_signature().kwargs.items()
        #     if k in required_input_keys
        # ]
        # result.update(most_similar)

        if len(result) != len(required_input_keys):
            raise ValueError(
                f"Could not find enough arguments to call {algorithm.get_signature()}"
            )
        return result


class PipelineSpaceBuilder:
    def __init__(
        self, path_to_llm: str, examples_description: dict[str, dict[str, str]]
    ) -> None:
        self.path_to_llm = path_to_llm
        self.examples_description = examples_description

    def find_initial_valid_nodes(
        self,
        dataset_description: str,
        algorithms_pool: set[type[DspyAlgorithmBase]],
    ) -> list[DspyPipelineNode]:
        """Find the nodes of the graph that are valid to stablish an edge from
        the start node(representing a high level description of the dataset that is going to be used),
        using an llm that infers what signatures are compatible to the start node.
        """
        # model = guidance.models.LlamaCpp(self.path_to_llm)

        initial_valid_nodes = []
        # context = """\
        # A signature is a declarative specification of input/output behavior of a DSPy module.\

        # Instead of investing effort into how to get your LM to do a sub-task, signatures enable you to inform DSPy what the sub-task is. Later, the DSPy compiler will figure out how to build a complex prompt for your large LM (or finetune your small LM) specifically for your signature, on your data, and within your pipeline.\

        # A signature consists of three simple elements:\

        # A minimal description of the sub-task the LM is supposed to solve.\
        # A description of one or more input fields (e.g., input question) that will we will give to the LM.\
        # A description of one or more output fields (e.g., the question's answer) that we will expect from the LM.\

        # """
        # llm = outlines.models.transformers(self.path_to_llm, device="cuda")
        memory = Memory(str(uuid4()))
        for k, v in self.examples_description["inputs"].items():
            memory.insert([v], None)
        for a in algorithms_pool:
            # current_signature = a.input_types()

            if not a.is_compatible_with(memory):
                continue

            # instruction = f"""\
            #     Given the dataset_description: {dataset_description}\
            #     can the following signature be used directly?: {repr(current_signature)}
            # """
            # lm = (
            #     model
            #     + context
            #     + instruction
            #     + f"Answer: {guidance.select(['yes', 'no'], name='answer')}"
            # )
            # prompt = instruction_to_extract_arguments_to_json(
            #     dataset_description, repr(current_signature)
            # )
            # answer = outlines.generate.choice(llm, ["True", "False"])(prompt)
            # if lm["answer"] == "no":
            #     continue

            # if answer == "False":
            #     continue

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
        registry: list[type[DspyAlgorithmBase]],
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
        for start_node in initial_valid_nodes:
            initial_memory = Memory(str(uuid4()))
            output_types, inputs_to_mantain = start_node.algorithm.output_type()
            print("output types", output_types)
            initial_memory.insert([v.desc for k, v in output_types.items()], None)
            initial_memory.insert([v.desc for k, v in inputs_to_mantain.items()], None)
            self._dfs(
                G,
                start_node,
                teleprompter_node,
                initial_memory,
                set(),
                pool,
                initial_valid_nodes,
                registry,
            )

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

        pipeline_space = DspyPipelineSpace(
            G,
            path_to_llm=self.path_to_llm,
            examples_description=self.examples_description,
        )
        print(f"pipeline space {pipeline_space}")
        nx.draw_shell(G, with_labels=True, font_weight="bold")
        return pipeline_space

    # def _dfs(
    #     self,
    #     G,
    #     initial_valid_nodes: list[DspyPipelineNode],
    #     pool: set[type[DspyAlgorithmBase]],
    #     teleprompter_node: DspyPipelineNode,
    #     registry: list[type[DspyAlgorithmBase]],
    # ):
    #     for start_node in initial_valid_nodes:
    #         stack = [(start_node, {})]
    #         closed_nodes = set()

    #         # Memory to be used for simulating algorithms execution and stablish compatibility
    #         memory = Memory()
    # output_types = start_node.algorithm.output_type()
    # memory.insert([x.desc for x in output_types[0]])
    # memory.insert([x.desc for x in output_types[1]])
    # memory.insert(list(start_node.algorithm.input_args()))
    #         while stack:
    #             node, node_matches = stack.pop()

    #             # G.add_edge(node, teleprompter_node)
    #             if node != start_node:

    #             # Here are all the algorithms that could be added new at this point in the graph
    #             for algorithm in pool:
    #                 is_compatible, matches = algorithm.is_compatible_with(memory)
    #                 if not is_compatible:
    #                     continue

    #                 # We never want to apply the same exact algorithm twice
    #                 if algorithm == node.algorithm:
    #                     continue

    #                 p = DspyPipelineNode(
    #                     algorithm=algorithm,
    #                     registry=registry,
    #                 )

    #                 G.add_edge(node, p)

    #                 if p not in closed_nodes and p not in initial_valid_nodes:
    #                     stack.append((p, matches))

    #             # TODO Find a way to check if the last node would be sufficient to generate a valid solution
    #             # for a problem based on the dataset description, maybe with a dspy program
    #             G.add_edge(node, teleprompter_node)

    #             closed_nodes.add(node)

    def _dfs(
        self,
        G: Graph,
        node: DspyPipelineNode,
        telepropmter_node: DspyPipelineNode,
        memory: Memory,
        closed_nodes: set[DspyPipelineNode],
        pool: set[type[DspyAlgorithmBase]],
        initial_valid_nodes: list[DspyPipelineNode],
        registry: list[type[DspyAlgorithmBase]],
    ):
        for algorithm in pool:
            is_compatible, matches = algorithm.is_compatible_with(memory)
            if not is_compatible or algorithm == node.algorithm:
                continue

            p = DspyPipelineNode(
                algorithm=algorithm,
                registry=registry,
            )

            G.add_edge(node, p)
            G.add_edge(node, telepropmter_node)

            if p not in closed_nodes and p not in initial_valid_nodes:
                closed_nodes.add(p)
                data_copy = memory.get_all_data()
                cloned_memory = Memory(str(uuid4()))
                cloned_memory.upload_records(
                    documents=data_copy["documents"],
                    ids=data_copy["ids"],
                    metadatas=data_copy["metadatas"],
                )
                o, i = node.algorithm.output_type()
                for key in matches:
                    if key not in i:
                        cloned_memory.delete_from_db(key)
                    else:
                        cloned_memory.update(key, matches[key].desc)

                cloned_memory.insert([v.desc for k, v in o.items()], None)
                self._dfs(
                    G,
                    p,
                    telepropmter_node,
                    cloned_memory,
                    closed_nodes,
                    pool,
                    initial_valid_nodes,
                    registry,
                )
