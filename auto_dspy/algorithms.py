from typing import Callable, Optional
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature
from dspy.predict import ReAct, ChainOfThought, ProgramOfThought

from algorithms_base import AlgorithmSignature, DspyAlgorithmBase, DspyModuleGenerator
from autogoal_core.grammar import (
    CategoricalValue,
    ContinuousValue,
    DiscreteValue,
    BooleanValue,
)
from signatures import (
    CodeGenerator,
    GenerateAnswer,
    BasicQA,
    GenerateSearchQuery,
    ImageCaptioning,
    ImageClassification,
    LanguageTranslation,
    SummarizeText,
    TextClassification,
    ZeroShotTextClassification,
    ZeroShotImageClassification,
    TeleprompterSignature,
    GenerateQuestion,
    WikipediaRetrieval,
)

from hf_tasks import (
    Task,
    IMAGE_CLASSIFICATION,
    TRANSLATION,
    SUMMARIZATION,
    TEXT_CLASSIFICATION,
    TEXT_GENERATION,
    TOKEN_CLASSIFICATION,
    ZERO_SHOT_CLASSIFICATION,
    AUDIO_CLASSIFICATION,
    AUDIO_TO_AUDIO,
    SENTENCE_SIMILARITY,
    TABLE_QUESTION_ANSWERING,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from constants import (
    TIMM_MODELS_SUPPORTED,
    TRANSFORMERS_MODELS_SUPPORTED,
    REACT_MODULE,
    COT_MODULE,
    POT_MODULE,
)
from utils import instantiate_prompt_module

from model_backends import (
    TimmBackend,
    TransformersBackend,
    TransformersBackendConfig,
    TimmBackendConfig,
)

from autogoal_core.utils import nice_repr

from metaphor_python import Metaphor
import os
import interpreter
from dspy.teleprompt import BootstrapFewShot
import dspy


def run_algorithm_with_transformers_or_timm(
    model_name, task: Optional[Task] = None, *args
):
    if model_name in TIMM_MODELS_SUPPORTED:
        assert len(args) == 1, "Timm models only accept one input (image)"
        config = TimmBackendConfig(model=model_name)
        return TimmBackend(config).run(*args)
    elif model_name in TRANSFORMERS_MODELS_SUPPORTED:
        assert task is not None, "Task must be specified for transformers models"
        config = TransformersBackendConfig(model=model_name, task=task.task_name)
        return TransformersBackend(config).run(*args)
    else:
        raise ValueError(
            f"Model {model_name} , task {task} and args: {args} not supported"
        )


@nice_repr
class TeleprompterAlgorithm(DspyAlgorithmBase):
    def __init__(self) -> None:
        pass

    @classmethod
    def is_teleprompter(cls) -> bool:
        return True

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return TeleprompterSignature

    def run(self, metric: Callable, dspy_module: DspyModuleGenerator, trainset):
        teleprompter = BootstrapFewShot(metric=metric)
        return teleprompter.compile(dspy_module, trainset=trainset)


@nice_repr
class GenerateAnswerAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(COT_MODULE),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return GenerateAnswer

    def run(self, context, question):
        prompt_module = instantiate_prompt_module(
            self.prompt_technique, self.get_signature()
        )
        kwargs = {"context": context, "question": question}
        output_name = [
            k
            for k, v in self.get_signature().kwargs.items()
            if isinstance(v, OutputField)
        ][0]
        result = prompt_module(**kwargs)
        return result.completions[output_name]


@nice_repr
class BasicQAAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(COT_MODULE),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return BasicQA

    def run(self, question):
        prompt_module = instantiate_prompt_module(
            self.prompt_technique, self.get_signature()
        )
        kwargs = {"question": question}
        output_name = [
            k
            for k, v in self.get_signature().kwargs.items()
            if isinstance(v, OutputField)
        ][0]
        result = prompt_module(**kwargs)
        return result.completions[output_name]


# @nice_repr
# class GenerateAndRunSearchQueryAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         prompt_technique: CategoricalValue(COT_MODULE),
#     ):
#         self.prompt_technique = prompt_technique

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return GenerateSearchQuery

#     def run(self, question):
#         prompt_module = instantiate_prompt_module(
#             self.prompt_technique, self.get_signature()
#         )
#         kwargs = {"question": question}
#         output_name = [
#             k
#             for k, v in self.get_signature().kwargs.items()
#             if isinstance(v, OutputField)
#         ][0]
#         result = prompt_module(**kwargs)
#         search_query = result.completions[output_name]
#         METAPHOR_API_KEY = os.getenv("METAPHOR_API_KEY")
#         assert (
#             METAPHOR_API_KEY is not None
#         ), "You need to put a METAPHOR_API_KEY in a .env to be able to run search queries"
#         metaphor = Metaphor(METAPHOR_API_KEY)
#         if isinstance(search_query, list):
#             search_query = search_query[0]
#         search_response = metaphor.search(search_query, use_autoprompt=True)
#         contents_result = search_response.get_contents()
#         return [c.extract for c in contents_result.contents][0]


# @nice_repr
# class ImageCaptioningAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         prompt_technique: CategoricalValue(REACT_MODULE, COT_MODULE),
#     ):
#         self.prompt_technique = prompt_technique

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return ImageCaptioning

#     def run(self, image):
#         prompt_module = instantiate_prompt_module(
#             self.prompt_technique, self.get_signature()
#         )
#         kwargs = {"image": image}
#         return prompt_module(**kwargs)


# @nice_repr
# class ImageClassificationAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         models: CategoricalValue(
#             "mobilenetv3_large_100",
#             "adv_inception_v3",
#             "cspdarknet53",
#             "cspresnext50",
#             "densenet121",
#             "densenet161",
#             "densenet169",
#             "densenet201",
#             "densenetblur121d",
#             "dla34",
#             "dla46_c",
#         ),
#     ):
#         self.model_name = models

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return ImageClassification

#     def run(self, image):
#         return run_algorithm_with_transformers_or_timm(
#             self.model_name, IMAGE_CLASSIFICATION, image
#         )


# @nice_repr
# class LanguageTranslationAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         models: CategoricalValue(
#             "t5-small", "t5-base", "t5-large", "mbart-large-50-many-to-many-mmt"
#         ),
#     ):
#         self.model_name = models

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return LanguageTranslation

#     def run(self, text):
#         return run_algorithm_with_transformers_or_timm(
#             self.model_name, TRANSLATION, text
#         )


@nice_repr
class SummarizeTextAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(COT_MODULE),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return SummarizeText

    def run(self, text):
        prompt_module = instantiate_prompt_module(
            self.prompt_technique, self.get_signature()
        )
        kwargs = {"text": text}
        output_name = [
            k
            for k, v in self.get_signature().kwargs.items()
            if isinstance(v, OutputField)
        ][0]
        result = prompt_module(**kwargs)
        return result.completions[output_name]


# @nice_repr
# class TextClassificationAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         models: CategoricalValue(
#             "distilbert-base-uncased",
#         ),
#     ):
#         self.model_name = models

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return TextClassification

#     def run(self, text):
#         return run_algorithm_with_transformers_or_timm(
#             self.model_name, TEXT_CLASSIFICATION, text
#         )[0]["label"]


# @nice_repr
# class ZeroShotTextClassificationAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         models: CategoricalValue(
#             "facebook/bart-large-mnli",
#             "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
#             "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
#             "facebook/bart-large-mnli",
#             "joeddav/xlm-roberta-large-xnli",
#         ),
#     ):
#         self.model_name = models

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return ZeroShotTextClassification

#     def run(self, text):
#         return run_algorithm_with_transformers_or_timm(
#             self.model_name, ZERO_SHOT_CLASSIFICATION, text
#         )


# @nice_repr
# class ZeroShotImageClassificationAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         models: CategoricalValue(
#             "openai/clip-vit-large-patch14",
#         ),
#     ):
#         self.model_name = models

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return ZeroShotImageClassification

#     def run(self, image):
#         return run_algorithm_with_transformers_or_timm(
#             self.model_name, ZERO_SHOT_IMAGE_CLASSIFICATION, image
# )


# @nice_repr
# class CodeGeneratorAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         prompt_technique: CategoricalValue(COT_MODULE),
#     ) -> None:
#         self.prompt_technique = prompt_technique

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return CodeGenerator

#     def run(self, context, instruction):
#         prompt_module = instantiate_prompt_module(
#             self.prompt_technique, self.get_signature()
#         )
#         kwargs = {"context": context, "instruction": instruction}
#         output_name = [
#             k
#             for k, v in self.get_signature().kwargs.items()
#             if isinstance(v, OutputField)
#         ][0]
#         result = prompt_module(**kwargs)
#         code = result.completions[output_name]
#         if not isinstance(self.prompt_technique, ProgramOfThought):
#             interpreter.local = True
#             interpreter.temperature = 0
#             interpreter.model = "openhermes2.5-mistral"
#             interpreter.conversation_history = False
#             messages = interpreter.chat(f"Run this code and give me the result: {code}")
#             return messages[0]["output"]
#         return code


# @nice_repr
# class CodeGeneratorAlgorithm(DspyAlgorithmBase):
#     def __init__(
#         self,
#         prompt_technique: CategoricalValue(POT_MODULE),
#     ) -> None:
#         self.prompt_technique = prompt_technique

#     @classmethod
#     def get_signature(cls) -> type[AlgorithmSignature]:
#         return CodeGenerator

#     def run(self, instruction):
#         prompt_module = instantiate_prompt_module(
#             self.prompt_technique, self.get_signature()
#         )
#         kwargs = {"instruction": instruction}
#         output_name = [
#             k
#             for k, v in self.get_signature().kwargs.items()
#             if isinstance(v, OutputField)
#         ][0]
#         result = prompt_module(**kwargs)
#         return result.completions[output_name]


@nice_repr
class WikipediaRetrievalAlgorithm(DspyAlgorithmBase):
    def __init__(self) -> None:
        pass

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return WikipediaRetrieval

    def run(self, question):
        retrieval_module = dspy.Retrieve()
        context = retrieval_module(question).passages
        return context


@nice_repr
class GenerateQuestionAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(COT_MODULE),
    ) -> None:
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return GenerateQuestion

    def run(self, text):
        prompt_module = instantiate_prompt_module(
            self.prompt_technique, self.get_signature()
        )
        kwargs = {"text": text}
        output_name = [
            k
            for k, v in self.get_signature().kwargs.items()
            if isinstance(v, OutputField)
        ][0]
        result = prompt_module(**kwargs)
        return result.completions[output_name]
