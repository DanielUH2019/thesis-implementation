from typing import Optional
from dspy.signatures.signature import Signature
from dspy.predict import ReAct, ChainOfThought, ProgramOfThought
from .algorithms_base import AlgorithmSignature, DspyAlgorithmBase
from autogoal_core.grammar import (
    CategoricalValue,
    ContinuousValue,
    DiscreteValue,
    BooleanValue,
)
from .signatures import (
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
)

from .hf_tasks import (
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
from .constants import TIMM_MODELS_SUPPORTED, TRANSFORMERS_MODELS_SUPPORTED


from .model_backends import (
    TimmBackend,
    TransformersBackend,
    TransformersBackendConfig,
    TimmBackendConfig,
)


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
        raise ValueError(f"Model {model_name} not supported")


class GenerateAnswerAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(ReAct, ChainOfThought, ProgramOfThought),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return GenerateAnswer

    def run(self, context, question):
        return self.prompt_technique(context, question)


class BasicQAAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(ReAct, ChainOfThought, ProgramOfThought),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return BasicQA

    def run(self, question):
        return self.prompt_technique(question)


class GenerateSearchQueryAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(ReAct, ChainOfThought, ProgramOfThought),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return GenerateSearchQuery

    def run(self, context, question):
        return self.prompt_technique(context, question)


class ImageCaptioningAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        prompt_technique: CategoricalValue(ReAct, ChainOfThought, ProgramOfThought),
    ):
        self.prompt_technique = prompt_technique

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return ImageCaptioning

    def run(self, image):
        return self.prompt_technique(image)


class ImageClassificationAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        models: CategoricalValue(
            "mobilenetv3_large_100",
            "adv_inception_v3",
            "cspdarknet53",
            "cspresnext50",
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
            "densenetblur121d",
            "dla34",
            "dla46_c",
        ),
    ):
        self.model_name = models

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return ImageClassification

    def run(self, image):
        run_algorithm_with_transformers_or_timm(
            self.model_name, IMAGE_CLASSIFICATION, image
        )


class LanguageTranslationAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        models: CategoricalValue(
            "t5-small", "t5-base", "t5-large", "mbart-large-50-many-to-many-mmt"
        ),
    ):
        self.model_name = models

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return LanguageTranslation

    def run(self, text):
        run_algorithm_with_transformers_or_timm(self.model_name, TRANSLATION, text)


class SummarizeTextAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        models: CategoricalValue(
            "facebook/bart-large-cnn",
            "Falconsai/text_summarization",
            "mbart-large-50-many-to-many-mmt",
        ),
    ):
        self.model_name = models

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return SummarizeText

    def run(self, text):
        run_algorithm_with_transformers_or_timm(self.model_name, SUMMARIZATION, text)


class TextClassificationAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        models: CategoricalValue(
            "distilbert-base-uncased",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "distilbert-base-cased",
            "distilbert-base-multilingual-cased",
        ),
    ):
        self.model_name = models

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return TextClassification

    def run(self, text):
        run_algorithm_with_transformers_or_timm(
            self.model_name, TEXT_CLASSIFICATION, text
        )

class ZeroShotTextClassificationAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        models: CategoricalValue(
            "facebook/bart-large-mnli",
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
            "facebook/bart-large-mnli",
            "joeddav/xlm-roberta-large-xnli"
        ),
    ):
        self.model_name = models

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return ZeroShotTextClassification

    def run(self, text):
        run_algorithm_with_transformers_or_timm(
            self.model_name, ZERO_SHOT_CLASSIFICATION, text
        )

class ZeroShotImageClassificationAlgorithm(DspyAlgorithmBase):
    def __init__(
        self,
        models: CategoricalValue(
            "openai/clip-vit-large-patch14",
        ),
    ):
        self.model_name = models

    @classmethod
    def get_signature(cls) -> type[AlgorithmSignature]:
        return ZeroShotImageClassification

    def run(self, image):
        run_algorithm_with_transformers_or_timm(
            self.model_name, ZERO_SHOT_IMAGE_CLASSIFICATION, image
        )
