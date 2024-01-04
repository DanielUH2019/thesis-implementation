from algorithms_base import AlgorithmSignature
from dspy.signatures import InputField, OutputField


class TeleprompterSignature(AlgorithmSignature):
    """Powerful optimizer that can take any program and learn to bootstrap and select effective prompts for its modules"""

    dspy_module_generator = InputField()
    trainset = InputField()
    compiled_program = OutputField()


class GenerateQuestion(AlgorithmSignature):
    """Generate a question from a text that can be used to better understand the text."""

    text = InputField(desc="the text to be understood")
    question = OutputField(
        desc="a question that can be asked to better understand a text"
    )

    @classmethod
    def inputs_fields_to_output(cls) -> dict[str, InputField]:
        return {"instruction": InputField(desc="the instruction to be understood")}


class GenerateSearchQuery(AlgorithmSignature):
    """Write and run simple search query that will help answer a complex question."""

    question = InputField(desc="question")
    answer = OutputField(desc="may contain relevant facts")


class GenerateAnswer(AlgorithmSignature):
    """Answer questions with short factoid answers using some relevant facts."""

    context = InputField(desc="may contain relevant facts")
    question = InputField()
    answer = OutputField(desc="often between 1 and 5 words")


class BasicQA(AlgorithmSignature):
    """Answer questions with short factoid answers."""

    question = InputField()
    answer = OutputField(desc="often between 1 and 5 words")


class SummarizeText(AlgorithmSignature):
    """Generate a concise summary of a given text."""

    text = InputField(desc="the input text to be summarized")
    summary = OutputField(desc="a concise summary of the input text")


class ImageCaptioning(AlgorithmSignature):
    """Generate a descriptive caption for an input image."""

    image = InputField(desc="the input image to be captioned")
    caption = OutputField(desc="a descriptive caption for the input image")


class LanguageTranslation(AlgorithmSignature):
    """Translate text from one language to another."""

    source_text = InputField(desc="the text to be translated")
    target_language = InputField(desc="the language to translate to")
    translated_text = OutputField(desc="the translated text")


class ZeroShotTextClassification(AlgorithmSignature):
    """Analyze a given piece of text and classify into some candidate labels"""

    text = InputField(desc="the text for sentiment analysis")
    candidate_labels = InputField(desc="labels to choose for the text classification")
    predicted_classification = OutputField(
        desc="the classification of the input text, e.g., positive, negative, neutral"
    )


class TextClassification(AlgorithmSignature):
    """Analyze the sentiment of a given piece of text and classify it"""

    text = InputField(desc="text for sentiment analysis")
    predicted_classification = OutputField(
        desc="the classification of the input text, e.g., positive, negative, neutral"
    )


class ImageClassification(AlgorithmSignature):
    """Assign a label to an entire image"""

    image = InputField()
    label = OutputField()


class ZeroShotImageClassification(AlgorithmSignature):
    """Classify an images into one of several classes, without any prior training or knowledge of the classes."""

    image = InputField()
    label = OutputField()


class CodeGenerator(AlgorithmSignature):
    """Generate code that solves a problem or answer a question"""

    instruction = InputField(desc="problem or question that can be solve through programming")
    answer = OutputField()


class WikipediaRetrieval(AlgorithmSignature):
    """Retrieve information from wikipedia to provide useful facts"""

    question = InputField(desc="question")
    context = OutputField(desc="may contain relevant facts")
