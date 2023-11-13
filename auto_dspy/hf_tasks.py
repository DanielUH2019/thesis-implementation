# Define the tasks that the models will solve using HuggingFaces tasks

from enum import Enum
from pydantic import BaseModel


class GeneralTasks(Enum):
    """General tasks that can be solved by models"""

    COMPUTER_VISION = "Computer Vision"
    NLP = "Natural Language Processing"
    AUDIO = "Audio"
    TABULAR = "Tabular"


class Task(BaseModel):
    high_level_task_name: GeneralTasks
    task_name: str
    short_downstream_task_description: str


DEPTH_ESTIMATION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="depth-estimation",
    short_downstream_task_description="Depth estimation is the task of predicting depth of the objects present in an image.",
)

IMAGE_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="image-classification",
    short_downstream_task_description="Image classification is the task of assigning a label or class to an entire image.",
)

IMAGE_SEGMENTATION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="image-segmentation",
    short_downstream_task_description="Image Segmentation divides an image into segments where each pixel in the image is mapped to an object. ",
)

IMAGE_TO_IMAGE = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="image-to-image",
    short_downstream_task_description="Image-to-image is the task of transforming a source image to match the characteristics of a target image or a target image domain.",
)

OBJECT_DETECTION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="object-detection",
    short_downstream_task_description="Object Detection models allow users to identify objects of certain defined classes.",
)

VIDEO_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="video-classification",
    short_downstream_task_description="Video classification is the task of assigning a label or class to an entire video.",
)

UNCONDITIONAL_IMAGE_GENERATION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="unconditional-image-generation",
    short_downstream_task_description="Unconditional image generation is the task of generating images with no condition in any context (like a prompt text or another image).",
)

ZERO_SHOT_IMAGE_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.COMPUTER_VISION,
    task_name="zero-shot-image-classification",
    short_downstream_task_description="Zero shot image classification is the task of classifying previously unseen classes during training of a model.",
)

CONVERSATIONAL = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="conversational",
    short_downstream_task_description="Conversational response modelling is the task of generating conversational text that is relevant, coherent and knowledgable given a prompt.",
)

FILL_MASK = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="fill-mask",
    short_downstream_task_description="Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks.",
)

QUESTION_ANSWERING = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="question-answering",
    short_downstream_task_description="Question Answering models can retrieve the answer to a question from a given text, which is useful for searching for an answer in a document.",
)

SENTENCE_SIMILARITY = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="sentence-similarity",
    short_downstream_task_description="Sentence Similarity is the task of determining how similar two texts are.",
)

SUMMARIZATION = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="summarization",
    short_downstream_task_description="Summarization is the task of producing a shorter version of a document while preserving its important information.",
)

TABLE_QUESTION_ANSWERING = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="table-question-answering",
    short_downstream_task_description="Table Question Answering (Table QA) is the answering a question about an information on a given table.",
)

TEXT_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="text-classification",
    short_downstream_task_description="Text Classification is the task of assigning a label or class to a given text.",
)

TEXT_GENERATION = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="text-generation",
    short_downstream_task_description="Generating text is the task of producing new text.",
)

TOKEN_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="token-classification",
    short_downstream_task_description="Token classification is a natural language understanding task in which a label is assigned to some tokens in a text.",
)

TRANSLATION = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="translation",
    short_downstream_task_description="Translation is the task of converting text from one language to another.",
)

ZERO_SHOT_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.NLP,
    task_name="zero-shot-classification",
    short_downstream_task_description="Zero-shot text classification is a task in natural language processing where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes.",
)

AUDIO_CLASSIFICATION = Task(
    high_level_task_name=GeneralTasks.AUDIO,
    task_name="audio-classification",
    short_downstream_task_description="Audio classification is the task of assigning a label or class to a given audio.",
)

AUDIO_TO_AUDIO = Task(
    high_level_task_name=GeneralTasks.AUDIO,
    task_name="audio-to-audio",
    short_downstream_task_description="Audio-to-Audio is a family of tasks in which the input is an audio and the output is one or multiple generated audios.",
)

AUTOMATIC_SPEECH_RECOGNITION = Task(
    high_level_task_name=GeneralTasks.AUDIO,
    task_name="automatic-speech-recognition",
    short_downstream_task_description="Automatic Speech Recognition (ASR), also known as Speech to Text (STT), is the task of transcribing a given audio to text.",
)

TEXT_TO_SPEECH = Task(
    high_level_task_name=GeneralTasks.AUDIO,
    task_name="text-to-speech",
    short_downstream_task_description="Text-to-Speech (TTS) is the task of generating natural sounding speech given text input. ",
)


# class Tabular(Task):
#     short_task_description = "Tabular"


# class TabularClassification(Tabular):
#     short_downstream_task_description = "Tabular classification is the task of classifying a target category (a group) based on set of attributes."


# class TabularRegression(Tabular):
#     short_downstream_task_description = "Tabular regression is the task of predicting a numerical value given a set of attributes."
