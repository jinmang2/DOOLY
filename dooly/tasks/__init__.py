from .dependency_parsing import DependencyParsing
from .machine_reading_comprehension import MachineReadingComprehension
from .machine_translation import MachineTranslation, BackTranslationDA
from .named_entity_recognition import NamedEntityRecognition
from .natural_language_inference import NaturalLanguageInference
from .pos_tagging import PosTagging
from .question_generation import QuestionGeneration
from .word_embedding import WordEmbedding
from .word_sense_disambiguation import WordSenseDisambiguation
from .zero_shot_classification import ZeroShotClassification


DoolyTaskHub = {
    "bt": BackTranslationDA,
    "dp": DependencyParsing,
    "mrc": MachineReadingComprehension,
    "mt": MachineTranslation,
    "ner": NamedEntityRecognition,
    "nli": NaturalLanguageInference,
    "pos": PosTagging,
    "qg": QuestionGeneration,
    "word_embedding": WordEmbedding,
    "wsd": WordSenseDisambiguation,
    "zero_topic": ZeroShotClassification,
}
