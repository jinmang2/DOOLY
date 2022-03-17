import re
from typing import Optional

from .tasks import (
    DependencyParsing as DP,
    MachineReadingComprehension as MRC,
    MachineTranslation as MT,
    NamedEntityRecognition as NER,
    NaturalLanguageInference as NLI,
    PosTagging as POS,
    QuestionGeneration as QG,
    WordEmbedding,
    WordSenseDisambiguation as WSD,
    ZeroShotClassification as ZT,
)


def add_task(names, task):
    DOOLY_HUB_CONTENTS.update(dict.fromkeys(names, task))


DOOLY_HUB_CONTENTS = {}
add_task(["dp", "dep_parse", "dependency_parsing"], DP)
add_task(["mrc", "machine_reading_comprehension", "reading_comprehension"], MRC)
add_task(["mt", "nmt", "translation", "machine_translation", "neural_machine_translation"], MT)
add_task(["ner", "named_entity_recognition", "entity_recognition"], NER)
add_task(["nli", "natural_language_inference"], NLI)
add_task(["pos", "pos_tag", "pos_tagging"], POS)
add_task(["qg", "question_generation"], QG)
add_task(["word_embedding"], WordEmbedding)
add_task(["wsd", "word_sense_ambiguation"], WSD)
add_task(["zero_topic", "zt", "zsl", "zero_topic_classification", "zero_shot_classification"], ZT)


def normalize_task(task: str):
    task = re.sub(" +", " ", task.lower())
    task = task.replace("-", "_").replace(" ", "_")
    return task


LANG_ALIASES = {}
LANG_ALIASES.update(dict.fromkeys(["ko", "korean", "kor", "kr"], "ko"))
LANG_ALIASES.update(dict.fromkeys(["en", "english", "eng"], "en"))
LANG_ALIASES.update(dict.fromkeys(["ja", "japanese", "jap", "jp"], "ko"))
LANG_ALIASES.update(dict.fromkeys(["zh", "chinese", "chn", "cn"], "ko"))
LANG_ALIASES.update(dict.fromkeys(["je", "jejueo", "jje"], "je"))
LANG_ALIASES.update(dict.fromkeys(["multi", "multilingual"], "multi"))


class Dooly:
    def __new__(
        cls,
        task: str,
        lang: Optional[str] = None,
        n_model: Optional[str] = None,
        **kwargs
     ):
        task = normalize_task(task)
        task_cls = DOOLY_HUB_CONTENTS.get(task, None)
        if task_cls is None:
            raise KeyError (
                f"Unavailable task name '{task}'. See here {DOOLY_HUB_CONTENTS.keys()}"
            )
        if lang is not None:
            lang = LANG_ALIASES.get(lang.lower(), None)
        return task_cls.build(lang, n_model, **kwargs)

    @staticmethod
    def available_tasks() -> str:
        return f"Available tasks are {list(DOOLY_HUB_CONTENTS.keys())}."

    @staticmethod
    def available_models(task: str) -> str:
        if task not in DOOLY_HUB_CONTENTS:
            raise KeyError(
                f"Unknown task {task}. Please check available models via `available_tasks()`."
            )

        output = f"Available models for {task} are "
        for lang, models in DOOLY_HUB_CONTENTS[task].available_models.items():
            output += f"([lang]: {lang}, [model]: {', '.join(models)}), "
        return output[:-2]
