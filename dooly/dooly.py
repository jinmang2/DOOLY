import re
from typing import Optional

from .tasks import (
    DependencyParsing as DP,
    MachineReadingComprehension as MRC,
    MachineTranslation as MT,
    NamedEntityRecognition as NER,
    NaturalLanguageInference as NLI,
    PosTagging as POS,
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
add_task(["wsd", "word_sense_ambiguation"], WSD)
add_task(["zero_topic", "zt", "zsl", "zero_topic_classification", "zero_shot_classification"], ZT)


def normalize_task(task: str):
    task = re.sub(" +", " ", task.lower())
    task = task.replace("-", "_").replace(" ", "_")
    return task


class Dooly:
    def __new__(
        cls,
        task: str,
        lang: str,
        n_model: Optional[str] = None,
        **kwargs
     ):
        task = normalize_task(task)
        task_cls = DOOLY_HUB_CONTENTS.get(task, None)
        if task_cls is None:
            raise KeyError (
                f"Unavailable task name '{task}'. See here {DOOLY_HUB_CONTENTS.keys()}"
            )
        return task_cls.build(lang, n_model, **kwargs)
