import re
from typing import Optional

from .tasks import (
    NamedEntityRecognition as NER,
    NaturalLanguageInference as NLI,
    WordSenseDisambiguation as WSD,
    ZeroShotClassification as ZT,
)


DOOLY_HUB_CONTENTS = {}

def add_task(names, task):
    DOOLY_HUB_CONTENTS.update(dict.fromkeys(names, task))

add_task(["ner", "named_entity_recognition", "entity_recognition"], NER)
add_task(["nli", "natural_language_inference"], NLI)
add_task(["wsd", "word_sense_ambiguation"], WSD)
add_task(["zero_topic", "zt", "zsl", "zero_topic_classification", "zero_shot_classification"], ZT)


def normalize_task(task: str):
    return re.sub(" +", " ", task.lower()).replace(" ", "_")


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
