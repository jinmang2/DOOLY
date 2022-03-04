import re
from typing import Optional

from .tasks import (
    NamedEntityRecognition,
    WordSenseDisambiguation,
)


def add_task(names, task):
    DOOLY_HUB_CONTENTS.update(dict.fromkeys(names, task))


DOOLY_HUB_CONTENTS = {}
add_task(["ner", "named_entity_recognition", "entity_recognition"], NamedEntityRecognition)
add_task(["wsd", "word_sense_ambiguation"], WordSenseDisambiguation)


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
