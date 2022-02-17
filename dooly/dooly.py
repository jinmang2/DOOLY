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


class Dooly:
    def __new__(
        cls,
        task: str,
        lang: str,
        n_model: Optional[str] = None,
         **kwargs
     ):
        hub_interface = DOOLY_HUB_CONTENTS.get(task, None)
        if hub_interface is None:
            raise KeyError (
                f"Unavailable task name. See here {DOOLY_HUB_CONTENTS.keys()}"
            )
        return hub_interface.build(lang, n_model)
