import re
from typing import Optional

from .tasks import DoolyTaskHub


def get_normalize_task_name(task: str):
    task = re.sub(" +", " ", task.lower())
    task = task.replace("-", "_").replace(" ", "_")
    return task


def add_task(names, task):
    TASK_ALIASES.update(dict.fromkeys(names, task))


TASK_ALIASES = {}
add_task(["bt", "back_translation", "bt_aug", "bt_da", "back_translation_aug"], "bt")
add_task(["dp", "dep_parse", "dependency_parsing"], "dp")
add_task(["mrc", "machine_reading_comprehension", "reading_comprehension"], "mrc")
add_task(["mt", "nmt", "translation", "machine_translation"], "mt")
add_task(["ner", "named_entity_recognition", "entity_recognition"], "ner")
add_task(["nli", "natural_language_inference"], "nli")
add_task(["pos", "pos_tag", "pos_tagging"], "pos")
add_task(["qg", "question_generation"], "qg")
add_task(["word_embedding"], "word_embedding")
add_task(["wsd", "word_sense_ambiguation"], "wsd")
add_task(["zero_topic", "zt", "zsl", "zero_topic_classification"], "zero_topic")


def add_lang(names, lang):
    LANG_ALIASES.update(dict.fromkeys(names, lang))


LANG_ALIASES = {}
add_lang(["ko", "korean", "kor", "kr"], "ko")
add_lang(["en", "english", "eng"], "en")
add_lang(["ja", "japanese", "jap", "jp"], "ja")
add_lang(["zh", "chinese", "chn", "cn"], "zh")
add_lang(["je", "jejueo", "jje"], "je")
add_lang(["multi", "multilingual"], "multi")


class Dooly:
    def __new__(
        cls,
        task: str,
        lang: Optional[str] = None,
        n_model: Optional[str] = None,
        **kwargs,
     ):
        task = normalize_task(task)
        task = TASK_ALIASES.get(task, None)
        if task is None:
            raise KeyError(
                f"Unavailable task name '{task}'. See here {TASK_ALIASES.keys()}"
            )
        task_cls = DoolyTaskHub[task]
        if lang is not None:
            lang = LANG_ALIASES.get(lang.lower(), None)
        return task_cls.build(lang, n_model, **kwargs)

    @staticmethod
    def available_tasks() -> str:
        return f"Available tasks are {list(TASK_ALIASES.keys())}."

    @staticmethod
    def available_models(task: str) -> str:
        if task not in TASK_ALIASES:
            raise KeyError(
                f"Unknown task {task}. Please check available models via `available_tasks()`."
            )
        task_cls = DoolyTaskHub[TASK_ALIASES[normalize_task(task)]]

        output = f"Available models for `{task}` are "
        for lang, models in task_cls.available_models.items():
            output += f"([lang]: {lang}, [model]: {', '.join(models)}), "
        return output[:-2]
