from typing import Type

from transformers import PreTrainedTokenizer

from .base import Tokenizer
from .bpe import Gpt2BpeTokenizer, BpeJaZhTokenizer
from .char import CharS1Tokenizer, CharS2Tokenizer
from .hf_tokenizer import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
)
from .pos_tagger import (
    PosDpTokenizer,
)
from ..utils import DOOLY_HUB_NAME


DoolyTokenizerHub = {
    "dp": {
        "ko": {"posbert.base": PosDpTokenizer},
    },
    "mrc": {
        "ko": {"brainbert.base": RobertaTokenizerFast},
    },
    "mt": {
        "multi": {
            "transformer.large.mtpg": CharS2Tokenizer,
            "transformer.large.fast.mtpg": CharS2Tokenizer,
        },
    },
    "ner": {
        "ko": {"charbert.base": CharS1Tokenizer},
        "en": {"roberta.base": Gpt2BpeTokenizer},
        "ja": {"jaberta.base": BpeJaZhTokenizer},
        "zh": {"zhberta.base": BpeJaZhTokenizer},
    },
    "nli": {
        "ko": {"brainbert.base": RobertaTokenizerFast},
        "en": {"roberta.base": Gpt2BpeTokenizer},
        "ja": {"jaberta.base": BpeJaZhTokenizer},
        "zh": {"zhberta.base": BpeJaZhTokenizer},
    },
    "qg": {
        "ko": {"kobart.base": PreTrainedTokenizerFast},
    },
    "wsd": {
        "ko": {"transformer.large": CharS2Tokenizer},
    },
}
DoolyTokenizerHub["bt"] = DoolyTokenizerHub["mt"]
DoolyTokenizerHub["zero_topic"] = DoolyTokenizerHub["nli"]

available_tasks = list(DoolyTokenizerHub.keys())


def load_pretrained_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_class: Type[PreTrainedTokenizer],
    **kwargs,
) -> PreTrainedTokenizer:
    return tokenizer_class.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )


def load_tokenizer_from_dooly_hub(
    subfolder: str,
    tokenizer_class: Type[PreTrainedTokenizer],
    **kwargs,
) -> PreTrainedTokenizer:

    def _load_pretrained(
        pretrained_model_name_or_path: str, subfolder: str, **kwargs
    ):
        return tokenizer_class.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, **kwargs
        )

    return _load_pretrained(
        pretrained_model_name_or_path=DOOLY_HUB_NAME,
        subfolder=subfolder, **kwargs
    )


def load_dooly_tokenizer(
    pretrained_model_name_or_path: str = None,
    tokenizer_class: Type[PreTrainedTokenizer] = None,
    task: str = None,
    lang: str = None,
    n_model: str = None,
    **kwargs,
) -> PreTrainedTokenizer:
    if pretrained_model_name_or_path is not None:
        if tokenizer_class is None:
            raise ValueError(
                "If you are using the personal huggingface.co model, "
                "`tokenizer_class` parameter is required."
            )
        return load_pretrained_tokenizer(
            pretrained_model_name_or_path, tokenizer_class
        )

    if all([task is None and lang is None and n_model is None]):
        raise ValueError(
            "`task`, `lang`, and `n_model` parameters are required to "
            "access the subfolder of dooly-hub.\nCheck your parameters! "
            f"`task`: {task} `lang`: {lang} `n_model`: {n_model}."
        )

    assert (
        task in available_tasks
    ), f"Task `{task}` is not available. See here {available_tasks}."

    available_langs = DoolyTokenizerHub[task]
    assert lang in available_langs, (
        f"Language `{lang}` is not available in this task {task}. "
        f"See here {available_langs}."
    )

    available_models = available_langs[lang]
    assert n_model in available_models, (
        f"Model `{n_model}` is not available in this task-lang pair. "
        f"See here {available_models}."
    )

    subfolder = f"{task}/{lang}/{n_model}"
    subfolder_postfix = kwargs.pop("subfolder_postfix", None)
    if subfolder_postfix is not None:
        subfolder += f"/{subfolder_postfix}"

    if tokenizer_class is None:
        tokenizer_class = available_models[n_model]

    kwargs.update({"task": task, "lang": lang, "n_model": n_model})

    return load_tokenizer_from_dooly_hub(
        subfolder=subfolder, tokenizer_class=tokenizer_class, **kwargs
    )