from typing import Type, Union

import transformers
from ..utils import _locate, DOOLY_HUB_NAME


DoolyTokenizerHub = {
    "dp": {"ko": {"posbert.base": "pos_tagger.DoolyPosDpTokenizer"}},
    "mrc": {"ko": {"brainbert.base": "fast.RobertaTokenizerFast"}},
    "mt": {
        "multi": {
            "transformer.large.mtpg": "char.DoolyCharSeq2SeqNmtTokenizer",
            "transformer.large.fast.mtpg": "char.DoolyCharSeq2SeqNmtTokenizer",
        },
    },
    "ner": {
        "ko": {"charbert.base": "char.DoolyCharBertTokenizer"},
        "en": {"roberta.base": "bpe.DoolyGPT2TokenizerFast"},
        "ja": {"jaberta.base": "bpe.DoolyBertJaTokenizer"},
        "zh": {"zhberta.base": "bpe.DoolyBertZhTokenizer"},
    },
    "nli": {
        "ko": {"brainbert.base": "fast.RobertaTokenizerFast"},
        "en": {"roberta.base": "bpe.DoolyGPT2TokenizerFast"},
        "ja": {"jaberta.base": "bpe.DoolyBertJaTokenizer"},
        "zh": {"zhberta.base": "bpe.DoolyBertZhTokenizer"},
    },
    "qg": {"ko": {"kobart.base": "bpe.PreTrainedTokenizerFast"}},
    "wsd": {"ko": {"transformer.large": "char.DoolyCharSeq2SeqWsdTokenizer"}},
}
DoolyTokenizerHub["bt"] = DoolyTokenizerHub["mt"]
DoolyTokenizerHub["zero_topic"] = DoolyTokenizerHub["nli"]

available_tasks = list(DoolyTokenizerHub.keys())


def load_pretrained_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_class: Type[transformers.PreTrainedTokenizer],
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, **kwargs)


def load_tokenizer_from_dooly_hub(
    subfolder: str, tokenizer_class: Type[transformers.PreTrainedTokenizer], **kwargs
) -> transformers.PreTrainedTokenizer:
    def _load_pretrained(pretrained_model_name_or_path: str, subfolder: str, **kwargs):
        return tokenizer_class.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, **kwargs
        )

    return _load_pretrained(
        pretrained_model_name_or_path=DOOLY_HUB_NAME, subfolder=subfolder, **kwargs
    )


def load_dooly_tokenizer(
    pretrained_model_name_or_path: str = None,
    tokenizer_class: Union[str, Type[transformers.PreTrainedTokenizer]] = None,
    task: str = None,
    lang: str = None,
    n_model: str = None,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    if pretrained_model_name_or_path is not None:
        if tokenizer_class is None:
            raise ValueError(
                "If you are using the personal huggingface.co model, "
                "`tokenizer_class` parameter is required."
            )
        return load_pretrained_tokenizer(
            pretrained_model_name_or_path, tokenizer_class, **kwargs
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
        module_path = "dooly.tokenizers." + available_models[n_model]
    elif isinstance(tokenizer_class, str):
        module_path = tokenizer_class

    if not issubclass(tokenizer_class, transformers.PreTrainedTokenizer):
        tokenizer_class = _locate(module_path)

    kwargs.update({"task": task, "lang": lang, "n_model": n_model})

    return load_tokenizer_from_dooly_hub(
        subfolder=subfolder, tokenizer_class=tokenizer_class, **kwargs
    )
