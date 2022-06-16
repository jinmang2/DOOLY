from typing import Type
from transformers.modeling_utils import PreTrainedModel

from .modeling_bart import BartForConditionalGeneration
from .modeling_fsmt import FSMTForConditionalGeneration
from .modeling_roberta import (
    RobertaForDependencyParsing,
    RobertaForSpanPrediction,
    RobertaForSequenceTagging,
    RobertaForSequenceClassification,
)
from ..utils import register_subfolder, DOOLY_HUB_NAME


DoolyModelHub = {
    "dp": {
        "ko": {"posbert.base": RobertaForDependencyParsing},
    },
    "mrc": {
        "ko": {"brainbert.base": RobertaForSpanPrediction},
    },
    "mt": {
        "multi": {
            "transformer.large.mtpg": FSMTForConditionalGeneration,
            "transformer.large.fast.mtpg": FSMTForConditionalGeneration,
        },
    },
    "ner": {
        "ko": {"charbert.base": RobertaForSequenceTagging},
        "en": {"roberta.base": RobertaForSequenceTagging},
        "ja": {"jaberta.base": RobertaForSequenceTagging},
        "zh": {"zhberta.base": RobertaForSequenceTagging},
    },
    "nli": {
        "ko": {"brainbert.base": RobertaForSequenceClassification},
        "en": {"roberta.base": RobertaForSequenceClassification},
        "ja": {"jaberta.base": RobertaForSequenceClassification},
        "zh": {"zhberta.base": RobertaForSequenceClassification},
    },
    "qg": {
        "ko": {"kobart.base": BartForConditionalGeneration},
    },
    "wsd": {
        "ko": {"transformer.large": FSMTForConditionalGeneration},
    },
}
DoolyModelHub["bt"] = DoolyModelHub["mt"]
DoolyModelHub["zero_topic"] = DoolyModelHub["nli"]

available_tasks = list(DoolyModelHub.keys())


def load_pretrained_model(
    pretrained_model_name_or_path: str,
    model_class: Type[PreTrainedModel],
    **kwargs,
) -> PreTrainedModel:
    return model_class.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )


def load_model_from_dooly_hub(
    subfolder: str,
    model_class: Type[PreTrainedModel],
    **kwargs,
) -> PreTrainedModel:

    @register_subfolder
    def _load_pretrained(
        pretrained_model_name_or_path: str, subfolder: str, **kwargs
    ) -> PreTrainedModel:
        return model_class.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

    return _load_pretrained(
        pretrained_model_name_or_path=DOOLY_HUB_NAME,
        subfolder=subfolder, **kwargs
    )


def load_dooly_model(
    pretrained_model_name_or_path: str = None,
    model_class: Type[PreTrainedModel] = None,
    task: str = None,
    lang: str = None,
    n_model: str = None,
    **kwargs,
) -> PreTrainedModel:
    if pretrained_model_name_or_path is not None:
        if model_class is None:
            raise ValueError(
                "If you are using the personal huggingface.co model, "
                "`model_class` parameter is required."
            )
        return load_pretrained_model(
            pretrained_model_name_or_path, model_class
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

    available_langs = DoolyModelHub[task]
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

    if model_class is None:
        model_class = available_models[n_model]

    return load_model_from_dooly_hub(
        subfolder=subfolder, model_class=model_class, **kwargs
    )