import json
from packaging import version
from contextlib import contextmanager
from typing import Dict, Union, Optional, Tuple, List, TypeVar

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .modeling_bart import BartForConditionalGeneration
from .modeling_fsmt import FSMTConfig, FSMTForConditionalGeneration
from .modeling_roberta import (
    RobertaConfig,
    RobertaForDependencyParsing,
    RobertaForSpanPrediction,
    RobertaForSequenceTagging,
    RobertaForSequenceClassification,
)
from ..build_utils import (
    download_from_hf_hub,
    CONFIG_USER_AGENT,
    HUB_NAME,
    MODEL_USER_AGENT,
    CONFIG_NAME,
    WEIGHTS_NAME
)


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
        "ko": {"transformer.large": FSMTForConditionalGeneration}
    },
}
DoolyModelHub["bt"] = DoolyModelHub["mt"]
DoolyModelHub["zero_topic"] = DoolyModelHub["nli"]
available_tasks = list(DoolyModelHub.keys())

_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = True


class DoolyModel:

    @classmethod
    def build_model(cls, task: str, lang: str, n_model: str, **kwargs):
        assert task in available_tasks, (
            f"Task `{task}` is not available. See here {available_tasks}."
        )
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

        model_class = available_models[n_model]

        return cls._build_model(task, lang, n_model, model_class, **kwargs)

    @classmethod
    def _build_model_config(
        cls,
        task: str,
        lang: str,
        n_model: str,
        config_class: PretrainedConfig,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = False,
        **kwargs,
    ) -> PretrainedConfig:
        # Load from URL or cache if already cached
        resolved_config_file = download_from_hf_hub(
            model_id=HUB_NAME,
            filename=CONFIG_NAME,
            subfolder=f"{task}/{lang}/{n_model}",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            user_agent=CONFIG_USER_AGENT,
        )

        # _dict_from_json_file
        with open(resolved_config_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)

        return config_class.from_dict(config_dict, **kwargs)

    @classmethod
    def _build_model(
        cls,
        task: str,
        lang: str,
        n_model: str,
        model_class: PreTrainedModel,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = False,
        low_cpu_mem_usage: bool = False,
        _fast_init: bool = True,
        **kwargs,
    ) -> PreTrainedModel:
        if low_cpu_mem_usage:
            assert version.parse(torch.__version__) > version.parse("1.9"), (
                "torch>=1.9 is required for a normal functioning of this module"
                f"using the low_cpu_mem_usage=={low_cpu_mem_usage}, "
                f"but found torch=={torch.__version__}"
            )

        config_class: PretrainedConfig = model_class.config_class

        config = cls._build_model_config(
            task=task,
            lang=lang,
            n_model=n_model,
            config_class=config_class,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            **kwargs,
        )

        # Load from URL or cache if already cached
        resolved_archive_file = download_from_hf_hub(
            model_id=HUB_NAME,
            filename=WEIGHTS_NAME,
            subfolder=f"{task}/{lang}/{n_model}",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            user_agent=MODEL_USER_AGENT,
        )
        state_dict = torch.load(resolved_archive_file, map_location="cpu")

        if low_cpu_mem_usage:
            loaded_state_dict_keys = [k for k in state_dict.keys()]
            del state_dict # free CPU memory - will reload again later

        with no_init_weights(_enable=_fast_init):
            model = model_class(config, **kwargs)

        if low_cpu_mem_usage:
            model_class._load_state_dict_into_model_low_mem(
                model, loaded_state_dict_keys, resolved_archive_file
            )
        else:
            model, _, _, _, _ = model_class._load_state_dict_into_model(
                model,
                state_dict,
                HUB_NAME,
                ignore_mismatched_sizes=False,
                _fast_init=_fast_init,
            )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model
