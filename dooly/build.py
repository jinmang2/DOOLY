import abc
import json
import pickle
import inspect
import warnings
from packaging import version
from functools import partial
from contextlib import contextmanager
from typing import Dict, Union, Optional, Tuple, List, TypeVar

import torch

from transformers.file_utils import hf_bucket_url, cached_path
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


Tokenizer = TypeVar("Tokenizer") # ../tokenizers/base.py


HUB_NAME = "jinmang2/dooly-hub"
VOCAB_NAME = "vocab.json"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

TOKENIZER_USER_AGENT = {"file_type": "tokenizer",
                        "from_auto_class": False,
                        "is_fast": False,}

CONFIG_USER_AGENT = {"file_type": "config",
                     "from_auto_class": False,}

MODEL_USER_AGENT = {"file_type": "model",
                    "framework": "pytorch",
                    "from_auto_class": False,}

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


def download_from_hf_hub(
    model_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    resume_download: bool = False,
    user_agent: Optional[Union[Dict, str]] = None,
) -> Optional[str]:
    # Resolve a model identifier, a file name, and an optional revision id,
    # to a huggingface.co-hosted url, redirecting to Cloudfront
    # (a Content Delivery Network, or CDN) for large files.
    huggingface_co_resolved_file = hf_bucket_url(
        model_id=model_id,
        filename=filename,
        subfolder=subfolder,
        revision=revision,
    )
    # Given something that might be a URL (or might be a local path),
    # determine which. If it's a URL, download the file and cache it,
    # and return the path to the cached file. If it's already a local path,
    # make sure the file exists and then return the path
    # Do not extract files (extract_compressed_file and force_extract is False)
    resolved_file_path = cached_path(
        huggingface_co_resolved_file,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        user_agent=user_agent,
    )

    return resolved_file_path


class BuildMixin:

    @classmethod
    @abc.abstractmethod
    def build(
        cls,
        lang: Optional[str] = None,
        n_model: Optional[str] = None,
        **kwargs
    ):
        pass

    @classmethod
    def _check_available_lang(cls, lang: str):
        assert lang in cls.available_langs, (
            f"In {cls.__name__} task, the available languages are {cls.available_langs}. "
            f"But, {lang} was entered as input."
        )

    @classmethod
    def _check_available_model(cls, lang: str, n_model: str):
        assert n_model in cls.available_models[lang], (
            f"In {cls.__name__} task, the available models in the input langauge "
            f"are {cls.available_models[lang]}. But, {n_model} was entered as input."
        )

    @classmethod
    def _check_validate_input(cls, lang: str, n_model: str) -> Tuple[str, str]:
        if lang is None:
            lang = cls.available_langs[0]
        cls._check_available_lang(lang)
        if n_model is None:
            n_model = cls.available_models[lang][0]
        cls._check_available_model(lang, n_model)
        return lang, n_model

    @classmethod
    def _parse_build_kwargs(cls, tokenizer_class, **kwargs) -> Tuple[Dict]:
        # parse download keyword arguments
        download_kwargs = {}
        download_kwargs["revision"] = kwargs.pop("revision", None)
        download_kwargs["cache_dir"] = kwargs.pop("cache_dir", None)
        download_kwargs["force_download"] = kwargs.pop("force_download", False)
        download_kwargs["resume_download"] = kwargs.pop("resume_download", False)
        # parse tokenizer keyword arguments
        tokenizer_kwargs= {}
        tokenizer_params = inspect.signature(tokenizer_class.__init__).parameters.values()
        for param in tokenizer_params:
            if (param.name in ["self", "lang", "vocab"] or
                param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]):
                continue
            default = param.default
            if param.empty == default:
                warnings.warn(
                    "When implementing a class, be sure to write type hint and default value. "
                    f"Warning message depart from {param}"
                )
                default = None
            tokenizer_kwargs[param.name] = kwargs.pop(param.name, param.default)
        model_kwargs = kwargs
        return download_kwargs, tokenizer_kwargs, model_kwargs

    @classmethod
    def _build_tokenizer(
        cls,
        task: str,
        lang: str,
        n_model: str,
        tokenizer_class: Union[Tokenizer, PreTrainedTokenizerBase],
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = False,
        **kwargs,
    ) -> Union[Tokenizer, PreTrainedTokenizerBase]:

        if PreTrainedTokenizerBase in cls.__mro__:
            return tokenizer_class.from_pretrained(
                pretrained_model_name_or_path=HUB_NAME,
                subfolder=f"{task}/{lang}/{n_model}",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                **kwargs,
            )

        _download_from_hf_hub = partial(
            download_from_hf_hub,
            model_id=HUB_NAME,
            subfolder=f"{task}/{lang}/{n_model}",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            user_agent=TOKENIZER_USER_AGENT,
        )

        # Load from URL or cache if already cached
        resolved_vocab_file = _download_from_hf_hub(filename=VOCAB_NAME)

        # _dict_from_json_file
        with open(resolved_vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        tokenizer = tokenizer_class(lang, vocab, **kwargs)

        if "bpe" in tokenizer_class.__name__.lower():
            encoder = None
            bpe_merges = None

            if lang == "en":
                encoder_json = _download_from_hf_hub(filename="encoder.json")
                with open(encoder_json, "r") as f:
                    encoder = json.load(f)

                vocab_bpe = _download_from_hf_hub(filename="vocab.bpe")
                with open(vocab_bpe, "r", encoding="utf-8") as f:
                    bpe_data = f.read()
                bpe_merges = [tuple(merge_str.split())
                              for merge_str in bpe_data.split("\n")[1:-1]]

            tokenizer._build_bpe(lang, encoder, bpe_merges)

        return tokenizer

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

    @classmethod
    def _build_misc(
        cls,
        lang: str,
        n_model: str,
        filenames: List[str],
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = False,
    ) -> Tuple:
        misc = ()
        for filename in filenames:
            resolved_file_path = download_from_hf_hub(
                model_id=HUB_NAME,
                filename=filename,
                subfolder=f"{cls.task}/{lang}/{n_model}",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
            )

            # TODO: pickle, json 등 파일 유형별 읽는 코드 작성
            if filename.endswith(".pkl"):
                misc += (pickle.load(open(resolved_file_path, "rb", encoding="utf-8")),)
            elif filename.endswith(".items"):
                misc += (open(resolved_file_path, "r", encoding="utf-8").readlines(),)
            else:
                continue

        if len(misc) == 1:
            misc = misc[0]

        return misc
