import pickle
from typing import Dict, Union, Optional, Tuple, List, TypeVar, Any

import torch

from ...build_utils import download_from_hf_hub, HUB_NAME
from ...tokenizers import Tokenizer, DoolyTokenizer
from ...models import DoolyModel


class DoolyTaskBase:

    def __init__(self, lang, n_model, device):
        self.lang = lang
        self.n_model = n_model
        self.device = device

    def __repr__(self):
        task_info = f"[TASK]: {self.__class__.__name__}"
        # -1 is object, -2 is DoolyTaskBase.
        category = f"[CATEGORY]: {self.__class__.__mro__[-3].__name__}"
        lang_info = f"[LANG]: {self.lang}"
        model_info = f"[MODEL]: {self.n_model}"
        return "\n".join([task_info, category, lang_info, model_info])

    def _prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """ Prepare input to be placed on the same device in inference. """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._model.device)
        return inputs

    def set_device(self, device: str = None):
        if device is not None:
            self.device = device
        self._model.to(device)

    @classmethod
    def build(
        cls,
        lang: str = None,
        n_model: str = None,
        tok_kwargs: Dict = {},
        model_kwargs: Dict = {},
        **kwargs
    ):
        lang, n_model = cls._check_validate_input(lang, n_model)

        # parse device option from kwargs
        device = kwargs.pop("device", "cpu")

        # parse download keyword arguments
        dl_kwargs = {}
        dl_kwargs["revision"] = kwargs.pop("revision", None)
        dl_kwargs["cache_dir"] = kwargs.pop("cache_dir", None)
        dl_kwargs["force_download"] = kwargs.pop("force_download", False)
        dl_kwargs["resume_download"] = kwargs.pop("resume_download", False)

        # set tokenizer
        tokenizer = cls.build_tokenizer(cls.task, lang, n_model, **dl_kwargs, **tok_kwargs)
        # set model
        model = cls.build_model(cls.task, lang, n_model, **dl_kwargs, **model_kwargs)
        # set misc
        misc_files = cls.misc_files.get(lang, [])
        misc = cls.build_misc(lang, n_model, misc_files, **dl_kwargs)

        return cls(lang, n_model, device, tokenizer, model, misc)

    @staticmethod
    def build_tokenizer(task, lang, n_model, **kwargs):
        return DoolyTokenizer.build_tokenizer(task, lang, n_model, **kwargs)

    @staticmethod
    def build_model(task, lang, n_model, **kwargs):
        return DoolyModel.build_model(task, lang, n_model, **kwargs)

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
    def build_misc(cls, *args, **kwargs):
        return cls._build_misc(*args, **kwargs)

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
                misc += (pickle.load(open(resolved_file_path, "rb")),)
            elif filename.endswith(".items"):
                misc += (open(resolved_file_path, "r", encoding="utf-8").readlines(),)
            elif filename.endswith(".txt"):
                misc += (open(resolved_file_path, "r", encoding="utf-8").read().strip().splitlines(),)
            else:
                continue

        if len(misc) == 1:
            misc = misc[0]

        return misc
