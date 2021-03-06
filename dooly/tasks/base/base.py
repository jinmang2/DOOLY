import os
import abc
import json
import pickle
import zipfile
import inspect
from dataclasses import dataclass
from typing import Dict, Union, Optional, Tuple, List, Any, Callable

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import numpy_pad_and_concatenate

from ...build_utils import download_from_hf_hub, HUB_NAME, DEFAULT_DEVICE
from ...tokenizers import DoolyTokenizer
from ...models import DoolyModel


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# key: func_name, value: batch_args
_COLUMN_NAME_USED_IN_BATCH = {}


def batchify(*batch_args):
    """ Batchify inference methods """

    def _batchify(func: Callable):
        # Executed only when a function decorated with `batchify` is called
        # for the first time
        func_name = func.__name__
        if func_name not in _COLUMN_NAME_USED_IN_BATCH:
            _COLUMN_NAME_USED_IN_BATCH[func_name] = list(batch_args)
        batch_col_names = _COLUMN_NAME_USED_IN_BATCH[func_name]

        def merge(outputs, batch_outputs, padding_index: int = 0):
            if isinstance(batch_outputs, torch.Tensor):
                batch_outputs = batch_outputs.detach().cpu().numpy()
            if outputs is None:
                return batch_outputs
            assert type(outputs) == type(batch_outputs)
            if isinstance(batch_outputs, list):
                outputs += batch_outputs
            elif isinstance(batch_outputs, np.ndarray):
                outputs = numpy_pad_and_concatenate(
                    outputs, batch_outputs, padding_index=padding_index
                )
            return outputs

        # *args == (self,), **kwargs: function's keyword arguments
        def wrapper(*args, **kwargs):
            batch_size = kwargs.pop("batch_size", 1)
            verbose = kwargs.pop("verbose", True)

            batch_cols = {}
            for i, col_name in enumerate(batch_col_names):
                col = kwargs.pop(col_name, None)
                if i == 0:
                    n_samples = len(col)  # since list
                if col is None:
                    continue
                batch_cols.update({col_name: col})
            batch_cols = Dataset.from_dict(batch_cols)

            padding_index = 0
            if hasattr(args[0], "tokenizer"):
                padding_index = args[0].tokenizer.pad_token_id

            outputs = None
            for i in tqdm(
                range((n_samples - 1) // batch_size + 1), disable=not verbose
            ):
                cols = batch_cols[i * batch_size : (i + 1) * batch_size]  # noqa
                batch_outputs = func(*args, **cols, **kwargs)
                if isinstance(batch_outputs, tuple):
                    if outputs is None:
                        outputs = [None] * len(batch_outputs)
                    for ix, (output, batch_output) in enumerate(
                        zip(outputs, batch_outputs)
                    ):
                        outputs[ix] = merge(output, batch_output, padding_index)
                else:
                    outputs = merge(outputs, batch_outputs, padding_index)

            return outputs

        return wrapper

    return _batchify


@dataclass
class DoolyTaskConfig:
    lang: str
    n_model: str
    device: str
    misc_tuple: Tuple


class DoolyTaskBase:
    def __init__(self, config: DoolyTaskConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    def finalize(self):
        pass

    def __repr__(self):
        task_info = f"[TASK]: {self.__class__.__name__}"
        category = f"[CATEGORY]: {self.__class__.__mro__[1].__name__}"
        lang_info = f"[LANG]: {self.lang}"
        device_info = f"[DEVICE]: {self.device}" if self.device else ""
        model_info = f"[MODEL]: {self.n_model}"
        return "\n".join([task_info, category, lang_info, device_info, model_info])

    @property
    def lang(self):
        return self.config.lang

    @property
    def n_model(self):
        return self.config.n_model

    @property
    def device(self):
        return self.config.device

    @classmethod
    def build(
        cls,
        lang: str = None,
        n_model: str = None,
        tok_kwargs: Dict = {},
        model_kwargs: Dict = {},
        **kwargs,
    ):
        lang, n_model = cls._check_validate_input(lang, n_model)

        # parse device option from kwargs
        device = kwargs.pop("device", DEFAULT_DEVICE)

        # parse download keyword arguments
        dl_kwargs = {}
        dl_kwargs["revision"] = kwargs.pop("revision", None)
        dl_kwargs["cache_dir"] = kwargs.pop("cache_dir", None)
        dl_kwargs["force_download"] = kwargs.pop("force_download", False)
        dl_kwargs["resume_download"] = kwargs.pop("resume_download", False)

        # set tokenizer
        tokenizer = cls.build_tokenizer(
            cls.task, lang, n_model, **dl_kwargs, **tok_kwargs
        )
        # set model
        model = cls.build_model(cls.task, lang, n_model, **dl_kwargs, **model_kwargs)
        # set misc
        misc_files = cls.misc_files.get(lang, [])
        misc = cls.build_misc(lang, n_model, misc_files, **dl_kwargs)

        config = DoolyTaskConfig(
            lang=lang, n_model=n_model, device=device, misc_tuple=misc
        )

        init_params = inspect.signature(cls.__init__).parameters
        init_kwargs = {"config": config}
        if tokenizer is not None and "tokenizer" in init_params:
            init_kwargs.update({"tokenizer": tokenizer})
        if model is not None and "model" in init_params:
            init_kwargs.update({"model": model})

        return cls(**init_kwargs)

    @staticmethod
    def build_tokenizer(task: str, lang: str, n_model: str, **kwargs):
        return None

    @staticmethod
    def build_model(task: str, lang: str, n_model: str, **kwargs):
        return None

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
            lang = cls.get_default_lang()
        cls._check_available_lang(lang)
        if n_model is None:
            n_model = cls.get_default_model(lang)
        cls._check_available_model(lang, n_model)
        return lang, n_model

    @classmethod
    def get_default_lang(cls) -> str:
        return cls.available_langs[0]

    @classmethod
    def get_default_model(cls, lang: str) -> str:
        return cls.available_models[lang][0]

    @classmethod
    def build_misc(cls, *args, **kwargs) -> Tuple:
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

            # TODO: pickle, json ??? ?????? ????????? ?????? ?????? ??????
            if filename.endswith(".pkl"):
                try:
                    misc += (pickle.load(open(resolved_file_path, "rb")),)
                except pickle.UnpicklingError:
                    misc += (resolved_file_path,)
            elif filename.endswith(".json"):
                misc += (json.load(open(resolved_file_path, "r", encoding="utf-8")),)
            elif filename.endswith(".items"):
                misc += (open(resolved_file_path, "r", encoding="utf-8").readlines(),)
            elif filename.endswith(".txt"):
                misc += (
                    open(resolved_file_path, "r", encoding="utf-8")
                    .read()
                    .strip()
                    .splitlines(),
                )
            elif filename.endswith(".zip"):
                zip_file = zipfile.ZipFile(resolved_file_path)
                zip_path = os.path.join(*resolved_file_path.split("/")[:-1])
                zip_file.extractall(zip_path)
                zip_file.close()
                misc += (os.path.join(zip_path, filename.split(".zip")[0]),)
            else:
                continue

        return misc


class DoolyTaskWithModelTokenzier(DoolyTaskBase):
    def finalize(self):
        self._model.to(self.config.device)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def build_tokenizer(task: str, lang: str, n_model: str, **kwargs):
        return DoolyTokenizer.build_tokenizer(task, lang, n_model, **kwargs)

    @staticmethod
    def build_model(task: str, lang: str, n_model: str, **kwargs):
        return DoolyModel.build_model(task, lang, n_model, **kwargs)

    def _prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """ Prepare input to be placed on the same device in inference. """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._model.device)
        return inputs

    def _remove_unused_columns(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        signature = inspect.signature(self.model.forward)
        new_inputs = {}
        for k, v in inputs.items():
            if k in signature.parameters:
                new_inputs[k] = v
        return new_inputs

    def _preprocess(
        self,
        text: Union[List[str], str],
        text_pair: Union[List[str], str] = None,
        src_lang: Union[List[str], str] = None,
        tgt_lang: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        return_tokens: bool = False,
        return_tags: bool = False,
    ):
        inputs = self.get_inputs(
            text=text,
            text_pair=text_pair,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
            return_tags=return_tags,
        )

        outputs = (inputs,)
        if return_tokens:
            tokens = self.get_tokens(
                text=text,
                text_pair=text_pair,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                add_special_tokens=add_special_tokens,
                no_separator=no_separator,
                return_tags=return_tags,
            )
            outputs += (tokens,)

        return outputs

    def get_tokens(
        self,
        text: Union[List[str], str],
        text_pair: Union[List[str], str] = None,
        src_lang: Union[List[str], str] = None,
        tgt_lang: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        return_tags: bool = False,
    ):
        if not issubclass(self.tokenizer.__class__, PreTrainedTokenizerBase):
            tokens = self.tokenizer(
                text,
                text_pair,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                return_tokens=True,
                add_special_tokens=False,
                return_tags=return_tags,
            )
        else:
            # src_lang and tgt_lang must to be None
            if hasattr(self.tokenizer, "segment"):
                tokens = self.tokenizer.segment(text, text_pair)
            else:
                tokens = self.tokenizer.tokenize(
                    text, text_pair, add_special_tokens=False
                )

        return tokens

    def get_inputs(
        self,
        text: Union[List[str], str],
        text_pair: Union[List[str], str] = None,
        src_lang: Union[List[str], str] = None,
        tgt_lang: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        return_tags: bool = False,
    ):
        params = dict(
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if len(text) > 1:
            params.update({"padding": True})
        if not issubclass(self.tokenizer.__class__, PreTrainedTokenizerBase):
            params.update(
                {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "no_separator": no_separator,
                    "return_tags": return_tags,
                }
            )
        inputs = self.tokenizer(text, text_pair, **params)
        return inputs
