import json
import inspect
from functools import partial
from typing import Dict, Union, Optional, Tuple, List, TypeVar

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import Tokenizer
from .bpe import Gpt2BpeTokenizer, BpeJaZhTokenizer
from .char import CharS1Tokenizer, CharS2Tokenizer
from .tokenization_roberta import RobertaTokenizerFast
from ..build_utils import (
    download_from_hf_hub,
    HUB_NAME,
    VOCAB_NAME,
    TOKENIZER_USER_AGENT
)


DoolyTokenizerHub = {
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
    "wsd": {
        "ko": {"transformer.large": CharS2Tokenizer},
    },
}
available_tasks = list(DoolyTokenizerHub.keys())


class DoolyTokenizer:

    @classmethod
    def build_tokenizer(cls, task: str, lang: str, n_model: Optional[str] = None, **kwargs):
        assert task in available_tasks, (
            f"Task `{task}` is not available. See here {available_tasks}."
        )
        available_langs = DoolyTokenizerHub[task]
        assert lang in available_langs, (
            f"Language `{lang}` is not available in this task {task}. "
            f"See here {available_langs}."
        )
        available_models = available_langs[lang]
        if n_model is None:
            n_model = list(available_models.keys())[0]
        assert n_model in available_models, (
            f"Model `{n_model}` is not available in this task-lang pair. "
            f"See here {available_models}."
        )

        tokenizer_class = available_models[n_model]

        # return BuildMixin._build_tokenizer(task, lang, n_model, tokenizer_class, **kwargs)
        return cls._build_tokenizer(task, lang, n_model, tokenizer_class, **kwargs)

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

        if issubclass(tokenizer_class, PreTrainedTokenizerBase):
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
