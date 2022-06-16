import abc
import json
import collections
from typing import Union, List, Optional

from transformers import PreTrainedTokenizer
from ..utils.import_utils import (
    is_available_kss,
    is_available_nltk,
)


InputTexts = Union[str, List[str]]
TokenizedOutput = Union[List[str], List[List[str]]]


class SentTokenizeMixin:
    """ Sentence Tokenization Mixin """

    def _set_sent_tokenizer(self):
        if self.lang in ["ko", "multi"]:
            if is_available_kss():
                from kss import split_sentences

                self._ko_sent_tokenizer = split_sentences
            else:
                raise ModuleNotFoundError("Please install kss with: `pip install kss`.")
        if self.lang in ["en", "multi"]:
            if is_available_nltk():
                import nltk

                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt")

                from nltk.tokenize import sent_tokenize

                self._en_sent_tokenizer = sent_tokenize
            else:
                raise ModuleNotFoundError(
                    "Please install nltk with: `pip install nltk`."
                )

    def sent_tokenize(
        self,
        texts: InputTexts,
        langs: Optional[InputTexts] = None,
    ) -> List[List[str]]:
        if isinstance(texts, str):
            texts = [texts]

        if langs is None:
            langs = self.lang
        elif self.lang != "multi":  # F632
            raise AttributeError("`langs` parameter is only used for `multi` model.")

        if isinstance(langs, str):
            langs = [langs] * len(texts)

        do_per_sample = False
        if len(set(langs)) == 1 and langs[0] == "ko":
            # korean sentence splitter can be batched
            if not hasattr(self, "_ko_sent_tokenizer"):
                raise AttributeError
            try:
                sentences = self._ko_sent_tokenizer(texts)
            except Exception:
                do_per_sample = True
        else:
            do_per_sample = True

        if do_per_sample:
            sentences = []
            for text, lang in zip(texts, langs):
                if lang in "ko":
                    if not hasattr(self, "_ko_sent_tokenizer"):
                        raise AttributeError
                    sentences.append(self._ko_sent_tokenizer(text))
                elif lang == "en":
                    if not hasattr(self, "_en_sent_tokenizer"):
                        raise AttributeError
                    sentences.append(self._en_sent_tokenizer(text))
                else:  # lang in ["ja", "zh"]
                    text = text.replace("。", "。[SEP]")
                    text = text.replace("！", "！[SEP]")
                    text = text.replace("？", "？[SEP]")
                    if "[SEP]" in text:
                        sents = text.split("[SEP]")
                        sents = sents[:-1]
                    else:
                        sents = [text]
                    sentences.append(sents)
        num_sentences = [len(sents) for sents in sentences]
        return sentences, num_sentences


class DoolyPreTrainedTokenizer(PreTrainedTokenizer, SentTokenizeMixin):
    vocab_files_names = {"vocab_file": "vocab.json"}
    replacement: Optional[str] = None

    def __init__(
        self,
        vocab_file,
        cls_token: str = "<s>",
        sep_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        task: str = None,
        lang: str = None,
        n_model: str = None,
        **kwargs
    ):
        super().__init__(
            cls_token=cls_token,
            sep_token=sep_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        replacement = kwargs.pop("replacement", None)
        self.replacement = replacement or self.replacement

        self.task = task
        self.lang = lang
        self.n_model = n_model

    @abc.abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        pass

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)