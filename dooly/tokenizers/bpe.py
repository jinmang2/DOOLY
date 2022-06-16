from typing import List
from functools import lru_cache

from dataclasses import dataclass
from transformers import GPT2TokenizerFast
from transformers import BertTokenizer, BertJapaneseTokenizer

from . import load_dooly_tokenizer
from .base import DoolyPreTrainedTokenizer
from ..utils import recover_original_hf_bucket_url
from ..utils.import_utils import is_available_ipadic, is_available_fugashi


"""
Byte pair encoding utilities from GPT-2.
Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
"""


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class DoolyGPT2TokenizerFast(DoolyPreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "bpe_merge_file": "vocab.bpe",
        "bpe_vocab_file": "encoder.json",
    }

    def __init__(
        self,
        bpe_vocab_file=None,
        bpe_merge_file=None,
        errors: str = "replace",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bpe = GPT2TokenizerFast(
            vocab_file=bpe_vocab_file, merges_file=bpe_merge_file
        )
        self.bpe_vocab = self.bpe.backend_tokenizer.get_vocab()
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self.errors = errors

    @property
    def bpe_vocab_size(self) -> int:
        return len(self.get_bpe_vocab())

    def get_bpe_vocab(self):
        return self.bpe_vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return self.bpe.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        bpe_token = token
        if token not in self.all_special_tokens:
            bpe_token = self.bpe._tokenizer.token_to_id(token)
        return self.vocab.get(str(bpe_token), self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        bpe_id = self.ids_to_tokens.get(index, self.unk_token)
        if bpe_id not in self.all_special_tokens:
            bpe_id = int(bpe_id)
        return self.bpe._tokenizer.id_to_token(bpe_id)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text])\
            .decode("utf-8", errors=self.errors)
        return text


class DoolyBertTokenizer(DoolyPreTrainedTokenizer):
    replacement: str = "##"
    wp_path: str = None
    wp_tok_class: PreTrainedTokenizer = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with recover_original_hf_bucket_url():
            wp_path = kwargs.pop("wp_path", self.wp_path)
            self.wordpiece = load_dooly_tokenizer(wp_path, self.wp_tok_class)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return self.wordpiece.tokenize(text)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(
            [token.replace(" ", "").replace(self.replacement, "")
             for token in tokens]
        )
        return text


class DoolyBertJaTokenizer(DoolyBertTokenizer):
    wp_path: str = "cl-tohoku/bert-base-japanese-whole-word-masking"
    wp_tok_class: PreTrainedTokenizer = BertJapaneseTokenizer

    def __init__(self, **kwargs):
        if is_available_ipadic():
            import ipadic  # noqa
        else:
            raise ModuleNotFoundError(
                "Please install ipadic with: `pip install ipadic`")

        if is_available_fugashi():
            import fugashi  # noqa
        else:
            raise ModuleNotFoundError(
                "Please install fugashi with: `pip install fugashi`")

        super().__init__(**kwargs)


class DoolyBertZhTokenizer(DoolyBertTokenizer):
    wp_path: str = "bert-base-chinese"
    wp_tok_class: PreTrainedTokenizer = BertTokenizer