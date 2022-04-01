from typing import Dict, List, Set, Union
from functools import lru_cache

import torch

from .base import Tokenizer


TokenizedOutput = Union[List[str], List[List[str]]]
EncodedOutput = Union[List[int], List[List[int]], torch.Tensor]
DecodedOutput = Union[str, List[str]]


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


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        try:
            import regex as re

            self.re = re
        except ImportError:
            raise ImportError("Please install regex with: pip install regex")

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:  # noqa
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[str]:
        bpe_tokens = []
        for token in self.re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # bpe_tokens.extend(
            #     self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            # )
            bpe_tokens.extend(self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder.get(token, token) for token in tokens])
        text = self._decode(text)
        return text

    def _decode(self, text: str) -> str:
        return bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )


class Gpt2BpeTokenizer(Tokenizer):
    """ GPT2 BytePairEncoding Tokenizer """

    def _build_bpe(self, lang: str, encoder_json: Dict = None, bpe_merges: Dict = None):
        self._bpe = Encoder(encoder_json, bpe_merges)

    def __call__(
        self, *args, **kwargs
    ) -> Union[TokenizedOutput, Dict[str, EncodedOutput]]:
        return_tokens = kwargs.pop("return_tokens", False)
        add_special_tokens = kwargs.pop("add_special_tokens", False)

        if return_tokens and add_special_tokens:
            add_special_tokens = False

        kwargs.update(
            {"return_tokens": return_tokens, "add_special_tokens": add_special_tokens}
        )

        outputs = self.encode(*args, **kwargs)

        if return_tokens:
            _outputs = []
            if not isinstance(outputs[0], list):
                outputs = [outputs]
            for output in outputs:
                _outputs.append([self._bpe._decode(o) for o in output])
            if len(_outputs) == 1:
                _outputs = _outputs[0]
            outputs = _outputs

        return outputs

    def _tokenize(self, text: str) -> List[str]:
        # return list(map(str, self._bpe.encode(text)))
        return self._bpe.encode(text)

    def encode_line(
        self,
        tokenized: List[str],
        add_special_tokens: bool = False,
        use_pos_vocab: bool = False,
    ) -> List[int]:
        encoded = []
        for bpe_token in tokenized:
            if bpe_token not in ["<s>", "</s>", "<pad>", "<unk>"]:
                bpe_token = str(self._bpe.encoder[bpe_token])
            encoded.append(self.vocab.get(bpe_token, self.unk_token_id))

        if add_special_tokens:
            encoded = [self.cls_token_id] + encoded + [self.sep_token_id]
        return encoded

    def decode_line(self, ids: List[int], ignore_symbols: Set[int] = {}) -> str:
        x = super().decode_line(ids, ignore_symbols)
        return self._bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )


class BpeJaZhTokenizer(Tokenizer):
    """ BytePairEncoding Tokenizer for Ja and Zh """

    def _build_bpe(self, lang: str, encoder_json: Dict = None, bpe_merges: Dict = None):
        if lang == "ja":
            try:
                import ipadic  # noqa
            except ImportError:
                raise ImportError("Please install ipadic with: `pip install ipadic`")
            try:
                import fugashi  # noqa
            except ImportError:
                raise ImportError("Please install fugashi with: `pip install fugashi`")
            from transformers import BertJapaneseTokenizer

            model_name_or_path = "cl-tohoku/bert-base-japanese-whole-word-masking"
            self._bpe = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        elif lang == "zh":
            from transformers import BertTokenizer

            self._bpe = BertTokenizer.from_pretrained(
                "bert-base-chinese", do_lower_case=True
            )

    def _tokenize(self, text: str) -> List[str]:
        return self._bpe.tokenize(text)

    def _recover_original(self, decoded_text: str) -> str:
        return decoded_text.replace(" ", "")
