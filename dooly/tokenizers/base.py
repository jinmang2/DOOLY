import re
import unicodedata
from abc import abstractmethod
from typing import List, Union, Dict, Set, Callable, Optional

import torch


SPACE_NORMALIZER = re.compile(r"\s+")

TokenizedOutput = Union[List[str], List[List[str]]]
EncodedOutput = Union[List[int], List[List[int]], torch.Tensor]
PaddedOutput = Union[List[List[int]], torch.Tensor]
DecodedOutput = Union[str, List[str]]


class _BaseTokenizer:

    def __init__(
        self,
        lang: str,
        vocab: Dict[str, int],
        cls_token: str = "<s>",
        sep_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        padding_side: str = "right",
        max_seq_length: int = 512,
    ):
        assert padding_side in ["right", "left"]
        self.lang = lang
        self.vocab = vocab
        self.id2token = {i: tok for tok, i in vocab.items()}
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.padding_side = padding_side
        self.max_seq_length = max_seq_length

    @property
    def cls_token_id(self) -> int:
        return self.vocab[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self.vocab[self.sep_token]

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def nspecial(self) -> int:
        return 4 # cls, sep, pad, unk

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[str, bool] = False,
        return_tokens: bool = False,
        return_tensors: Union[str, bool] = False,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Union[TokenizedOutput, Dict[str, EncodedOutput]]:
        return self.encode(
            text=text,
            text_pair=text_pair,
            padding=padding,
            return_tokens=return_tokens,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

    def _normalize(self, text: str) -> str:
        """ Unicode normalization and whitespace removal (often needed for context) """
        text = unicodedata.normalize("NFKC", data)
        text = self._normalize_space(text)
        return text

    @staticmethod
    def _normalize_space(text: str) -> str:
        return SPACE_NORMALIZER.sub(" ", text).strip()

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        pass

    def tokenize(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        no_separator: bool = False,
    ) -> List[str]:
        tokenized = self._tokenize(text)

        if add_special_tokens:
            tokenized = [self.cls_token] + tokenized + [self.sep_token]

        if text_pair is not None:
            tokenized += [self.sep_token] if not no_separator else []
            tokenized += self._tokenize(text_pair)

            if add_special_tokens:
                tokenized += [self.sep_token]

        return tokenized

    def encode_line(
        self,
        tokenized: List[str],
        add_special_tokens: bool = False
    ) -> List[int]:
        encoded = []
        for token in tokenized:
            encoded.append(self.vocab.get(token, self.unk_token_id))

        if add_special_tokens:
            encoded = [self.cls_token_id] + encoded + [self.sep_token_id]
        return encoded

    def encode(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[str, bool] = False,
        return_tokens: bool = False,
        return_tensors: Union[str, bool] = False,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Union[TokenizedOutput, Dict[str, EncodedOutput]]:
        """ Encode tokens to ids, used for single or batched sentence """

        assert isinstance(return_tensors, bool) or return_tensors == "pt"
        return_tensors = (return_tensors == "pt") or return_tensors

        assert text_pair is None or type(text) == type(text_pair)

        if isinstance(text, str):
            return self.encode(
                text=[text],
                text_pair=[text_pair],
                padding=padding,
                return_tokens=return_tokens,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                add_special_tokens=add_special_tokens,
                no_separator=no_separator,
            )

        if text_pair is None:
            text_pair = [None] * len(text)

        assert len(text) == len(text_pair)

        texts, text_pairs = text, text_pair
        input_ids = []

        for text, text_pair in zip(texts, text_pairs):
            tokenized = self.tokenize(
                text=text,
                text_pair=text_pair,
                no_separator=no_separator,
                add_special_tokens=add_special_tokens,
            )
            encoded = None
            attn_mask = False

            if not return_tokens:
                encoded = self.encode_line(tokenized=tokenized)

            input_ids.append(tokenized if return_tokens else encoded)

        if return_tokens:
            input_ids = input_ids if len(texts) > 1 else input_ids[0]
            return input_ids

        attention_mask = None
        if return_tensors or padding:
            padded = self.pad(
                sequences={"input_ids": input_ids},
                padding=padding,
                return_tensors=return_tensors,
            )
            input_ids = padded["input_ids"]
            attention_mask = padded["attention_mask"]

        batch_encoding = {"input_ids": input_ids}

        if return_attention_mask and attention_mask is not None:
            batch_encoding.update({"attention_mask": attention_mask})

        return batch_encoding

    def decode_line(self, ids: List[int], ignore_symbols: Set[int] = {}) -> str:
        sent = []
        for _id in ids:
            if _id not in ignore_symbols:
                sent.append(self.id2token.get(_id, self.unk_token))
        return " ".join(sent)

    def _recover_original(self, decoded_text: str) -> str:
        return decoded_text

    def decode(
        self,
        ids: EncodedOutput,
        ignore_symbols: List[int] = [],
    ) -> DecodedOutput:

        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()

        if isinstance(ids[0], int):
            return self.decode(
                ids=[ids],
                ignore_symbols=ignore_symbols,
                recover_original=recover_original,
            )

        ignore_symbols = set(None or ignore_symbols)
        ignore_symbols.update(
            [self.cls_token_id, self.sep_token_id, self.pad_token_id])

        list_of_ids = ids
        decoded_texts = []
        for ids in list_of_ids:
            decoded = self.decode_line(ids, ignore_symbols)
            decoded = self._recover_original(decoded)
            decoded_texts.append(decoded)

        if len(decoded_texts) == 1:
            decoded_texts = decoded_texts[0]

        return decoded_texts

    def pad(
        self,
        sequences: Dict[str, EncodedOutput],
        padding: Union[str, bool] = True,
        return_tensors: bool = True,
        pad_to_multiple_of: Union[int, bool] = False, # match to hf pad methdo
    ) -> Dict[str, PaddedOutput]:
        """Pad batched sequences.
        if return_tensors, then return torch.LongTensor object.
        """

        input_ids = sequences.get("input_ids")
        assert input_ids is not None

        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        max_length = -1
        if padding == "max_length":
            max_length = self.max_seq_length
        else:
            max_length = max(len(ids) for ids in input_ids)

        padded = {"input_ids": [], "attention_mask": []}
        for ids in input_ids:
            seq_len = len(ids)
            if self.padding_side == "right":
                ids = ids + [self.pad_token_id] * (max_length - seq_len)
                attn_mask = [1] * seq_len + [0] * (max_length - seq_len)
            else:
                ids = [self.pad_token_id] * (max_length - seq_len) + ids
                attn_mask = [0] * (max_length - seq_len) + [1] * seq_len
            padded["input_ids"].append(ids)
            padded["attention_mask"].append(attn_mask)

        if return_tensors:
            for k, v in padded.items():
                padded[k] = torch.LongTensor(v)

        return padded


class Tokenizer(_BaseTokenizer):
    """ Whitespace Base Tokenizer with sentence tokenizer """

    def _set_sent_tokenizer(self):
        if self.lang in ["ko", "multi"]:
            from kss import split_sentences
            self._ko_sent_tokenizer = split_sentences
        if self.lang in ["en", "multi"]:
            from nltk.tokenize import sent_tokenize
            self._en_sent_tokenizer = sent_tokenize

    def sent_tokenize(
        self,
        texts: Union[str, List[str]],
        langs: Optional[Union[str, List[str]]] = None,
    ) -> List[List[str]]:
        if isinstance(texts, str):
            texts = [texts]

        if langs is None:
            langs = self.lang
        elif self.lang is not "multi":
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
            except:
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
                else: # lang in ["ja", "zh"]
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
