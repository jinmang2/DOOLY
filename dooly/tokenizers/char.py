import re
from typing import Union, List, Dict, Optional

import torch
from transformers import BatchEncoding, TensorType

from .base import DoolyPreTrainedTokenizer


class DoolyCharTokenizer(DoolyPreTrainedTokenizer):
    replacement: str = "▁"
    __SPACE_NORMALIZER__ = re.compile(r"\s+")

    def _normalize_space(self, text: str) -> str:
        return self.__SPACE_NORMALIZER__.sub(" ", text).strip()

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        return text.replace(" ", "").replace(self.replacement, " ").strip()

    def _tokenize_chatbpe_style(self, text: str) -> List[str]:
        """e.g.,
        >>> text = "손흥민은 28세의 183 센티미터, 77 킬로그램이며, 현재 주급은 약 3억 원이다."
        >>> tokenizer._tokenize_charbpe_style(text)
        ['▁손', '흥', '민', '은', '▁2', '8', '세', '의', '▁1', '8', '3', '▁센', '티', '미', '터',
         ',', '▁7', '7', '▁킬', '로', '그', '램', '이', '며', ',', '▁현', '재', '▁주', '급', '은',
         '▁약', '▁3', '억', '▁원', '이', '다', '.']
        """
        x = text.strip()
        x = [c for c in self._normalize_space(x)]

        tokenized = list()
        for i in range(len(x)):
            if x[i] == " ":
                x[i + 1] = self.replacement + f"{x[i + 1]}"
                continue
            else:
                tokenized.append(x[i])
        tokenized[0] = self.replacement + f"{tokenized[0]}"
        return tokenized

    def _tokenize_whitespace_style(self, text: str) -> List[str]:
        """e.g.,
        >>> text = "손흥민은 28세의 183 센티미터, 77 킬로그램이며, 현재 주급은 약 3억 원이다."
        >>> tokenizer._tokenize_whitespace_style(text)
        ['손', '흥', '민', '은', '▁', '2', '8', '세', '의', '▁', '1', '8', '3', '▁', '센', '티',
         '미', '터', ',', '▁', '7', '7', '▁', '킬', '로', '그', '램', '이', '며', ',', '▁', '현',
         '재', '▁', '주', '급', '은', '▁', '약', '▁', '3', '억', '▁', '원', '이', '다', '.']
        """
        text = text.strip()
        text = text.replace(" ", self.replacement)
        text = " ".join([c for c in text])

        text = self._normalize_space(text)
        tokenized = text.split()
        return tokenized


class DoolyCharBertTokenizer(DoolyCharTokenizer):
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return self._tokenize_chatbpe_style(text)


# TODO: WSD를 위한 target text tokenize function 작성
# PORORO WSD transformer:
#   src_tokens: token_ids_0 + [</s>]
#   tgt_tokens: ??? -> 분석 필요
class DoolyCharSeq2SeqWsdTokenizer(DoolyCharTokenizer):
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return self._tokenize_whitespace_style(text)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return token_ids_0 + sep
        return token_ids_0 + sep + token_ids_1 + sep


class DoolyCharSeq2SeqNmtTokenizer(DoolyCharTokenizer):
    """
    - mBART:
      src_tokens: token_ids_0 + [</s>] + [src_lang_code]
      tgt_tokens: token_ids_1 + [</s>] + [tgt_lang_code]
    - PORORO NMT transformer:
      src_tokens: [src_lang_code] + token_ids_0 + [tgt_lang_code] + [</s>]
      tgt_tokens: [<s>] + token_ids_1 + [</s>]
    """

    __LANG_TO_CODE__ = {
        "ko": "[ko_KR]",
        "en": "[en_XX]",
        "ja": "[ja_XX]",
        "zh": "[zh_CN]",
    }

    def __init__(self, **kwargs):
        super().__init__(
            additional_special_tokens=list(self.lang_to_code.values()), **kwargs
        )

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return self._tokenize_whitespace_style(text)

    def __call__(
        self,
        text,
        text_pair,
        src_lang: Union[str, List[str]] = None,
        tgt_lang: Union[str, List[str]] = None,
        **kwargs,
    ) -> BatchEncoding:
        if src_lang is None and tgt_lang is None:
            return super().__call__(text, text_pair, **kwargs)

        assert text_pair is not None

        add_special_tokens = kwargs.pop("add_special_tokens", True)
        return_tensors = kwargs.get("return_tensors", None)

        # tokenize source text
        batch_encodings = super().__call__(text, add_special_tokens=False, **kwargs)
        batch_encodings = self.add_language_tokens(
            batch_encodings, src_lang, tgt_lang, add_special_tokens, return_tensors
        )

        # tokenize target text
        label_encodings = super().__call__(
            text_pair, add_special_tokens=add_special_tokens, **kwargs
        )

        batch_encodings["labels"] = label_encodings["input_ids"]
        return batch_encodings

    @property
    def lang_to_code(self) -> Dict[str, str]:
        return self.__LANG_TO_CODE__

    @property
    def lang_code_to_id(self) -> Dict[str, int]:
        return {
            code: self.vocab.get(code, self.unk_token_id)
            for code in self.lang_to_code.values()
        }

    def lang_to_id(self, lang: str) -> int:
        lang_code = self.lang_to_code.get(lang, self.unk_token)
        return self.lang_code_to_id.get(lang_code, self.unk_token_id)

    def add_language_tokens(
        self,
        batch_encodings: BatchEncoding,
        src_lang: Union[str, List[str]],
        tgt_lang: Union[str, List[str]],
        add_special_tokens: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:
        if return_tensors is not None and return_tensors != "pt":
            raise ValueError("Only support tensor type `pt`.")

        input_ids = batch_encodings.pop("input_ids")
        attention_mask = batch_encodings.get("attention_mask", None)
        token_type_ids = batch_encodings.get("token_type_ids", None)

        src_langs = src_lang
        if isinstance(src_lang, str):
            src_langs = [src_lang] * len(input_ids)
        tgt_langs = tgt_lang
        if isinstance(tgt_lang, str):
            tgt_langs = [tgt_lang] * len(input_ids)

        assert len(src_langs) == len(input_ids)
        assert len(tgt_langs) == len(input_ids)

        token_added_ids = []
        if attention_mask is not None:
            token_added_masks = []
        if token_type_ids is not None:
            token_added_type_ids = []

        for i in range(len(input_ids)):
            _input_ids = input_ids[i]
            if attention_mask is not None:
                _attention_mask = attention_mask[i]
            if token_type_ids is not None:
                _token_type_ids = token_type_ids[i]
            src_lang = src_langs[i]
            tgt_lang = tgt_langs[i]

            maximum_idx = [
                i for i, val in enumerate(_input_ids) if val != self.pad_token_id
            ]
            idx_to_add = 0
            if len(maximum_idx) > 0:
                idx_to_add = max(maximum_idx) + 1

            src_lang = self.lang_to_id(src_lang)
            tgt_lang = self.lang_to_id(tgt_lang)
            sep = self.sep_token_id

            _input_ids = self.insert_tokens(
                _input_ids, [src_lang], [tgt_lang, sep], idx_to_add, return_tensors
            )
            token_added_ids.append(_input_ids)

            if attention_mask is not None:
                _attention_mask = self.insert_tokens(
                    _attention_mask, [1], [1, 1], idx_to_add, return_tensors
                )
                token_added_masks.append(_attention_mask)

            if token_type_ids is not None:
                _token_type_ids = self.insert_tokens(
                    _token_type_ids, [0], [0, 0], idx_to_add, return_tensors
                )
                token_added_type_ids.append(_token_type_ids)

        def unsqueeze_and_cat(tensorlist: List[torch.Tensor]) -> torch.Tensor:
            dims = [0] * len(tensorlist)
            tensorlist = list(map(torch.unsqueeze, tensorlist, dims))
            return torch.cat(tensorlist)

        if return_tensors:
            token_added_ids = unsqueeze_and_cat(token_added_ids)
            if attention_mask is not None:
                token_added_masks = unsqueeze_and_cat(token_added_masks)
            if token_type_ids is not None:
                token_added_type_ids = unsqueeze_and_cat(token_added_type_ids)

        batch_encodings["input_ids"] = token_added_ids
        if attention_mask is not None:
            batch_encodings["attention_mask"] = token_added_masks
        if token_type_ids is not None:
            batch_encodings["token_type_ids"] = token_added_type_ids
        return batch_encodings

    def insert_tokens(
        self,
        ids: Union[List[int], torch.Tensor],
        prefix: List[int],
        suffix: List[int],
        idx_to_add: int,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Union[List[int], torch.Tensor]:
        if return_tensors:
            ids = torch.cat(
                [
                    torch.tensor(prefix, requires_grad=False),
                    ids[:idx_to_add],
                    torch.tensor(suffix, requires_grad=False),
                    ids[idx_to_add:],
                ]
            ).long()
            return ids
        return prefix + ids[:idx_to_add] + suffix + ids[idx_to_add:]
