from typing import List, Optional

from .base import Tokenizer


class CharS1Tokenizer(Tokenizer):
    """ Character Tokenizer with style 1 """

    def _recover_original(self, decoded_text: str) -> str:
        return decoded_text.replace(" ", "").replace("▁", " ").strip()

    def _tokenize(self, text: str):
        x = text.strip()
        x = [c for c in self._normalize_space(x)]

        tokenized = list()
        for i in range(len(x)):
            if x[i] == " ":
                x[i+1] = f"▁{x[i+1]}"
                continue
            else:
                tokenized.append(x[i])
        tokenized[0] = f"▁{tokenized[0]}"
        return tokenized


class CharS2Tokenizer(Tokenizer):
    """ Character Tokenizer with style 2 """

    def _recover_original(self, decoded_text: str) -> str:
        return decoded_text.replace(" ", "").replace("▁", " ").strip()

    def _tokenize(self, text: str) -> List[str]:
        text = text.strip()
        text = text.replace(" ", "▁")
        text = " ".join([c for c in text])

        tokenized = self._normalize_space(text)
        tokenized = tokenized.split()
        return tokenized

    def tokenize(
        self,
        text: str,
        text_pair: Optional[str] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        **kwargs
    ) -> List[str]:
        if (src_lang is None) ^ (tgt_lang is None):
            src_lang = tgt_lang = None

        text = text.strip()
        if self.sub_tokenizer.get(src_lang, None) is not None:
            sub_tokenizer = self.sub_tokenizer[src_lang]
            if hasattr(sub_tokenizer, "segment"):
                tokenized = sub_tokenier.segment(text)
            elif hasattr(sub_tokenizer, "tokenize"):
                tokenized = sub_tokenizer.tokenize(text, add_special_tokens=False)
            else:
                raise AttributeError
        else:
            tokenized = self._tokenize(text)

        if src_lang is not None:
            tokenized = [self._langtok(src_lang)] + tokenized
        if tgt_lang is not None:
            tokenized = tokenized + [self._langtok(tgt_lang)]

        if add_special_tokens:
            tokenized += [self.sep_token]

        if text_pair is not None:
            tokenized += self._tokenize(text_pair)

            if add_special_tokens:
                tokenized += [self.sep_token]

        return tokenized
