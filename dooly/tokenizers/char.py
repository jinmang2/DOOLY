from typing import List

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

    def _tokenize(self, text: str) -> List[str]:
        text = text.replace(" ", "▁")
        text = " ".join([c for c in text])

        tokenized = self._normalize_space(text)
        tokenized = tokenized.split()
        return tokenized

    def tokenize(
        self,
        text: str,
        text_pair: str = None,
        add_special_tokens: bool = False,
        no_separator: bool = False,
    ) -> List[str]:
        tokenized = self._tokenize(text)

        if add_special_tokens:
            tokenized += [self.sep_token]

        if text_pair is not None:
            tokenized += self._tokenize(text_pair)

            if add_special_tokens:
                tokenized += [self.sep_token]

        return tokenized
