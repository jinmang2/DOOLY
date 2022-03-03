from typing import Union, List

import numpy as np

from .base import DoolyTaskBase


class SequenceTagging(DoolyTaskBase):

    def predict_tags(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True, # ENBERTa, JaBERTa, ZhBERTa에선 없음
        no_separator: bool = False,
        do_sent_split: bool = True,
    ):
        # Tokenize
        # Only support DoolyTokenizer subclass
        tokens = self._tokenizer(
            sentences,
            return_tokens=True,
            add_special_tokens=False
        )
        tokens = [tokens] if len(sentences) == 1 else tokens
        inputs = self._tokenizer(
            sentences,
            return_tensors=True,
            no_separator=no_separator,
            add_special_tokens=add_special_tokens,
        )

        # Predict tags and ignore <s> & </s> tokens
        inputs = self._prepare_inputs(inputs)
        logits = self._model(**inputs).logits
        if add_special_tokens:
            logits = logits[:, 1:-1, :]
        results = logits.argmax(dim=-1).cpu().numpy()

        # Label mapping
        labelmap = lambda x: self._model.config.id2label[x]
        labels = np.vectorize(labelmap)(results)

        token_label_pairs = [
            [
                (tok, l)
                for tok, l in zip(sent, label)
            ] for sent, label in zip(tokens, labels)
        ]

        return token_label_pairs
