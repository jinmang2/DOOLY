from typing import Union, List

import torch
import numpy as np
from transformers import PreTrainedTokenizerBase

from .base import DoolyTaskBase


class TextClassification(DoolyTaskBase):

    @torch.no_grad()
    def predict_outputs(
        self,
        sentences1: Union[List[str], str],
        sentences2: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        # show_probs: bool = False,
    ):
        # Get input_ids
        params = dict(
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if not issubclass(self._tokenizer.__class__, PreTrainedTokenizerBase):
            params.update(dict(no_separator=no_separator))
        inputs = self._tokenizer(sentences1, sentences2, **params)

        # Predict tags and ignore <s> & </s> tokens
        inputs = self._prepare_inputs(inputs)
        logits = self._model(**inputs).logits
        results = logits.argmax(dim=-1).cpu().numpy()

        # Label mapping
        labelmap = lambda x: self._model.config.id2label[x]
        labels = np.vectorize(labelmap)(results)

        # if show_probs:
        #     probs = softmax(logits.cpu().numpy()).tolist()
        #     probs = {label_fn(i): prob for i, prob in enumerate(probs)}
        #     return probs

        return list(map(str.capitalize, labels.tolist()))
