import torch
import numpy as np
from typing import Union, List

from .base import batchify, DoolyTaskWithModelTokenzier


class TextClassification(DoolyTaskWithModelTokenzier):
    @batchify("sentences1", "sentences2")
    @torch.no_grad()
    def predict_outputs(
        self,
        sentences1: Union[List[str], str],
        sentences2: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        # show_probs: bool = False,
    ):
        # Tokenize and get input_ids
        (inputs,) = self._preprocess(
            text=sentences1,
            text_pair=sentences2,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        # Predict tags and ignore <s> & </s> tokens
        inputs = self._prepare_inputs(inputs)
        logits = self._model(**inputs).logits
        results = logits.argmax(dim=-1).cpu().numpy()

        # Label mapping
        labelmap = lambda x: self._model.config.id2label[x]  # noqa
        labels = np.vectorize(labelmap)(results)

        # if show_probs:
        #     probs = softmax(logits.cpu().numpy()).tolist()
        #     probs = {label_fn(i): prob for i, prob in enumerate(probs)}
        #     return probs

        return list(map(str.capitalize, labels.tolist()))
