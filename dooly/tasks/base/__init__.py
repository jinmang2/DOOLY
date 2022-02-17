from typing import Dict, Union, Tuple, List, Any

import numpy as np
import torch

from ...build import BuildMixin


class DoolyTaskBase(BuildMixin):

    def __init__(self, lang: str, n_model: str):
        self.lang = lang
        self.n_model = n_model

    def __repr__(self):
        task_info = f"[TASK]: {self.__class__.__name__}"
        lang_info = f"[LANG]: {self.lang}"
        model_info = f"[MODEL]: {self.n_model}"
        return "\n".join([task_info, lang_info, model_info])

    def _prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """ Prepare input to be placed on the same device in inference. """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._model.device)
        return inputs

    def predict(
        self,
        head: str,
        tokens: torch.LongTensor,
        segments: torch.LongTensor,
        return_logits: bool = False,
    ):
        raise NotImplementedError(
            "`func: predict` is not implemented properly!")

    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:
        raise NotImplementedError(
            "`func: predict_span` is not implemented properly!")

    def predict_output(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True, # ENBERTa에선 없음
        no_separator: bool = False,
        show_probs: bool = False,
    ) -> Union[str, Dict]:
        raise NotImplementedError(
            "`func: predict_output` is not implemented properly!")

    def predict_srl(
        self,
        sentence: str,
        segment: str,
    ):
        raise NotImplementedError(
            "`func: predict_srl` is not implemented properly!")

    def predict_tags(
        self,
        sentence: Union[List[str], str],
        add_special_tokens: bool = True, # ENBERTa, JaBERTa, ZhBERTa에선 없음
        no_separator: bool = False,
        do_sent_split: bool = True,
    ):
        """
        Supported task: NER
        """
        # Tokenize
        # Only support DoolyTokenizer subclass
        tokens = self._tokenizer(
            sentence,
            return_tokens=True,
            add_special_tokens=False
        )
        token = [token] if len(texts) == 1 else token
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

    def fill_mask(self, masked_input: str, topk: int = 5): # PosRoBERTa에선 topk가 15
        raise NotImplementedError(
            "`func: fill_mask` is not implemented properly!")

    def predict_dependency(
        self,
        tokens: torch.LongTensor,
        segments: torch.LongTensor,
    ):
        raise NotImplementedError(
            "`func: predict_dependency` is not implemented properly!")
