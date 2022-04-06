from typing import Union, List, Dict
import numpy as np

from .natural_language_inference import NaturalLanguageInference


class ZeroShotClassification(NaturalLanguageInference):
    """
    Conduct zero-shot topic classification

    English (`roberta.base.en.nli`)
        - dataset: MNLI (Adina Williams et al. 2017)
        - metric: N/A

    Korean (`brainbert.base.ko.kornli`)
    - metric: N/A
        - dataset: KorNLI (Ham et al. 2020)

    Japanese (`jaberta.base.ja.nli`)
    - metric: N/A
        - dataset: XNLI (Alexis Conneau et al. 2018)

    Chinese (`zhberta.base.zh.nli`)
        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: N/A

    Args:
        sent (str): sentence to be classified
        labels (List[str]): candidate labels

    Returns:
        List[Tuple(str, float)]: confidence scores corresponding to each input label

    """

    def finalize(self):  # overrides
        self._model.to(self.device)
        self._contra_label_name = "contradiction"
        self._entail_label_name = "entailment"

    @property
    def _templates(self) -> Dict[str, str]:
        return {
            "ko": "이 문장은 {label}에 관한 것이다.",
            "ja": "この文は、{label}に関するものである。",
            "zh": "这句话是关于{label}的。",
            "en": "This sentence is about {label}.",
        }

    @property
    def contra_label_name(self):
        return self._contra_label_name

    @contra_label_name.setter
    def contra_label_name(self, name: str):
        self._contra_label_name = name

    @property
    def entail_label_name(self):
        return self._entail_label_name

    @entail_label_name.setter
    def entail_label_name(self, name: str):
        self._entail_label_name = name

    @property
    def not_neutral_label_ids(self) -> List[str]:
        label2id = self._model.config.label2id
        contradiction_id = label2id.get(self.contra_label_name, None)
        entailment_id = label2id.get(self.entail_label_name, None)
        assert contradiction_id is not None and entailment_id is not None
        return [contradiction_id, entailment_id]

    def __call__(
        self,
        sentences: Union[List[str], str],
        labels: List[str],
        add_special_tokens: bool = True,
        batch_size: int = 32,
    ) -> List[Dict[str, float]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        n_samples = len(sentences)

        cands = [self._templates[self.lang].format(label=label) for label in labels]

        # all_probs.shape == (n_samples, n_labels)
        all_probs = np.array([], dtype=np.float64).reshape(n_samples, 0)
        for cand in cands:
            probs = np.array([], dtype=np.float64)
            for i in range(n_samples // batch_size + 1):
                sents = sentences[i * batch_size : (i + 1) * batch_size]  # noqa
                inputs = self._tokenizer(
                    sents,
                    [cand] * len(sents),
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens,
                )
                inputs = self._prepare_inputs(inputs)
                # Throw away "neutral"
                logits = self._model(**inputs).logits
                preds = logits[:, self.not_neutral_label_ids]
                # Take the probability of "entailment" as the probability of the label being true
                probs = np.hstack(
                    [probs, preds.softmax(dim=-1)[:, 1].detach().cpu().numpy()]
                )
            all_probs = np.hstack([all_probs, probs.reshape(-1, 1)])

        all_probs = (all_probs * 100).round(2)

        results = [dict(zip(labels, prob)) for prob in all_probs.tolist()]

        if n_samples == 1:
            results = results[0]

        return results

    def __repr__(self):
        _repr = super().__repr__().split("\n")
        category = f"[CATEGORY]: {self.__class__.__mro__[2].__name__}"
        _repr[1] = category
        return "\n".join(_repr)
