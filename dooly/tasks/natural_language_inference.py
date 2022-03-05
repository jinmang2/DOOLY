from typing import List, Dict, Tuple, Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, TextClassification
from ..tokenizers import Tokenizer as _Tokenizer

Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]


class NaturalLanguageInference(TextClassification):
    """
    Conduct Natural Language Inference

    English (`roberta.base.en.nli`)
        - dataset: MNLI (Adina Williams et al. 2017)
        - metric: Accuracy (87.6)

    Korean (`brainbert.base.ko.kornli`)
        - dataset: KorNLI (Ham et al. 2020)
        - metric: Accuracy (82.75)

    Japanese (`jaberta.base.ja.nli`)
        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: Accuracy (85.27)

    Chinese (`zhberta.base.zh.nli`)
        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: Accuracy (84.25)

    Args:
        sent_a: (str) first sentence to be encoded
        sent_b: (str) second sentence to be encoded

    Returns:
        str: predicted NLI label - Neutral, Entailment, or Contradiction

    """
    task: str = "nli"
    available_langs: List[str] = ["ko", "en", "ja", "zh"]
    available_models: Dict[str, List[str]] = {
        "ko": ["brainbert.base"],
        "en": ["roberta.base"],
        "ja": ["jaberta.base"],
        "zh": ["zhberta.base"],
    }
    misc_files: Dict[str, List[str]] = {}

    def __init__(
        self,
        config: DoolyTaskConfig,
        tokenizer: Tokenizer,
        model: PreTrainedModel,
    ):
        super().__init__(config=config)
        self._tokenizer = tokenizer
        self._model = model
        self.finalize()

    def __call__(
        self,
        sentences1: Union[List[str], str],
        sentences2: Union[List[str], str],
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ):
        assert type(sentences1) == type(sentences2)

        if isinstance(sentences1, str):
            sentences1 = [sentences1]
            sentences2 = [sentences2]
        else:
            assert len(sentences1) == len(sentences2)

        labels = self.predict_outputs(
            sentences1=sentences1,
            sentences2=sentences2,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        if len(sentences1) == 1:
            labels = labels[0]

        return labels
