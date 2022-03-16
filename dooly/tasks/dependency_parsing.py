from typing import List, Dict, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, SequenceTagging
from ..tokenizers import Tokenizer as _Tokenizer


Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]


class DependencyParsing(SequenceTagging):
    """
    Conduct dependency parsing

    Korean (`posbert.base.ko.dp`)
        - dataset: https://corpus.korean.go.kr/ 구문 분석 말뭉치
        - metric: UAS (90.57), LAS (95.96)

    Args:
        sent: (str) sentence to be parsed dependency

    Returns:
        List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

    """
    task: str = "dp"
    available_langs: List[str] = ["ko"]
    available_models: Dict[str, List[str]] = {
        "ko": ["posbert.base"],
    }
    misc_files: Dict[str, List[str]] = {
        "ko": ["label0.json", "label1.json"],
    }

    def __init__(
        self,
        config: DoolyTaskConfig,
        tokenizer: Tokenizer,
        model: PreTrainedModel,
    ):
        super().__init__(config=config)
        self._tokenizer = tokenizer
        self._model = model

        self._label0 = {i: s for s, i in config.misc_tuple[0].items()}
        self._label1 = {i: s for s, i in config.misc_tuple[1].items()}

        self.finalize()

    def __call__(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        tokens, heads, labels, sent_lengths = self.predict_dependency(
            sentences=sentences,
            add_special_tokens=add_special_tokens,
        )

        results = []
        iterator = zip(sentences, tokens, heads, labels, sent_lengths)
        for sentence, token, head, label, sent_length in iterator:
            head = [int(h)-1 for h in head][:sent_length] # due to default <s> token
            label = label[:sent_length]
            results.append(self._postprocess(sentence, token, head, label))

        return results

    def _postprocess(
        self,
        ori: str,
        tokens: List[str],
        heads: List[int],
        labels: List[int],
    ) -> List:
        """
        Postprocess dependency parsing output

        Args:
            ori (sent): original sentence
            heads (List[str]): dependency heads generated by model
            labels (List[str]): tag labels generated by model

        Returns:
            List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

        """
        eojeols = ori.split()

        indices = [i for i, token in enumerate(tokens) if token == "▃"]
        real_heads = [head for i, head in enumerate(heads) if i in indices]
        real_labels = [label for i, label in enumerate(labels) if i in indices]

        result = []
        curr = 0
        for head, label, eojeol in zip(real_heads, real_labels, eojeols):
            curr += 1

            try:
                head_eojeol = indices.index(head) + 1
            except:
                head_eojeol = -1

            if head_eojeol == curr:
                head_eojeol = -1

            result.append((curr, eojeol, head_eojeol, label))

        return result