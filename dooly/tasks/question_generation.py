from typing import List, Dict, Tuple, Union, Callable, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, Seq2Seq
from ..tokenizers import Tokenizer as _Tokenizer

from .base.wikipedia2vec import (
    QueryParser,
    WhooshIndex,
    Wikipedia2Vec,
    SimilarWords,
)

Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]


class QuestionGeneration(Seq2Seq):
    """
    Question generation using BART model

    Korean (`kobart.base.ko.qg`)
        - dataset: KorQuAD 1.0 (Lim et al. 2019) + AI hub Reading Comprehension corpus + AI hub Commonsense corpus
        - metric: Model base evaluation using PororoMrc (`brainbert.base`)
            - EM (82.59), F1 (94.06)
        - ref: https://www.aihub.or.kr/aidata/86
        - ref: https://www.aihub.or.kr/aidata/84

    Args:
        answer (str): answer text
        context (str): source article
        beam (int): beam search size
        temperature (float): temperature scale
        top_k (int): top-K sampling vocabulary size
        top_p (float): top-p sampling ratio
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio
        n_wrong (int): number of wrong answer candidate
        return_context (bool): return context together or not

    Returns:
        str : question (if `n_wrong` < 1)
        Tuple[str, List[str]] : question, wrong_answers (if `n_wrong` > 1)

    """
    task: str = "qg"
    available_langs: List[str] = ["ko"]
    available_models: Dict[str, List[str]] = {
        "ko": ["kobart.base"],
    }
    misc_files: Dict[str, List[str]] = {
        "ko": ["kowiki_20200720_100d.pkl", "ko_indexdir.zip"]
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

        # set sentence tokenizer
        self._tokenizer._set_sent_tokenizer()

        self._sim_words = SimilarWords(
            model=Wikipedia2Vec(self.misc_tuple[0], self.device),
            idx=WhooshIndex.open_dir(self.misc_tuple[1]),
        )
        self.finalize()

    def __call__(
        self,
        answer: Union[List[str], str],
        context: Union[List[str], str],
        add_special_tokens: bool = True,
        do_sent_split: bool = True,
        beams: int = 5,
        max_len_a: int = 1,
        max_len_b: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        no_repeat_ngram_size: int = 4,
        length_penalty: float = 1.0,
        **kwargs,
    ):
        if isinstance(answer, str) and isinstance(context, str):
            context = self._focus_answer(context, answer)
        elif isinstance(answer, list) and isinstance(context, str):
            context = [self._focus_answer(context, a) for a in answer]
        elif isinstance(answer, str) and isinstance(context, list):
            context = [self._focus_answer(c, answer) for c in context]
        elif isinstance(answer, list) and isinstance(context, list):
            assert len(answer) == len(context), (
                "length of answer list and context list must be same."
            )
            context = [self._focus_answer(c, a) for c, a in zip(context, answer)]
        return None
