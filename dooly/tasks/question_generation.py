from typing import List, Dict, Union, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, Seq2Seq
from ..tokenizers import Tokenizer as _Tokenizer

from .base.wikipedia2vec import (
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

        self._max_length = model.config.max_position_embeddings or 1024

        # set sentence tokenizer
        self._tokenizer._set_sent_tokenizer()

        self._start_hl_token = "<unused0>"
        self._end_hl_token = "<unused1>"

        self._sim_words = SimilarWords(
            model=Wikipedia2Vec(config.misc_tuple[0], self.device),
            idx=WhooshIndex.open_dir(config.misc_tuple[1]),
        )
        self.finalize()

    @property
    def start_hl_token(self):
        """ Get start highlight token """
        return self._start_hl_token

    @start_hl_token.setter
    def start_hl_token(self, val):
        """ Set start highlight token """
        self._start_hl_token = val

    @property
    def end_hl_token(self):
        """ Get end highlight token """
        return self._end_hl_token

    @end_hl_token.setter
    def end_hl_token(self, val):
        """ Set end highlight token """
        self._end_hl_token = val

    @property
    def max_length(self):
        return self._max_length

    def _focus_answer(self, context: str, answer: str, truncate: bool = True):
        """
        add answer start and end token
        and truncate context text to make inference speed fast

        Args:
            context (str): context string
            answer (str): answer string
            truncate (bool): truncate or not

        Returns:
            context (str): preprocessed context string

        """

        start_idx = context.find(answer)
        end_idx = start_idx + len(answer) + len(self.start_hl_token)
        # insert highlight tokens
        context = context[:start_idx] + self.start_hl_token + context[start_idx:]
        context = context[:end_idx] + self.end_hl_token + context[end_idx:]

        if len(context) < self.max_length or not truncate:
            return context

        sentences = self.tokenizer.sent_tokenize(context)
        answer_sent_idx = None
        for i in range(len(sentences)):
            if self.start_hl_token in sentences[i]:
                answer_sent_idx = i
                break

        i, j = answer_sent_idx, answer_sent_idx
        truncated_context = [sentences[answer_sent_idx]]

        while len(" ".join(truncated_context)) < self.max_length:
            prev_context_length = len(" ".join(truncated_context))
            i -= 1
            j += 1

            if i > 0:
                truncated_context = [sentences[i]] + truncated_context
            if j < len(sentences):
                truncated_context = truncated_context + [sentences[j]]
            if len(" ".join(truncated_context)) == prev_context_length:
                break

        truncated_context = " ".join(truncated_context)
        if len(truncated_context) > self.max_length:
            if start_idx < len(context) // 2:
                truncated_context = truncated_context[: self.max_length]
            else:
                truncated_context = truncated_context[
                    len(truncated_context) - self.max_length :
                ]

        return truncated_context

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
        return_context: bool = False,
        batch_size: int = 32,
        verbose: bool = True,
        n_wrong: int = 0,
        **kwargs,
    ):
        assert isinstance(n_wrong, int)

        if isinstance(answer, str) and isinstance(context, str):
            context = [self._focus_answer(context, answer)]
        elif isinstance(answer, list) and isinstance(context, str):
            context = [self._focus_answer(context, a) for a in answer]
        elif isinstance(answer, str) and isinstance(context, list):
            context = [self._focus_answer(c, answer) for c in context]
        elif isinstance(answer, list) and isinstance(context, list):
            assert len(answer) == len(
                context
            ), "length of answer list and context list must be same."
            context = [self._focus_answer(c, a) for c, a in zip(context, answer)]

        generated = self.generate(
            text=context,
            add_special_tokens=add_special_tokens,
            beams=beams,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

        if issubclass(self.tokenizer.__class__, PreTrainedTokenizerBase):
            decoded_text = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        else:
            decoded_text = self.tokenizer.decode(generated)

        output = [self._postprocess(text) for text in decoded_text]

        if n_wrong > 0:
            if isinstance(context, str) and isinstance(answer, str):
                wrong_answers = self._sim_words._extract_wrongs(answer)
                output = output, wrong_answers[:n_wrong]

            elif isinstance(context, list) and isinstance(answer, str):
                wrong_answers = self._sim_words._extract_wrongs(answer)[:n_wrong]
                output = [(o, wrong_answers) for o in output]

            elif isinstance(context, str) and isinstance(answer, list):
                wrong_answers = [
                    self._sim_words._extract_wrongs(a)[:n_wrong] for a in answer
                ]
                output = [(output, w) for w in wrong_answers]

            else:
                wrong_answers = [
                    self._sim_words._extract_wrongs(a)[:n_wrong] for a in answer
                ]
                output = [(o, w) for o, w in zip(output, wrong_answers)]

        return output

    def _postprocess(self, text: str):
        text = text.strip()
        if not text.endswith("?"):
            text += "?"
        return text
