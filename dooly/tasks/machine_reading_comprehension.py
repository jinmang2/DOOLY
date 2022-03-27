from typing import List, Dict, Tuple, Union, Callable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, SequenceTagging
from ..tokenizers import Tokenizer as _Tokenizer

Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]


class MachineReadingComprehension(SequenceTagging):
    """
    Conduct machine reading comprehension with query and its corresponding context

    Korean (`brainbert.base.ko.korquad`)
        - dataset: KorQuAD 1.0 (Lim et al. 2019)
        - metric: EM (84.33), F1 (93.31)

    Args:
        query: (str) query string used as query
        context: (str) context string used as context
        postprocess: (bool) whether to apply mecab based postprocess

    Returns:
        Tuple[str, Tuple[int, int]]: predicted answer span and its indices

    """
    task: str = "mrc"
    available_langs: List[str] = ["ko"]
    available_models: Dict[str, List[str]] = {
        "ko": ["brainbert.base"],
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

        try:
            import os
            if os.name == "nt":
                from eunjeon import Mecab
            else:
                from mecab import Mecab
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install python-mecab-ko with: `pip install python-mecab-ko`. "
                "If you are a windows user, install the eunjeon library with: `pip install eunjeon`."
            )

        self._tagger = Mecab()
        self._doc_stride = 128

        self.finalize()

    @property
    def doc_stride(self):
        return self._doc_stride

    @doc_stride.setter
    def doc_stride(self, val: int):
        self._doc_stride = val

    def get_inputs(
        self,
        text: Union[List[str], str],
        text_pair: Union[List[str], str] = None,
        src_lang: Union[List[str], str] = None,
        tgt_lang: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = True,
        **kwargs
    ): # overrides
        params = dict(
            # return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if not issubclass(self.tokenizer.__class__, PreTrainedTokenizerBase):
        #     params.update(dict(no_separator=no_separator))
        #     inputs = self.tokenizer(text, text_pair, **params)
        #     # Only support text: List[str]
        #     for k, v in inputs.items():
        #         inputs[k] = v[:, :self.model.config.max_position_embeddings]
        #     inputs["example_id"] = list(range(len(text)) if isinstance(text, list) else 1)
        # else:
            raise AttributeError
        pad_on_right = self.tokenizer.padding_side == "right"
        params.update(dict(
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.model.config.max_position_embeddings-2,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        ))
        inputs = self.tokenizer(text, text_pair, **params)
        # Since one example might give us several features if it has a long context,
        # we need a map from a feature to its corresponding example.
        # This key gives us just that.
        sample_mapping = inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context,
        # so we keep the corresponding example_id and we will store the offset mappings.
        inputs["example_id"] = []

        for i, input_ids in enumerate(inputs["input_ids"]):
            # Find the CLS token in the input_ids
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question)
            sequence_ids = inputs.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans,
            # this is the index of the example containing this span of text
            inputs["example_id"].append(sample_mapping[i])

            # Set to None the offset_mapping that are note part of the context
            # so it's easy to determine if a token position is part of the context or note
            inputs["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(inputs["offset_mapping"][i])
            ]

        return inputs

    def __call__(
        self,
        query: Union[List[str], str],
        context: Union[List[str], str],
        postprocess: Union[bool, Callable] = True,
        n_best_size: int = 1,
        null_score_diff_threshold: float = 0.0,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        return_logits: bool = False,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        postprocess_fn = None
        if callable(postprocess):
            postprocess_fn = postprocess
        elif postprocess:
            postprocess_fn = self._postprocess

        # single query -> single context
        if type(query) == type(context) and isinstance(query, str):
            query = [query]
            context = [context]
        # single query -> multiple context
        if isinstance(query, str):
            query = [query] * len(context)
        # multiple query -> single context
        if isinstance(context, str):
            context = [context] * len(query)

        _, all_nbest, _ = self.predict_span(
            question=query,
            context=context,
            n_best_size=n_best_size,
            null_score_diff_threshold=null_score_diff_threshold,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
            batch_size=batch_size,
            verbose=verbose,
        )

        if postprocess_fn is not None:
            for nbest in all_nbest:
                for pred in nbest:
                    pred["text"] = postprocess_fn(self._tagger, pred["text"])

        if not return_logits:
            all_nbest = [
                [(pred["text"], pred["offsets"]) for pred in nbest]
                for nbest in all_nbest
            ]

        if n_best_size == 1:
            all_nbest = [nbest[0] for nbest in all_nbest]

        if len(all_nbest) == 1:
            all_nbest = all_nbest[0]

        return all_nbest

    @staticmethod
    def _postprocess(tagger, text: str) -> str:
        assert hasattr(tagger, "pos")
        # First, strip punctuations
        text = text.strip("""!"\#$&'()*+,\-./:;<=>?@\^_‘{|}~《》""")

        # Complete imbalanced parentheses pair
        if text.count("(") == text.count(")") + 1:
            text += ")"
        elif text.count("(") + 1 == text.count(")"):
            text = "(" + text

        # Preserve beginning tokens since we only want to extract noun phrase of the last eojeol
        noun_phrase = " ".join(text.rsplit(" ", 1)[:-1])
        tokens = text.split(" ")
        eojeols = list()
        for token in tokens:
            eojeols.append(tagger.pos(token))
        last_eojeol = eojeols[-1]

        # Iterate backwardly to remove unnecessary postfixes
        i = 0
        for i, token in enumerate(last_eojeol[::-1]):
            _, pos = token
            # 1. The loop breaks when you meet a noun
            # 2. The loop also breaks when you meet a XSN (e.g. 8/SN+일/NNB LG/SL 전/XSN)
            if (pos[0] in ("N", "S")) or pos.startswith("XSN"):
                break
        idx = len(last_eojeol) - i

        # Extract noun span from last eojeol and postpend it to beginning tokens
        ext_last_eojeol = "".join(morph for morph, _ in last_eojeol[:idx])
        noun_phrase += " " + ext_last_eojeol
        return noun_phrase.strip()
