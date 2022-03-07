from typing import List, Dict, Tuple, Union, Optional
from collections import namedtuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, Seq2Seq
from ..tokenizers import Tokenizer as _Tokenizer


Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]
Wdetail = namedtuple(
    typename="detail",
    field_names="morph pos sense_id original meaning english",
)


class WordSenseDisambiguation(Seq2Seq):
    """
    Conduct Word Sense Disambiguation

    Korean (`transformer.large`)
        - dataset: https://corpus.korean.go.kr/ 어휘 의미 분석 말뭉치
        - metric: TBU

    Args:
        text (str): sentence to be inputted

    Returns:
        List[Tuple[str, str]]: list of token and its disambiguated meaning tuple

    """
    task: str = "wsd"
    available_langs: List[str] = ["ko"]
    available_models: Dict[str, List[str]] = {
        "ko": ["transformer.large"]
    }
    misc_files = {
        "ko": ["morph2idx.ko.pkl", "tag2idx.ko.pkl", "wsd-dicts.ko.pkl"]
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

        self._cands = ["NNG", "NNB", "NNBC", "VV", "VA", "MM", "MAG", "NP", "NNP"]
        self._morph2idx: Dict[str, int] = config.misc_tuple[0] # morpheme to index
        self._tag2idx: Dict[str, int] = config.misc_tuple[1] # tag to index

        query2origin, query2meaning, query2eng, _ = config.misc_tuple[2]
        self._query2origin: Dict[str, str] = query2origin # query to origin
        self._query2meaning: Dict[str, str] = query2meaning # query to meaning
        self._query2eng: Dict[str, str] = query2eng # query to english

        self.finalize()

    def __call__(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True,
        beams: int = 5,
        max_len_a: int = 4,
        max_len_b: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        generated = self.generate(
            sentences,
            add_special_tokens=add_special_tokens,
            beams=beams,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            **kwargs,
        )

        results = []
        for output in generated:
            decoded_text = self.tokenizer.decode(output, recover_original=False)
            results.append(self._postprocess(decoded_text))

        if len(results) == 1:
            results = results[0]

        return results

    def _postprocess(self, output: str) -> List[Wdetail]:
        eojeols = output.split("▁")

        result = []
        for i, eojeol in enumerate(eojeols):
            pairs = eojeol.split("++")

            for pair in pairs:
                morph, tag = pair.strip().split(" || ")

                if "__" in morph:
                    morph, sense = morph.split(" __ ")
                else:
                    sense = None

                morph = "".join([c for c in morph if c != " "])

                if tag not in self._cands:
                    wdetail = Wdetail(morph, tag, None, None, None, None)
                    result.append(wdetail)
                    continue

                sense = str(sense).zfill(2)

                query_morph = morph + "다" if tag[0] == "V" else morph
                query_sense = "" if sense is None else "__" + sense

                query = f"{query_morph}{query_sense}"
                origin = self._query2origin.get(query, None)
                meaning = self._query2meaning.get(query, None)
                eng = self._query2eng.get(query, None)

                if meaning is None:
                    query = f"{query_morph}__00"
                    meaning = self._query2meaning.get(query, None)
                    if meaning:
                        sense = "00"

                wdetail = Wdetail(morph, tag, sense, origin, meaning, eng)
                result.append(wdetail)

            if i != len(eojeols) - 1:
                wdetail = Wdetail("▁", "SPACE", None, None, None, None)
                result.append(wdetail)

        return result
