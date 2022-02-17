from typing import List, Dict
from collections import namedtuple

from .base import DoolyTaskBase
from ..models import DoolyModel
from ..tokenizers import DoolyTokenizer


class WordSenseDisambiguation(DoolyTaskBase):
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

    def __init__(
        self,
        lang: str,
        n_model: str,
        tokenizer: DoolyTokenizer,
        model: DoolyModel,
        morph2idx: Dict[str, int],
        tag2idx: Dict[str, int],
        query2origin: Dict[str, str],
        query2meaning: Dict[str, str],
        query2eng: Dict[str, str],
    ):
        super().__init__(lang=lang, n_model=n_model)
        self._tokenizer = tokenizer
        self._model = model
        self._cands = ["NNG", "NNB", "NNBC", "VV", "VA", "MM", "MAG", "NP", "NNP"]
        self._morph2idx = morph2idx # morpheme to index
        self._tag2idx = tag2idx # tag to index
        self._query2origin = query2origin # query to origin
        self._query2meaning = query2meaning # query to meaning
        self._query2eng = query2eng # query to english
        self._Wdetail = namedtuple(
            typename="detail",
            field_names="morph pos sense_id original meaning english",
        )

    @classmethod
    def build(
        cls,
        lang: str = None,
        n_model: str = None,
        **kwargs
    ):
        lang, n_model = cls._check_validate_input(lang, n_model)

        dl_kwargs, tok_kwargs, model_kwargs = cls._parse_build_kwargs(
            DoolyTokenizer, **kwargs)

        # set tokenizer
        tokenizer = DoolyTokenizer.from_pretrained(
            cls.task, lang, n_model, **dl_kwargs, **tok_kwargs)
        # set model
        model = DoolyModel.from_pretrained(
            cls.task, lang, n_model, **dl_kwargs, **model_kwargs)
        # set misc
        misc_files = ["morph2idx.ko.pkl", "tag2idx.ko.pkl", "wsd-dicts.ko.pkl"]
        misc = cls._build_misc(lang, n_model, misc_files, **dl_kwargs)
        morph2idx = misc[0]
        tag2idx = misc[1]
        query2origin, query2meaning, query2eng, _ = misc[2]

        return cls(lang, n_model, tokenizer, model, morph2idx, tag2idx,
                   query2origin, query2meaning, query2eng)

    def _postprocess(self, output):
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
                    wdetail = self._Wdetail(morph, tag, None, None, None, None)
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

                wdetail = self._Wdetail(morph, tag, sense, origin, meaning, eng)
                result.append(wdetail)

            if i != len(eojeols) - 1:
                wdetail = self._Wdetail("▁", "SPACE", None, None, None, None)
                result.append(wdetail)

        return result
