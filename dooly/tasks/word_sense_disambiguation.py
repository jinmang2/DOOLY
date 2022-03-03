from typing import List, Dict, Tuple, Union
from collections import namedtuple

from .base import Seq2Seq


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
        lang: str,
        n_model: str,
        device: str,
        tokenizer,
        model,
        misc: Tuple,
    ):
        super().__init__(lang=lang, n_model=n_model, device=device)
        self._tokenizer = tokenizer
        self._model = model.eval().to(device)
        self._cands = ["NNG", "NNB", "NNBC", "VV", "VA", "MM", "MAG", "NP", "NNP"]
        self._morph2idx: Dict[str, int] = misc[0] # morpheme to index
        self._tag2idx: Dict[str, int] = misc[1] # tag to index
        query2origin, query2meaning, query2eng, _ = misc[2]
        self._query2origin: Dict[str, str] = query2origin # query to origin
        self._query2meaning: Dict[str, str] = query2meaning # query to meaning
        self._query2eng: Dict[str, str] = query2eng # query to english
        self._Wdetail = namedtuple(
            typename="detail",
            field_names="morph pos sense_id original meaning english",
        )

    def __call__(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True,
        no_separator: bool = False,
        beams: int = 5,
        max_len_a: int = 4,
        max_len_b: int = 50,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self._tokenizer(
            sentences,
            return_tensors=True,
            no_separator=no_separator,
            add_special_tokens=add_special_tokens,
        )
        input_ids = self._prepare_inputs(inputs)["input_ids"]

        # beam search
        # @TODO: fix miss-match fairseq vs transformers
        outputs = self._model.generate(
            input_ids,
            num_beams=beams,
            max_length=input_ids.shape[-1] * max_len_a + max_len_b,
        )

        result = []
        for output in outputs:
            decoded_text = self._tokenizer.decode(output)
            result.append(self._postprocess(decoded_text))

        return result

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
