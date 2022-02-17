from typing import List, Dict, Tuple, Union
from collections import defaultdict

from .word_sense_disambiguation import WordSenseDisambiguation

from .base import DoolyTaskBase
from ..models import DoolyModel
from ..tokenizers import DoolyTokenizer


class NamedEntityRecognition(DoolyTaskBase):
    """
    Conduct named entity recognition

    Korean (`charbert.base`)
        - dataset: https://corpus.korean.go.kr/ 개체명 분석 말뭉치
        - metric: F1 (89.63)

    English (`roberta.base`)
        - dataset: OntoNotes 5.0
        - metric: F1 (91.63)

    Japanese (`jaberta.base`)
        - dataset: Kyoto University Web Document Leads Corpus
        - metric: F1 (76.74)
        - ref: https://github.com/ku-nlp/KWDLC

    Chinese (`zhberta.base`)
        - dataset: OntoNotes 5.0
        - metric: F1 (79.06)

    Args:
        sent: (str) sentence to be sequence labeled

    Returns:
        List[Tuple[str, str]]: token and its predicted tag tuple list

    """
    task: str = "ner"
    available_langs: List[str] = ["ko", "en", "ja", "zh"]
    available_models: Dict[str, List[str]] = {
        "ko": ["charbert.base"],
        "en": ["roberta.base"],
        "ja": ["jaberta.base"],
        "zh": ["zhberta.base"],
    }

    def __init__(
        self,
        lang: str,
        n_model: str,
        tokenizer: DoolyTokenizer,
        model: DoolyModel,
        wsd_dict: Dict = {},
    ):
        super().__init__(lang=lang, n_model=n_model)
        self._tokenizer = tokenizer
        self._model = model
        self._wsd_dict = wsd_dict

        self._wsd = None
        self._cls2cat = None
        self._quant2cat = None
        self._term2cat = None

    def __call__(
        self,
        sentence: Union[List[str], str],
        add_special_tokens: bool = True, # ENBERTa, JaBERTa, ZhBERTa에선 없음
        no_separator: bool = False,
        do_sent_split: bool = True,
        ignore_labels: List[int] = [],
        apply_wsd: bool = False,
    ):
        if apply_wsd and self.lang != "ko":
            apply_wsd = False

        if instance(sentence, str):
            sentence = [sentence]

        sentences = sentence

        # Sentence split
        if do_sent_split:
            texts, n_sents = self._tokenizer.sent_tokenize(sentences)
            texts = [sentence for sentences in texts for sentence in sentences]
        else:
            n_sents = [1] * len(sentences)

        token_label_pairs = self.predict_tags(
            sentence=sentences,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
            do_sent_split=do_sent_split,
        )

        # Post processing
        results = []
        ix = 0
        for n_sent in n_sents:
            result = []
            for _ in range(n_sent):
                res = []
                sentence = token_label_pairs[ix]
                for pair in self._postprocess(sentence):
                    if pair[1] not in ignore_labels:
                        if apply_wsd:
                            pair = self._apply_wsd(pair)
                        res.append(pair)
                result.extend(self._apply_dict(res))
                result.extend([(" ", "O")])
                ix += 1
            results.append(result[:-1])
        return results

    def _apply_dict(self, tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Apply pre-defined dictionary to get detail tag info
        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result
        Returns:
            List[Tuple[str, str]]: dict-applied result
        """
        result = []
        for pair in tags:
            word, tag = pair
            if (tag in self._wsd_dict.keys()) and (word in self._wsd_dict[tag]):
                result.append((word, self._wsd_dict[tag][word].upper()))
            else:
                result.append(pair)
        return result

    def _postprocess(self, tags: List[Tuple[str, str]]):
        """
        Postprocess characted tags to concatenate BIO
        Args:
            tags (List[Tuple[str, str]]): characted token and its corresponding tag tuple list
        Returns:
            List(Tuple[str, str]): postprocessed entity token and its corresponding tag tuple list
        """
        if self.lang == "ko":
            result = self._postprocess_char(tags)
        else:
            result = self._postprocess_bpe(tags)
        return result

    def _remove_head(tag: str) -> str:
        if "-" in tag:
            tag = tag[2:]
        return tag

    def _postprocess_bpe(self, tags: List[Tuple[str, str]]):
        result = list()

        word = tags[0][0]
        tag = tags[0][1]
        for pair in tags[1:]:
            token, label = pair

            if "I" in label:
                word += token
            else:
                word = word.strip()
                if self.lang == "ja":
                    # Since `cl-tohoku/bert-base-japanese-whole-word-masking` model use
                    # metaspace "##", replace metaspace to empty string.
                    word = word.replace("##", "")
                if self.lang == "en" and word.endswith("."):
                    # In the case of English, The end mark(".") is treated as O-tag.
                    result.append((word[:-1], self._remove_head(tag)))
                    result.append((".", "O"))
                else:
                    result.append((word, self._remove_head(tag)))
                word = token
                tag = label

        word = word.strip()
        if self.lang == "en" and word.endswith("."):
            # In the case of English, The end mark(".") is treated as O-tag.
            result.append((word[:-1], self._remove_head(tag)))
            result.append((".", "O"))
        else:
            result.append((word, self.predict_srl_remove_head(tag)))

        return [pair for pair in result if pair[0]]

    def _postprocess_char(self, tags: List[Tuple[str, str]]):
        result = list()

        tmp_word = tags[0][0]
        prev_ori_tag = tags[0][1]
        prev_tag = self._remove_head(prev_ori_tag)
        for pair in tags[1:]:
            char, ori_tag = pair
            tag = self._remove_head(ori_tag)

            if ("▁" in char) and ("I-" not in ori_tag):
                result.append((tmp_word, prev_orig))
                result.append((" ", "O"))

                tmp_word = char
                prev_tag = tag
                continue

            if (tag == prev_tag) and (("I-" in ori_tag) or ("O" in ori_tag)):
                tmp_word += char
            elif (tag != prev_tag) and ("I-" in ori_tag) and (tag != "O"):
                tmp_word += char
            else:
                result.append((tmp_word, prev_tag))
                tmp_word += char

            prev_tag = tag
        result.append((tmp_word, prev_tag))

        result = [
            (pair[0].replace("▁", " ").strip(), pair[1])
            if pair[0] != " " else (" ", "O")
            for pair in result
        ]
        return result

    @classmethod
    def build(
        cls,
        lang: str = None,
        n_model: str = None,
        **kwargs
    ):
        lang, n_model = cls._check_validate_input(lang, n_model)

        use_sentence_tokenizer = use_wsd = False
        if "charbert" in n_model:
            use_sentence_tokenizer = use_wsd = True

        dl_kwargs, tok_kwargs, model_kwargs = cls._parse_build_kwargs(
            DoolyTokenizer, **kwargs)

        # set tokenizer
        tokenizer = DoolyTokenizer.from_pretrained(
            cls.task, lang, n_model, **dl_kwargs, **tok_kwargs)
        if use_sentence_tokenizer:
            tokenizer._set_sent_tokenizer()
        # set model
        model = DoolyModel.from_pretrained(
            cls.task, lang, n_model, **dl_kwargs, **model_kwargs)
        # set misc
        wsd_dict = {}
        if use_wsd:
            misc_files = ["wiki.ko.items"]
            f_wsd_dict = cls._build_misc(lang, n_model, misc_files, **dl_kwargs)
            wsd_dict = defaultdict(dict)
            for line in f_wsd_dict:
                origin, target, word = line.strip().split("\t")
                wsd_dict[origin][word] = target

        return cls(lang, n_model, tokenizer, model, wsd_dict)
