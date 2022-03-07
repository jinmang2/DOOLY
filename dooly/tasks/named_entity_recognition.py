import re
from typing import List, Dict, Tuple, Union
from collections import defaultdict

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, SequenceTagging
from ..tokenizers import Tokenizer as _Tokenizer


SentenceWithTags = List[Tuple[str, str]]
Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]


class NamedEntityRecognition(SequenceTagging):
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
    misc_files: Dict[str, List[str]] = {
        "ko": ["wiki.ko.items"]
    }

    def __init__(
        self,
        config: DoolyTaskConfig,
        tokenizer: Tokenizer,
        model: PreTrainedModel,
    ):
        super().__init__(config=config)

        use_sentence_tokenizer = "charbert" in n_model
        if use_sentence_tokenizer:
            tokenizer._set_sent_tokenizer()

        self._tokenizer = tokenizer
        self._model = model
        self._wsd_dict = config.misc_tuple[0] # wiki.ko.items

        self._wsd = None
        self._cls2cat = None
        self._quant2cat = None
        self._term2cat = None
        self.finalize()

    def __call__(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True, # ENBERTa, JaBERTa, ZhBERTa에선 없음
        do_sent_split: bool = True,
        ja_zh_split_force: bool = False, # deprecated
        ignore_labels: List[int] = [],
        apply_wsd: bool = False,
    ):
        if apply_wsd and self.lang != "ko":
            apply_wsd = False

        if do_sent_split and self.lang == "en":
            if not hasattr(self._tokenizer, "en_sent_tokenizer"):
                do_sent_split = False
        elif self.lang in ["ja", "zh"] and not ja_zh_split_force:
            do_sent_split = False

        if isinstance(sentences, str):
            sentences = [sentences]

        # Sentence split
        if do_sent_split:
            sentences, n_sents = self._tokenizer.sent_tokenize(sentences)
            sentences = [sentence for sents in sentences for sentence in sents]
        else:
            n_sents = [1] * len(sentences)

        token_label_pairs = self.predict_tags(
            sentences=sentences,
            add_special_tokens=add_special_tokens,
        )

        # Post processing
        postprocessed = [self._postprocess(sentence) for sentence in token_label_pairs]

        if apply_wsd:
            postprocessed = self._apply_wsd(postprocessed)

        sents_with_tag = self._apply_dict(postprocessed)

        # Merge divided sentences into batches
        results = []
        ix = 0
        for n_sent in n_sents:
            result = []
            for _ in range(n_sent):
                result.extend(sents_with_tag[ix])
                result.extend([(" ", "O")])
                ix += 1
            results.append(result[:-1])

        if len(results) == 1:
            results = results[0]

        return results

    def _template_match(self, text, expression2category):
        """
        Apply template match using regular expression

        Args:
            text (str): text to be searched
            expression2category (dict): regular expression dict

        Returns:
            str: regex matched category

        """
        for expression, category in expression2category.items():
            if re.search(expression, text) is not None:
                return category

    def _apply_wsd(self, tags: List[SentenceWithTags]) -> List[SentenceWithTags]:
        """
        Apply Word Sense Disambiguation to get detail tag info

        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result

        Returns:
            List[Tuple[str, str]]: wsd-applied result

        """
        if self._wsd is None:
            from . import WordSenseDisambiguation
            self._wsd = WordSenseDisambiguation.build(lang="ko", n_model="transformer.large")

        if self._cls2cat is None:
            self._cls2cat = dict()
            lines = self._build_misc(self.lang, self.n_model, ["wsd.cls.txt"])[0]
            for line in lines:
                morph, homonymno, category = line.split()
                classifier = f"{morph}__NNB__{homonymno}"  # bound noun
                self._cls2cat[classifier] = category

        if self._quant2cat is None:
            self._quant2cat = dict()
            self._term2cat = dict()
            lines = self._build_misc(self.lang, self.n_model, ["re.templates.txt"])[0]
            for line in lines:
                category, ner_category, expression = line.split(" ", 2)
                if ner_category == "QUANTITY":
                    self._quant2cat[expression] = category
                elif ner_category == "TERM":
                    self._term2cat[expression] = category

        def convert_tags_to_input_text_with_markers(tags: SentenceWithTags):
            input_text_with_markers = str()
            target_token_ids = []

            for idx, ner_token in enumerate(tags):
                surface, tag = ner_token
                # as {} will be used as special symbols
                surface = surface.replace("{", "｛")
                surface = surface.replace("}", "｝")

                if tag == "TERM":
                    cat = self._template_match(surface, self._term2cat)
                    if cat is not None:
                        tags[idx] = (surface, cat)
                    input_text_with_markers += surface
                elif tag == "QUANTITY":
                    cat = self._template_match(surface, self._quant2cat)
                    if cat is not None:
                        tags[idx] = (surface, cat)
                        input_text_with_markers += surface
                    else:
                        target_token_ids.append(idx)
                        input_text_with_markers += "{" + surface + "}"
                else:
                    input_text_with_markers += surface

            return input_text_with_markers, target_token_ids

        inputs = []
        targets = []
        for sent_tags in tags:
            results = convert_tags_to_input_text_with_markers(sent_tags)
            inputs.append(results[0])
            targets.append(results[1])

        batch_results = self._wsd(inputs)

        def convert_wsd_results_to_tags(wsd_results, tags, target_token_ids):
            action = False
            has_category = False
            categories = []

            for wsd_token in wsd_results:
                morph, tag, homonymno = wsd_token[:3]
                if morph == "{":
                    has_category = False
                    action = True
                elif morph == "}":
                    if has_category is False:
                        categories.append("QUANTITY")  # original category
                    has_category = False
                    action = False

                if action:
                    if homonymno is None:
                        homonymno = "00"

                    query = f"{morph}__{tag}__{homonymno}"
                    if query in self._cls2cat:
                        category = self._cls2cat[query]
                        categories.append(category)
                        has_category = True
                        action = False

            assert len(target_token_ids) == len(categories)

            for target_token_id, cat in zip(target_token_ids, categories):
                tags[target_token_id] = (tags[target_token_id][0], cat)

            return tags

        outputs = []
        for wsd_results, _tags, target_token_ids in zip(batch_results, tags, targets):
            outputs.append(convert_wsd_results_to_tags(wsd_results, _tags, target_token_ids))

        return outputs

    def _apply_dict(self, tags: List[SentenceWithTags]) -> List[SentenceWithTags]:
        """
        Apply pre-defined dictionary to get detail tag info

        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result

        Returns:
            List[Tuple[str, str]]: dict-applied result

        """
        results = []
        for _tags in tags:
            result = []
            for pair in _tags:
                word, tag = pair
                if (tag in self._wsd_dict.keys()) and (word in self._wsd_dict[tag]):
                    result.append((word, self._wsd_dict[tag][word].upper()))
                else:
                    result.append(pair)
            results.append(result)
        return results

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

    @staticmethod
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
        elif self.lang == "ja":
            result.append((word.replace("##", ""), self._remove_head(tag)))
        else:
            result.append((word, self._remove_head(tag)))

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
                result.append((tmp_word, prev_tag))
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
                tmp_word = char

            prev_tag = tag
        result.append((tmp_word, prev_tag))

        result = [
            (pair[0].replace("▁", " ").strip(), pair[1])
            if pair[0] != " " else (" ", "O")
            for pair in result
        ]
        return result

    @classmethod
    def build_misc(
        cls,
        lang: str,
        n_model: str,
        misc_files: List[str],
        **dl_kwargs,
    ) -> Tuple: # overrides
        wsd_dict = {}
        if "charbert" in n_model:
            f_wsd_dict = cls._build_misc(lang, n_model, misc_files, **dl_kwargs)
            wsd_dict = defaultdict(dict)
            for line in f_wsd_dict:
                origin, target, word = line.strip().split("\t")
                wsd_dict[origin][word] = target
        return wsd_dict
