import os
import re
import abc
import json
import inspect
from typing import List, Tuple, Union, Dict, Callable, Any

from transformers import BatchEncoding
from transformers.tokenization_utils_base import TruncationStrategy

from .base import DoolyPreTrainedTokenizer
from ..utils.import_utils import (
    is_available_mecab,
    is_available_ipadic,
    is_available_fugashi,
    is_available_jieba,
    is_available_nltk,
)


PosTagResult = Union[Tuple[str, str], str]


class PosTagger:
    @property
    def tagger(self):
        return self._tagger

    @abc.abstractmethod
    def pos(self, sent: str, **kwargs) -> PosTagResult:
        pass


class MecabKoPosTagger(PosTagger):
    def __init__(self):
        if is_available_mecab():
            if os.name == "nt":
                try:
                    from mecab import Mecab
                except ImportError:
                    from eunjeon import Mecab
            else:
                from mecab import Mecab
        else:
            raise ModuleNotFoundError(
                "Please install python-mecab-ko with: `pip install python-mecab-ko`. "
                "If you are a windows user, install the eunjeon library with: `pip install eunjeon`."
            )

        self._tagger = Mecab()

    def pos(self, sent: str, **kwargs) -> PosTagResult:
        return_surface = kwargs.get("return_surface", False)
        return_string = kwargs.get("return_string", False)

        sent = sent.strip()
        sent_ptr = 0
        results = []

        if return_surface:
            analyzed = self.tagger.pos(sent)
        else:
            if hasattr(self.tagger, "parse"):
                analyzed = self.tagger.parse(sent)
            else:
                # deprecated! (eunjeon + parse)
                return_surface = True
                analyzed = self.tagger.pos(sent)

        for unit in analyzed:
            if not return_surface:
                morph, token = self._postprocess(unit)
            else:
                token = unit
                morph = unit[0]
            if sent[sent_ptr] == " ":
                # Move sent pointer to whitespace token to reserve whitespace
                # cf. to prevent double white-space, we move pointer to next eojeol
                while sent[sent_ptr] == " ":
                    sent_ptr += 1
                results.append((" ", "SPACE"))
            if isinstance(token, tuple):
                results.append(token)
            elif isinstance(token, list):
                results.extend(token)
            sent_ptr += len(morph)

        if return_string:
            return self.string(results)

        return results

    def string(self, result: List[Tuple[str, str]]) -> str:
        res_str = ""
        for pair in result:
            if pair[1] == "SPACE":
                res_str = res_str[:-1]
                res_str += " "
            else:
                res_str += f"{pair[0]}/{pair[1]}+"
        return res_str

    def _postprocess(self, unit: str) -> Tuple[str, str]:
        # Should split line with tap since comma is frequently used in input sentence
        morph = unit[0]
        features = unit[1]
        pos = features.pos
        analysis = features.expression

        if analysis and ("+" in analysis):
            if "*" in analysis:
                token = [morph.rsplit("/", 1)[0] for morph in analysis.split("+")]
                token = [(t.split("/")[0], t.split("/")[1]) for t in token]
            else:
                analysis = analysis.replace("+/", "[PLUS]/")
                analysis = analysis.replace("+", "[SEP]")
                analysis = analysis.replace("[PLUS]", "+")
                token = [
                    (pair.split("/")[0], pair.split("/")[1])
                    for pair in analysis.split("[SEP]")
                ]
        else:
            token = (morph, pos)

        return morph, token


class NltkEnPostTagger(PosTagger):
    def __init__(self):
        if is_available_nltk():
            import nltk

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger")

        else:
            raise ModuleNotFoundError("Please install fugashi with: `pip install nltk`")

        self._tagger = nltk

    def pos(self, sent: str, **kwargs) -> PosTagResult:
        sent = self._clean(sent)
        words = self.tagger.word_tokenize(sent)
        pos_tags = self.tagger.pos_tag(words)
        return self._align(sent, pos_tags)

    def _clean(self, sent: str) -> str:
        sent = sent.strip()
        sent = re.sub("\s", " ", sent)  # noqa
        sent = re.sub(" +", " ", sent)
        return sent

    def _align(self, sent: str, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        result = list()

        while True:
            token = tokens.pop(0)
            word = token[0]

            # correct strange behaviors of nltk 'word_tokenize'
            # https://github.com/nltk/nltk/issues/1630
            if (word in ("``", "''")) and (sent[0] == '"'):
                word = '"'
                token = ('"', '"')
            if (word == "...") and (sent[0] == "…"):  # ellipsis
                word = "…"
                token = ("…", "…")

            if sent.startswith(f"{word} "):
                sent = sent[len(f"{word} ") :]  # noqa
                result.append(token)
                result.append((" ", "SPACE"))
            elif sent.startswith(word):
                sent = sent[len(word) :]  # noqa
                result.append(token)
            else:
                raise ValueError(f"Can't align the {token} to {sent}")

            if not tokens:
                break

        return result


class MecabJaPosTagger(PosTagger):
    def __init__(self):
        if is_available_fugashi():
            import fugashi
        else:
            raise ModuleNotFoundError(
                "Please install fugashi with: `pip install fugashi`"
            )

        if is_available_ipadic():
            import ipadic
        else:
            raise ModuleNotFoundError(
                "Please install ipadic with: `pip install ipadic`"
            )

        dic_dir = ipadic.DICDIR
        mecabrc = os.path.join(dic_dir, "mecabrc")
        mecab_option = f"-d {dic_dir} -r {mecabrc} "
        self._tagger = fugashi.GenericTagger(mecab_option)

    def pos(self, sent: str, **kwargs) -> PosTagResult:
        mecab_output = self.tagger.parse(sent)

        pairs = list()
        for line in mecab_output.split("\n"):
            if line == "EOS":
                break
            token, tag = line.split("\t")
            tags = tag.split(",")
            pairs.append((token, tags[0]))

        return pairs


class JiebaZhPosTagger(PosTagger):
    def __init__(self):
        if is_available_jieba():
            import jieba
            import jieba.posseq
        else:
            raise ModuleNotFoundError("Please install jieba with: `pip install jieba`")

        self._tagger = jieba.posseq

    def pos(self, sent: str, **kwargs) -> PosTagResult:
        jieba_output = self.tagger.cut(sent)
        return [(word.word, word.flat) for word in list(jieba_output)]


PosTaggerMap = {
    "ko": MecabKoPosTagger,
    "en": NltkEnPostTagger,
    "ja": MecabJaPosTagger,
    "zh": JiebaZhPosTagger,
}


class DoolyPosDpTokenizer(DoolyPreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "pos_vocab_file": "pos_vocab.json",
    }
    replacement: str = "▃"

    def __init__(self, pos_vocab_file, **kwargs):
        super().__init__(**kwargs)

        with open(pos_vocab_file, "r", encoding="utf-8") as f:
            self.pos_vocab = json.load(f)

        # set pos tagger
        tagger_cls = PosTaggerMap.get(self.lang, None)
        self.pos_tagger = tagger_cls()

    def _tokenize(self, text: str, **kwargs) -> Tuple[List[str], List[str]]:
        text = text.strip()
        pairs = self.pos_tagger.pos(text, return_surface=True)
        tokens = ["<s>", self.replacement] + [
            pair[0] if pair[0] != " " else self.replacement for pair in pairs
        ]
        tags = [
            pair[1] if pair[0] != " " else pairs[i + 1][1]
            for i, pair in enumerate(pairs)
        ]
        prefix = ["XX", tags[0]]
        tags = prefix + tags

        res_tags = []
        for tag in tags:
            if "+" in tag:
                tag = tag[: tag.find("+")]
            res_tags.append(tag)

        return tokens, res_tags

    @staticmethod
    def _sanitize_kwargs(
        method: Callable, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_kwargs = {}
        for arg_name in inspect.getfullargspec(method).args:
            val = kwargs.pop(arg_name, None)
            if val is not None:
                orig_kwargs[arg_name] = val
        return orig_kwargs, kwargs

    def convert_tags_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if tokens is None:
            return None

        def _convert_tag_to_id(token: str) -> int:
            if token is None:
                return None
            return self.pos_vocab.get(token, self.unk_token_id)

        if isinstance(tokens, str):
            return _convert_tag_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(_convert_tag_to_id(token))
        return ids

    def _get_input_ids(self, text: str, **kwargs) -> Tuple[List[int], List[int]]:
        tokens, tags = self.tokenize(text, **kwargs)
        return (
            self.convert_tokens_to_ids(tokens),
            self.convert_tags_to_ids(tags),
        )

    def _encode_plus(self, text: str, text_pair: str, **kwargs) -> BatchEncoding:
        if kwargs.pop("return_offsets_mapping", False):
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )
        if kwargs.pop("is_split_into_words", False):
            raise NotImplementedError(
                "is_split_into_words is not available when using Pos Tokenizer. "
            )
        orig_kwargs, kwargs = self._sanitize_kwargs(super()._encode_plus, **kwargs)
        first_ids, first_tag_ids = self._get_input_ids(text, **kwargs)
        second_ids = second_tag_ids = None
        if text_pair is not None:
            second_ids, second_tag_ids = self._get_input_ids(text_pair, **kwargs)

        batch_outputs = self.prepare_for_model(
            ids=first_ids, pair_ids=second_ids, prepend_batch_axis=True, **orig_kwargs
        )
        pos_outputs = self.prepare_for_model(
            ids=first_tag_ids,
            pair_ids=second_tag_ids,
            prepend_batch_axis=True,
            **orig_kwargs,
        )
        batch_outputs.update({"segment_labels": pos_outputs["input_ids"]})
        return batch_outputs

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs) -> BatchEncoding:
        if kwargs.pop("return_offsets_mapping", False):
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )
        if kwargs.pop("is_split_into_words", False):
            raise NotImplementedError(
                "is_split_into_words is not available when using Pos Tokenizer. "
            )
        orig_kwargs, kwargs = self._sanitize_kwargs(
            super()._batch_encode_plus, **kwargs
        )
        input_ids = []
        tag_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids, first_tag_ids = self._get_input_ids(ids, **kwargs)
            second_ids = second_tag_ids = None
            if pair_ids is not None:
                second_ids, second_tag_ids = _get_input_ids(pair_ids, **kwargs)
            input_ids.append((first_ids, second_ids))
            tag_ids.append((first_tag_ids, second_tag_ids))

        batch_outputs = self._batch_prepare_for_model(input_ids, **orig_kwargs)
        pos_outputs = self._batch_prepare_for_model(tag_ids, **orig_kwargs)
        batch_outputs.update({"segment_labels": pos_outputs["input_ids"]})
        return BatchEncoding(batch_outputs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        return text.replace(self.replacement, " ").strip()
