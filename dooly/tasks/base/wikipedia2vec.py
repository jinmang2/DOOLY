# Copyright (c) Studio Ousia, its affiliates and Kakao Brain. All Rights Reserved

import re
import string
import joblib
import importlib
from itertools import chain
from collections import OrderedDict, Counter

import numpy as np
import six
import torch

from typing import List


def is_available_marisa_trie():
    return importlib.util.find_spec("marisa_trie")


def is_available_whoosh():
    return importlib.util.find_spec("whoosh")


def is_available_joblib():
    return importlib.util.find_spec("joblib")


if is_available_whoosh():
    from whoosh.qparser import QueryParser
    from whoosh import index as WhooshIndex
else:
    whoosh_not_found_err = ModuleNotFoundError(
        "Please install whoosh with: `pip install whoosh`."
    )

    class QueryParser:
        def __new__(cls, *args, **kwargs):
            raise whoosh_not_found_err

    class WhooshIndex:
        def __new__(cls, *args, **kwargs):
            raise whoosh_not_found_err

        @classmethod
        def open_dir(cls, *args, **kwargs):
            raise whoosh_not_found_err


class Wikipedia2VecItem(object):
    r"""Python wrapper class for wikipedia2vec item class"""

    def __init__(self, index, count, doc_count):
        self.index = index
        self.count = count
        self.doc_count = doc_count


class Wikipedia2VecWord(Wikipedia2VecItem):
    r"""Python wrapper class for wikipedia2vec word class"""

    def __init__(self, text, index, count, doc_count):
        super().__init__(index, count, doc_count)
        self.text = text

    def __repr__(self):
        return f"<Word {self.text}>"

    def __reduce__(self):
        return self.__class__, (
            self.text,
            self.index,
            self.count,
            self.doc_count,
        )


class Wikipedia2VecEntity(Wikipedia2VecItem):
    r"""Python wrapper class for wikipedia2vec entity class"""

    def __init__(self, title, index, count, doc_count):
        super().__init__(index, count, doc_count)
        self.title = title

    def __repr__(self):
        return f"<Entity {self.title}>"

    def __reduce__(self):
        return self.__class__, (
            self.title,
            self.index,
            self.count,
            self.doc_count,
        )


class Wikipedia2VecDict(object):
    r"""Python wrapper class for wikipedia2vec dictionary class"""

    def __init__(
        self,
        word_dict,
        entity_dict,
        redirect_dict,
        word_stats,
        entity_stats,
        language,
        lowercase,
        build_params,
        min_paragraph_len=0,
        uuid="",
        device="cuda",
    ):
        self._word_dict = word_dict
        self._word_dict = word_dict
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict
        self._word_stats = word_stats[: len(self._word_dict)]
        self._entity_stats = entity_stats[: len(self._entity_dict)]
        self.min_paragraph_len = min_paragraph_len
        self.uuid = uuid
        self.language = language
        self.lowercase = lowercase
        self.build_params = build_params
        self._entity_offset = len(self._word_dict)
        self.device = device

    @property
    def entity_offset(self):
        return self._entity_offset

    @property
    def word_size(self):
        return len(self._word_dict)

    @property
    def entity_size(self):
        return len(self._entity_dict)

    def __len__(self):
        return len(self._word_dict) + len(self._entity_dict)

    def __iter__(self):
        return chain(self.words(), self.entities())

    def words(self):
        for (word, index) in six.iteritems(self._word_dict):
            yield Wikipedia2VecWord(word, index, *self._word_stats[index])

    def entities(self):
        for (title, index) in six.iteritems(self._entity_dict):
            yield Wikipedia2VecEntity(
                title,
                index + self._entity_offset,
                *self._entity_stats[index],
            )

    def get_word(self, word, default=None):
        index = self.get_word_index(word)

        if index == -1:
            return default
        return Wikipedia2VecWord(word, index, *self._word_stats[index])

    def get_entity(self, title, resolve_redirect=True, default=None):
        index = self.get_entity_index(title, resolve_redirect=resolve_redirect)

        if index == -1:
            return default

        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Wikipedia2VecEntity(
            title,
            index,
            *self._entity_stats[dict_index],
        )

    def get_word_index(self, word):
        try:
            return self._word_dict[word]
        except KeyError:
            return -1

    def get_entity_index(self, title, resolve_redirect=True):
        if resolve_redirect:
            try:
                index = self._redirect_dict[title][0][0]
                return index + self._entity_offset
            except KeyError:
                pass
        try:
            index = self._entity_dict[title]
            return index + self._entity_offset
        except KeyError:
            return -1

    def get_item_by_index(self, index):
        if index < self._entity_offset:
            return self.get_word_by_index(index)
        return self.get_entity_by_index(index)

    def get_word_by_index(self, index):
        word = self._word_dict.restore_key(index)
        return Wikipedia2VecWord(
            word,
            index,
            *self._word_stats[index],
        )

    def get_entity_by_index(self, index):
        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Wikipedia2VecEntity(
            title,
            index,
            *self._entity_stats[dict_index],
        )

    @staticmethod
    def load(target, device, mmap=True):

        if is_available_marisa_trie():
            from marisa_trie import RecordTrie, Trie
        else:
            raise ModuleNotFoundError(
                "Please install marisa trie with: `pip install marisa_trie`."
            )

        if is_available_joblib():
            import joblib
        else:
            raise ModuleNotFoundError(
                "Please install joblib with: `pip install joblib`."
            )

        word_dict = Trie()
        entity_dict = Trie()
        redirect_dict = RecordTrie("<I")

        if not isinstance(target, dict):
            if mmap:
                target = joblib.load(target, mmap_mode="r")
            else:
                target = joblib.load(target)

        word_dict.frombytes(target["word_dict"])
        entity_dict.frombytes(target["entity_dict"])
        redirect_dict.frombytes(target["redirect_dict"])

        word_stats = target["word_stats"]
        entity_stats = target["entity_stats"]
        if not isinstance(word_stats, np.ndarray):
            word_stats = np.frombuffer(
                word_stats,
                dtype=np.int32,
            ).reshape(-1, 2)
            word_stats = torch.tensor(
                word_stats,
                device=device,
                requires_grad=False,
            )
            entity_stats = np.frombuffer(
                entity_stats,
                dtype=np.int32,
            ).reshape(-1, 2)
            entity_stats = torch.tensor(
                entity_stats,
                device=device,
                requires_grad=False,
            )

        return Wikipedia2VecDict(
            word_dict,
            entity_dict,
            redirect_dict,
            word_stats,
            entity_stats,
            **target["meta"],
        )


class Wikipedia2Vec(object):
    def __init__(self, model_file, device):
        """
        Torch Wikipedia2Vec Wrapper class for word embedding task
        """

        model_object = joblib.load(model_file)

        if isinstance(model_object["dictionary"], dict):
            self.dictionary = Wikipedia2VecDict.load(
                model_object["dictionary"],
                device,
            )
        else:
            # for backward compatibilit
            self.dictionary = model_object["dictionary"]

        self.syn0 = torch.tensor(
            model_object["syn0"],
            device=device,
            requires_grad=False,
        )
        self.syn1 = torch.tensor(
            model_object["syn1"],
            device=device,
            requires_grad=False,
        )
        self.train_params = model_object.get("train_params")
        self.device = device

    def get_vector(self, item: Wikipedia2VecItem):
        return self.syn0[item.index]

    def get_word(self, word, default=None):
        return self.dictionary.get_word(word, default)

    def get_entity(self, title, resolve_redirect=True, default=None):
        return self.dictionary.get_entity(
            title,
            resolve_redirect,
            default,
        )

    def get_word_vector(self, word):
        obj = self.dictionary.get_word(word)

        if obj is None:
            return KeyError()
        return self.syn0[obj.index]

    def get_entity_vector(self, title, resolve_redirect=True):
        obj = self.dictionary.get_entity(
            title,
            resolve_redirect=resolve_redirect,
        )

        if obj is None:
            raise KeyError()
        return self.syn0[obj.index]

    def most_similar(self, item, count=100, min_count=None):
        vec = self.get_vector(item)

        return self.most_similar_by_vector(vec, count, min_count=min_count)

    def most_similar_by_vector(self, vec, count=100, min_count=None):
        if min_count is None:
            min_count = 0

        counts = torch.cat(
            [
                torch.tensor(
                    self.dictionary._word_stats[:, 0],
                    device=self.device,
                    requires_grad=False,
                ),
                torch.tensor(
                    self.dictionary._entity_stats[:, 0],
                    device=self.device,
                    requires_grad=False,
                ),
            ]
        )

        dst = self.syn0 @ vec / torch.norm(self.syn0, dim=1) / torch.norm(vec)
        dst[counts < min_count] = -100
        indexes = torch.argsort(-dst)

        return [
            (
                self.dictionary.get_item_by_index(ind),
                dst[ind],
            )
            for ind in indexes[:count]
        ]


class SimilarWords:
    def __init__(self, model, idx):
        self._wikipedia2vec = model
        self._ix = idx
        self._searcher = self._ix.searcher()

    def _normalize(self, word: str):
        """
        normalize input string
        Args:
            word (str): input string
        Returns:
            normalized string
        """
        return word.lower().replace(" ", "_")

    def _entity(self, entity: str):
        """
        find entity in entity dictionary
        Args:
            entity (str): entity string
        Returns:
            wikipedia2vec entity
        """
        entity = self._normalize(entity)
        entity = self._wikipedia2vec.get_entity(entity)
        return entity

    def _similar(self, entity: str):
        """
        find similar word with given entity
        Args:
            entity (str): answer that user inputted
        Returns:
            dict: category to entity list
        """
        entity_hit = None
        entity_ = self._entity(entity)
        headword2relatives = {}

        if not entity_:
            return headword2relatives

        from_searchterms = QueryParser(
            "searchterms",
            self._ix.schema,
        ).parse(entity)
        hits = self._searcher.search(from_searchterms)

        for hit in hits:
            wiki_title = hit["wiki_title"]
            if wiki_title == entity:
                entity_hit = hit

        if not entity_hit:
            return headword2relatives

        results = self._wikipedia2vec.most_similar(entity_)
        categories = entity_hit["categories"].split(";")
        category2entities = {category: [] for category in categories}

        if not results:
            return category2entities

        for result in results:
            if hasattr(result[0], "text"):
                continue

            if result[0].title == entity_.title or "분류" in result[0].title:
                continue

            idx = result[0].index.item()
            from_idx = QueryParser("entity_idx", self._ix.schema).parse(str(idx))
            hits2 = self._searcher.search(from_idx)

            if hits2:
                categories2 = hits2[0]["categories"].split(";")
                for each in categories2:
                    if each in category2entities:
                        category2entities[each].append(result[0].title)

        return category2entities

    def _extract_wrongs(self, entity: str) -> List[str]:
        """
        extract wrong answers candidates
        Args:
            entity: entity string
        Returns:
            wrong_list (List[str]): wrong answer candidates list
        """

        entity_list = []
        answer = self._normalize_answer(entity)
        sims = self._similar(answer)
        sims = list(sims.items())

        if len(sims) == 0:
            return entity_list

        for key, val in sims:
            if key.lower() != "word":
                entity_list += self._compare_with_answer(val, answer)

        return list(OrderedDict.fromkeys(entity_list))

    def _compare_with_answer(
        self,
        entity_list: List[str],
        answer: str,
    ) -> List[str]:
        """
        add wrong answer candidate to list
        after compare with answer using n-gram (f1 score)
        Args:
            entity_list (List[str]): wrong answers candidates
            answer (str): answer that will be compared wrong answer
        Returns:
            result_list (List[str]): wrong answer candidates list
        """

        result_list = []
        for e in entity_list:
            if "분류" not in e:
                e = re.sub(r"\([^)]*\)", "", e).strip()
                if self._f1_score(e, answer) < 0.5:
                    result_list.append(e)

        return result_list

    def _normalize_answer(self, s):
        """
        normalize answer string
        Args:
            s: sentence
        Returns:
            normalized sentence
        References:
            https://korquad.github.io/
        """

        def remove_(text):
            text = re.sub("'", " ", text)
            text = re.sub('"', " ", text)
            text = re.sub("《", " ", text)
            text = re.sub("》", " ", text)
            text = re.sub("<", " ", text)
            text = re.sub(">", " ", text)
            text = re.sub("〈", " ", text)
            text = re.sub("〉", " ", text)
            text = re.sub("\\(", " ", text)
            text = re.sub("\\)", " ", text)
            text = re.sub("‘", " ", text)
            text = re.sub("’", " ", text)
            return text

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(remove_(s))))

    def _f1_score(self, prediction, ground_truth):
        """
        compute F1 score
        Args:
            prediction: prediction answer
            ground_truth: ground truth answer
        Returns:
            F1 score between prediction and ground truth
        References:
            https://korquad.github.io/
        """

        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()

        # F1 by character
        prediction_Char = []
        for tok in prediction_tokens:
            now = [a for a in tok]
            prediction_Char.extend(now)

        ground_truth_Char = []
        for tok in ground_truth_tokens:
            now = [a for a in tok]
            ground_truth_Char.extend(now)

        common = Counter(prediction_Char) & Counter(ground_truth_Char)
        num_same = sum(common.values())
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_Char)
        recall = 1.0 * num_same / len(ground_truth_Char)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
