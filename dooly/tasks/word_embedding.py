from typing import List, Dict, Tuple, Union, Optional
from collections import OrderedDict

from .base import DoolyTaskConfig, Miscellaneous
from .base.wikipedia2vec import (
    QueryParser,
    WhooshIndex,
    Wikipedia2Vec,
)


class WordEmbedding(Miscellaneous):
    """
    Get vector or find similar word and entity from pretrained model using wikipedia

    See also:
        Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia (https://arxiv.org/abs/1812.06280)

    Korean (`wikipedia2vec.ko`)
        - dataset: kowiki-20200720
        - metric: N/A

    English (`wikipedia2vec.en`)
        - dataset: enwiki-20180420
        - metric: N/A

    Japanese (`wikipedia2vec.ja`)
        - dataset: jawiki-20180420
        - metric: N/A

    Chinese (`wikipedia2vec.zh`)
        - dataset: zhwiki-20180420
        - metric: N/A

    Args:
        query (str): input qeury
        top_n (int): number of result word or entity (need for `find_similar_words`)
        group (bool): return grouped dictionary or not (need for `find_similar_words`)

    Notes:
        Wikipedia2Vec has two diffrent kinds of output format following below.
        1. 'something' (word) : word2vec result (non-hyperlink in wikipedia documents)
        2. 'something' (other) : entity2vec result (hyperlink in wikipedia documents)

    """
    task: str = "word_embedding"
    available_langs: List[str] = ["ko", "en", "ja", "zh"]
    available_models: Dict[str, List[str]] = {
        "ko": ["wikipedia2vec.ko"],
        "en": ["wikipedia2vec.en"],
        "ja": ["wikipedia2vec.ja"],
        "zh": ["wikipedia2vec.zh"],
    }
    misc_files: Dict[str, List[str]] = {
        "ko": ["kowiki_20200720_100d.pkl", "ko_indexdir.zip"],
        "en": ["enwiki_20180420_100d.pkl", "en_indexdir.zip"],
        "ja": ["jawiki_20180420_100d.pkl", "ja_indexdir.zip"],
        "zh": ["zhwiki_20180420_100d.pkl", "zh_indexdir.zip"],
    }

    def __init__(self, config: DoolyTaskConfig):
        super().__init__(config=config)
        self._model = Wikipedia2Vec(config.misc_tuple[0], self.device)
        self._ix = WhooshIndex.open_dir(config.misc_tuple[1])

    def __call__(self, query: str, **kwargs):
        searchterm = self._normalize(query)
        word2vec = self._get_word_vector(searchterm)
        entity2vec = self._get_entity_vectors(searchterm)
        word2vec.update(entity2vec)

        if not word2vec:
            raise ValueError(f"Oops! {query} does NOT exist in our database.")

        return word2vec

    def _normalize(self, query):
        """
        normalize input query

        Args:
            query (str): input query

        Returns:
            str: normalized input qeury

        """
        searchterm = query.lower().replace(" ", "_")
        return searchterm

    def _get_word_vector(self, word: str):
        """
        get word vector from word string

        Args:
            word (str): word string

        Returns:
            OrderedDict: {word_string: word_vector}

        """
        headword2vec = OrderedDict()
        Word = self._model.get_word(word)

        if Word is not None:
            vec = self._model.get_word_vector(word)
            headword = f"{Word.text} (word)"
            headword2vec[headword] = vec

        return headword2vec

    def _get_entity_vectors(self, entity: str):
        """
        get entity vector from entity string

        Args:
            entity (str): entity string

        Returns:
            OrderedDict: {entity_string: entity_vector}

        """
        headword2vec = OrderedDict()
        with self._ix.searcher() as searcher:
            query = QueryParser("searchterms", self._ix.schema).parse(entity)
            hits = searcher.search(query)

            for hit in hits:
                if "wiki_title" in hit:
                    wiki_title = hit["wiki_title"]
                    category = hit["categories"]
                    headword = f"{wiki_title} ({category})"
                    Entity = self._model.get_entity(wiki_title)
                    if Entity is not None:
                        vec = self._model.get_entity_vector(wiki_title)
                        headword2vec[headword] = vec
        return headword2vec

    @staticmethod
    def _append(headword, relative, headword2relatives):
        """
        append relative to dictionary

        Args:
            headword: head word
            relative: relative word or entity dictionary
            headword2relatives: given result dictionary

        """

        if headword in headword2relatives:
            headword2relatives[headword].append(relative)
        else:
            headword2relatives[headword] = [relative]

    def _postprocess(self, headword2relatives):
        """
        postprocessing for better output format

        Args:
            headword2relatives (OrderedDict):

        Returns:
            OrderedDict: postprocessed output

        """
        new_headword2relatives = OrderedDict()
        for headword, relatives in headword2relatives.items():
            cat2words = OrderedDict()
            for relative in relatives:
                word, category = relative.rsplit(" (", 1)
                category = category[:-1]
                categories = category.split(";")
                for category in categories:
                    self._append(category, word, cat2words)
            new_headword2relatives[headword] = cat2words

        return new_headword2relatives

    def find_similar_words(self, query, top_n=5, group=False):
        """
        find similar words from input query

        Args:
            query (str): input query
            top_n (int): number of result
            group (bool): return grouped dictionary or not

        Returns:
            OrderedDict: word or entity search result

        """

        searchterm = self._normalize(query)

        # Final return
        headword2relatives = OrderedDict()

        with self._ix.searcher() as searcher:
            # Word
            Word = self._model.get_word(searchterm)
            if Word is not None:
                word = Word.text
                headword = f"{word} (word)"
                results = self._model.most_similar(Word, top_n + 1)
                # note that the first result is the word itself.
                if len(results) > 1:
                    for result in results[1:]:  # returned by wikipedia2vec
                        if hasattr(result[0], "text"):  # word
                            relative = result[0].text
                            relative_ = f"{relative} (word)"
                            self._append(
                                headword,
                                relative_,
                                headword2relatives,
                            )
                        else:  # entity
                            relative = result[0].title
                            idx = result[0].index.item()

                            from_idx = QueryParser(
                                "entity_idx",
                                self._ix.schema,
                            ).parse(str(idx))
                            hits = searcher.search(from_idx)
                            if len(hits) > 0:
                                category = hits[0]["categories"]
                                relative_ = f"{relative} ({category})"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )
                            else:
                                relative_ = f"{relative} (misc)"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )

            # Entity
            from_searchterms = QueryParser(
                "searchterms",
                self._ix.schema,
            ).parse(searchterm)
            hits = searcher.search(from_searchterms)

            # returned by indexer <Hit {'categories': 'human', 'display': 'Messi', 'wiki_title': 'Messi (2014 film)'}>
            for hit in hits:
                wiki_title = hit["wiki_title"]
                Entity = self._model.get_entity(wiki_title)
                entity = Entity.title
                category = hit["categories"]
                headword = f"{entity} ({category})"

                results = self._model.most_similar(Entity, top_n + 1)
                # note that the first result is the word itself.
                if len(results) > 1:
                    for result in results[1:]:
                        if hasattr(result[0], "text"):  # word
                            relative = result[0].text
                            relative_ = f"{relative} (word)"
                            self._append(
                                headword,
                                relative_,
                                headword2relatives,
                            )
                        else:  # entity
                            relative = result[0].title
                            idx = result[0].index.item()

                            from_idx = QueryParser(
                                "entity_idx",
                                self._ix.schema,
                            ).parse(str(idx))
                            hits = searcher.search(from_idx)
                            if len(hits) > 0:
                                category = hits[0]["categories"]
                                relative_ = f"{relative} ({category})"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )
                            else:
                                relative_ = f"{relative} (misc)"
                                self._append(
                                    headword,
                                    relative_,
                                    headword2relatives,
                                )

        return self._postprocess(
            headword2relatives) if group else headword2relatives
