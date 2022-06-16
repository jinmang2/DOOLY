import os
import json
import shutil
from zipfile import ZipFile

from .base import DoolyConverter, FsmtConverter, RobertaConverter
from .kobart_utils import download

from ..tokenizers.fast import (
    convert_vocab_from_fairseq_to_hf,
    build_custom_roberta_tokenizer,
    PreTrainedTokenizerFast,
)
from ..models import (
    RobertaForCharNER,
    RobertaForDependencyParsing,
    RobertaForSpanPrediction,
    RobertaForSequenceClassification,
)


class DpConverter(RobertaConverter):
    name = "dp"
    pororo_task_head_name = "dependency_parse_head"
    hf_model_class = RobertaForDependencyParsing

    def load_and_save_vocab(self):
        # vocab
        super().load_and_save_vocab()
        # label0 and label1
        label0 = self._pororo_model.task.label0_dictionary.indices
        label1 = self._pororo_model.task.label1_dictionary.indices
        self.save_vocab(label0, "label0.json")
        self.save_vocab(label1, "label1.json")
        # pos vocab
        pos_vocab = self._pororo_model.task.source_dictionary.indices
        self.save_vocab(pos_vocab, "pos_vocab.json")

    def porting_pororo_to_hf(self, pororo_model, hf_model):
        hf_model = super().porting_pororo_to_hf(self, pororo_model, hf_model)

        cls_head = pororo_model.model.classification_heads[self.pororo_task_head_name]

        # pre-head attention
        hf_model.classifier.head_attn_pre.in_proj_weight = (
            cls_head.head_attn_pre.in_proj_weight
        )
        hf_model.classifier.head_attn_pre.in_proj_bias = (
            cls_head.head_attn_pre.in_proj_bias
        )
        hf_model.classifier.head_attn_pre.out_proj.weight = (
            cls_head.head_attn_pre.out_proj.weight
        )
        hf_model.classifier.head_attn_pre.out_proj.bias = (
            cls_head.head_attn_pre.out_proj.bias
        )

        # post-head attention
        hf_model.classifier.head_attn_post.in_proj_weight = (
            cls_head.head_attn_post.in_proj_weight
        )
        hf_model.classifier.head_attn_post.in_proj_bias = (
            cls_head.head_attn_post.in_proj_bias
        )
        hf_model.classifier.head_attn_post.out_proj.weight = (
            cls_head.head_attn_post.out_proj.weight
        )
        hf_model.classifier.head_attn_post.out_proj.bias = (
            cls_head.head_attn_post.out_proj.bias
        )

        return hf_model


class MrcConverter(RobertaConverter):
    name = "mrc"
    pororo_task_head_name = "span_prediction_head"
    hf_model_class = RobertaForSpanPrediction

    def get_model_config(self, pororo_model):
        config = super().get_model_config(pororo_model)
        config.span_head_dropout = 0.0
        config.span_head_inner_dim = 768
        return config


class MtConverter(FsmtConverter):
    name = "mt"

    def load_and_save_vocab(self):
        # vocab for encode
        super().load_and_save_vocab()
        # vocab for tokenize (only support en sub-tokenizer)
        file_path = "tokenizers/bpe32k.en/"
        for filename in ["merges.txt", "vocab.json"]:
            shutil.move(
                src=os.path.join(self._pororo_save_path, file_path, filename),
                dst=os.path.join(self.save_path, "en"),
            )
        convert_vocab_from_fairseq_to_hf(os.path.join(self.save_path, "en/vocab.json"))
        sub_tokenizer = build_custom_roberta_tokenizer(
            name_or_path="",
            vocab_filename=os.path.join(self.save_path, "en/vocab.json"),
            merges_filename=os.path.join(self.save_path, "en/merges.txt"),
        )
        sub_tokenizer.save_pretrained(os.path.join(self.save_path, "en"))


class NerConverter(RobertaConverter):
    name = "ner"
    pororo_task_head_name = "sequence_tagging_head"
    hf_model_class = RobertaForCharNER

    def get_model_config(self, pororo_model):
        config = super().get_model_config(pororo_model)
        if self.lang == "ko":
            config.id2label = {
                0: "O",
                1: "I-ORGANIZATION",
                2: "I-CIVILIZATION",
                3: "I-QUANTITY",
                4: "I-DATE",
                5: "B-CIVILIZATION",
                6: "I-PERSON",
                7: "I-LOCATION",
                8: "B-ORGANIZATION",
                9: "B-QUANTITY",
                10: "I-ARTIFACT",
                11: "B-PERSON",
                12: "B-DATE",
                13: "B-LOCATION",
                14: "I-EVENT",
                15: "I-TERM",
                16: "B-ARTIFACT",
                17: "B-TERM",
                18: "I-TIME",
                19: "B-EVENT",
                20: "I-STUDY_FIELD",
                21: "B-ANIMAL",
                22: "I-THEORY",
                23: "I-MATERIAL",
                24: "B-TIME",
                25: "I-ANIMAL",
                26: "B-STUDY_FIELD",
                27: "B-MATERIAL",
                28: "B-THEORY",
                29: "I-PLANT",
                30: "B-PLANT",
            }
            config.label2id = {
                "O": 0,
                "I-ORGANIZATION": 1,
                "I-CIVILIZATION": 2,
                "I-QUANTITY": 3,
                "I-DATE": 4,
                "B-CIVILIZATION": 5,
                "I-PERSON": 6,
                "I-LOCATION": 7,
                "B-ORGANIZATION": 8,
                "B-QUANTITY": 9,
                "I-ARTIFACT": 10,
                "B-PERSON": 11,
                "B-DATE": 12,
                "B-LOCATION": 13,
                "I-EVENT": 14,
                "I-TERM": 15,
                "B-ARTIFACT": 16,
                "B-TERM": 17,
                "I-TIME": 18,
                "B-EVENT": 19,
                "I-STUDY_FIELD": 20,
                "B-ANIMAL": 21,
                "I-THEORY": 22,
                "I-MATERIAL": 23,
                "B-TIME": 24,
                "I-ANIMAL": 25,
                "B-STUDY_FIELD": 26,
                "B-MATERIAL": 27,
                "B-THEORY": 28,
                "I-PLANT": 29,
                "B-PLANT": 30,
            }
        elif self.lang == "en":
            config.id2label = {
                0: "O",
                1: "I-ORG",
                2: "I-PERSON",
                3: "I-DATE",
                4: "I-GPE",
                5: "B-ORG",
                6: "B-GPE",
                7: "B-PERSON",
                8: "B-DATE",
                9: "I-MONEY",
                10: "B-CARDINAL",
                11: "I-PERCENT",
                12: "B-NORP",
                13: "I-CARDINAL",
                14: "B-MONEY",
                15: "B-PERCENT",
                16: "I-TIME",
                17: "I-LOC",
                18: "I-FAC",
                19: "I-QUANTITY",
                20: "I-NORP",
                21: "I-EVENT",
                22: "B-ORDINAL",
                23: "I-PRODUCT",
                24: "B-LOC",
                25: "B-TIME",
                26: "I-LAW",
                27: "B-QUANTITY",
                28: "B-FAC",
                29: "B-PRODUCT",
                30: "B-EVENT",
                31: "I-ORDINAL",
                32: "B-LAW",
                33: "B-LANGUAGE",
                34: "I-LANGUAGE",
            }
            config.label2id = {
                "O": 0,
                "I-ORG": 1,
                "I-PERSON": 2,
                "I-DATE": 3,
                "I-GPE": 4,
                "B-ORG": 5,
                "B-GPE": 6,
                "B-PERSON": 7,
                "B-DATE": 8,
                "I-MONEY": 9,
                "B-CARDINAL": 10,
                "I-PERCENT": 11,
                "B-NORP": 12,
                "I-CARDINAL": 13,
                "B-MONEY": 14,
                "B-PERCENT": 15,
                "I-TIME": 16,
                "I-LOC": 17,
                "I-FAC": 18,
                "I-QUANTITY": 19,
                "I-NORP": 20,
                "I-EVENT": 21,
                "B-ORDINAL": 22,
                "I-PRODUCT": 23,
                "B-LOC": 24,
                "B-TIME": 25,
                "I-LAW": 26,
                "B-QUANTITY": 27,
                "B-FAC": 28,
                "B-PRODUCT": 29,
                "B-EVENT": 30,
                "I-ORDINAL": 31,
                "B-LAW": 32,
                "B-LANGUAGE": 33,
                "I-LANGUAGE": 34,
            }
        elif self.lang == "ja":
            config.id2label = {
                0: "O",
                1: "B-LOCATION",
                2: "I-LOCATION",
                3: "I-ARTIFACT",
                4: "I-ORGANIZATION",
                5: "I-DATE",
                6: "B-DATE",
                7: "I-PERSON",
                8: "B-ORGANIZATION",
                9: "B-ARTIFACT",
                10: "B-PERSON",
                11: "I-OPTIONAL",
                12: "B-OPTIONAL",
                13: "I-MONEY",
                14: "I-PERCENT",
                15: "B-MONEY",
                16: "I-TIME",
                17: "B-PERCENT",
                18: "B-TIME",
            }
            config.label2id = {
                "O": 0,
                "B-LOCATION": 1,
                "I-LOCATION": 2,
                "I-ARTIFACT": 3,
                "I-ORGANIZATION": 4,
                "I-DATE": 5,
                "B-DATE": 6,
                "I-PERSON": 7,
                "B-ORGANIZATION": 8,
                "B-ARTIFACT": 9,
                "B-PERSON": 10,
                "I-OPTIONAL": 11,
                "B-OPTIONAL": 12,
                "I-MONEY": 13,
                "I-PERCENT": 14,
                "B-MONEY": 15,
                "I-TIME": 16,
                "B-PERCENT": 17,
                "B-TIME": 18,
            }
        elif self.lang == "zh":
            config.id2label = {
                0: "O",
                1: "I-ORG",
                2: "I-PERSON",
                3: "I-GPE",
                4: "I-DATE",
                5: "B-GPE",
                6: "B-PERSON",
                7: "B-ORG",
                8: "B-DATE",
                9: "B-CARDINAL",
                10: "I-MONEY",
                11: "I-CARDINAL",
                12: "I-EVENT",
                13: "I-TIME",
                14: "I-FAC",
                15: "I-LOC",
                16: "I-PERCENT",
                17: "I-NORP",
                18: "I-QUANTITY",
                19: "B-LOC",
                20: "B-NORP",
                21: "B-TIME",
                22: "I-LAW",
                23: "B-FAC",
                24: "B-MONEY",
                25: "I-ORDINAL",
                26: "B-ORDINAL",
                27: "B-EVENT",
                28: "I-PRODUCT",
                29: "B-QUANTITY",
                30: "B-PERCENT",
                31: "I-LANGUAGE",
                32: "B-PRODUCT",
                33: "B-LANGUAGE",
                34: "B-LAW",
            }
            config.label2id = {
                "O": 0,
                "I-ORG": 1,
                "I-PERSON": 2,
                "I-GPE": 3,
                "I-DATE": 4,
                "B-GPE": 5,
                "B-PERSON": 6,
                "B-ORG": 7,
                "B-DATE": 8,
                "B-CARDINAL": 9,
                "I-MONEY": 10,
                "I-CARDINAL": 11,
                "I-EVENT": 12,
                "I-TIME": 13,
                "I-FAC": 14,
                "I-LOC": 15,
                "I-PERCENT": 16,
                "I-NORP": 17,
                "I-QUANTITY": 18,
                "B-LOC": 19,
                "B-NORP": 20,
                "B-TIME": 21,
                "I-LAW": 22,
                "B-FAC": 23,
                "B-MONEY": 24,
                "I-ORDINAL": 25,
                "B-ORDINAL": 26,
                "B-EVENT": 27,
                "I-PRODUCT": 28,
                "B-QUANTITY": 29,
                "B-PERCENT": 30,
                "I-LANGUAGE": 31,
                "B-PRODUCT": 32,
                "B-LANGUAGE": 33,
                "B-LAW": 34,
            }

    def get_misc_filenames(self):
        misc_files = []
        if self.lang == "ko":
            misc_files = [
                f"misc/wiki.{self.lang}.items",
                "misc/wsd.cls.txt",
                "misc/re.templates.txt",
            ]
        return misc_files


class NliConverter(RobertaConverter):
    name = "nli"
    pororo_task_head_name = "sentence_classification_head"
    hf_model_class = RobertaForSequenceClassification

    def get_model_config(self, pororo_model):
        config = super().get_model_config(pororo_model)
        if self.lang == "ko":
            config.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
            config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        elif self.lang == "en":
            config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
            config.label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}
        else:  # ja, zh case, E261 at least two spaces before inline comment
            config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
            config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        return config


class QgConverter(DoolyConverter):
    name = "qg"

    def get_kobart_tokenizer(self):
        file_path, is_cached = download(
            url="s3://skt-lsl-nlp-model/KoBART/tokenizers/kobart_base_tokenizer_cased_cf74400bce.zip",
            chksum="cf74400bce",
            cachedir=self.save_path,
        )
        cachedir_full = os.path.expanduser(self.save_path)
        if (
            not os.path.exists(os.path.join(cachedir_full, "emji_tokenizer"))
            or not is_cached
        ):
            if not is_cached:
                shutil.rmtree(
                    os.path.join(cachedir_full, "emji_tokenizer"), ignore_errors=True
                )
            zipf = ZipFile(os.path.expanduser(file_path))
            zipf.extractall(path=cachedir_full)
        tok_path = os.path.join(cachedir_full, "emji_tokenizer/model.json")

        with open(tok_path, "r", encoding="utf-8") as f:
            tok_model = json.load(f)

        tok_model["post_processor"] = {
            "type": "RobertaProcessing",
            "sep": ["</s>", 1],
            "cls": ["<s>", 0],
            "trim_offsets": True,
            "add_prefix_space": True,
        }

        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(tok_model, f, ensure_ascii=False)

        kobart_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
        return kobart_tokenizer

    def load_and_save_vocab(self):
        kobart_tokenizer = self.get_kobart_tokenizer()
        kobart_tokenizer.save_pretrained(self.save_path)

    def get_model_config(self):
        return None

    def intialize_hf_model(self, config):
        return None

    def porting_pororo_to_hf(self, pororo_model, hf_model):
        return pororo_model.model

    def get_misc_filenames(self):
        misc_files = []
        if self.lang == "ko":
            misc_files = ["misc/ko_indexdir.zip"]
        return misc_files


class StsConverter(RobertaConverter):
    name = "sts"
    pororo_task_head_name = "sentence_classification_head"
    hf_model_class = RobertaForSequenceClassification

    def convert(self):
        if "sbert" not in self.n_model:
            return super().convert()

        # UKPLab/sentence_transformers
        self._hf_model = self._pororo_model
        self._hf_model.save(path=self.save_path, create_model_card=False)

        return self._pororo_model, self._hf_model


class WsdConverter(FsmtConverter):
    name = "wsd"

    def get_misc_filenames(self):
        misc_files = [
            f"misc/morph2idx.{self.lang}.pkl",
            f"misc/tag2idx.{self.lang}.pkl",
            f"misc/wsd-dicts.{self.lang}.pkl",
        ]
        return misc_files
