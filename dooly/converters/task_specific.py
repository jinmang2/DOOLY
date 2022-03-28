import os
import shutil
from zipfile import ZipFile

from .base import DoolyConverter, FsmtConverter, RobertaConverter
from .kobart_utils import download

from ..tokenizers.hf_tokenizer import (
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

        hf_model.classifier.head_attn_pre.in_proj_weight = cls_head.head_attn_pre.in_proj_weight
        hf_model.classifier.head_attn_pre.in_proj_bias = cls_head.head_attn_pre.in_proj_bias
        hf_model.classifier.head_attn_pre.out_proj.weight = cls_head.head_attn_pre.out_proj.weight
        hf_model.classifier.head_attn_pre.out_proj.bias = cls_head.head_attn_pre.out_proj.bias

        hf_model.classifier.head_attn_post.in_proj_weight = cls_head.head_attn_post.in_proj_weight
        hf_model.classifier.head_attn_post.in_proj_bias = cls_head.head_attn_post.in_proj_bias
        hf_model.classifier.head_attn_post.out_proj.weight = cls_head.head_attn_post.out_proj.weight
        hf_model.classifier.head_attn_post.out_proj.bias = cls_head.head_attn_post.out_proj.bias


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
                dst=os.path.join(self.save_path, "en")
            )
        convert_vocab_from_fairseq_to_hf(os.path.join(self.save_path, "en/vocab.json"))
        sub_tokenizer = build_custom_roberta_tokenizer(
            name_or_path="",
            vocab_filename=os.path.join(self.save_path, "en/vocab.json"),
            merges_filename=os.path.join(self.save_path, "en/merges.txt")
        )
        sub_tokenizer.save_pretrained(os.path.join(self.save_path, "en"))


class NerConverter(RobertaConverter):
    name = "ner"
    pororo_task_head_name = "sequence_tagging_head"
    hf_model_class = RobertaForCharNER

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
        else: # ja, zh case
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
        cachedir_full = os.path.expanduser(cachedir)
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


class WsdConverter(FsmtConverter):
    name = "wsd"

    def get_misc_filenames(self):
        misc_files = [
            f"misc/morph2idx.{self.lang}.pkl",
            f"misc/tag2idx.{self.lang}.pkl",
            f"misc/wsd-dicts.{self.lang}.pkl",
        ]
        return misc_files
