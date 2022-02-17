import os
import json
import pickle
import torch
import unittest
from collections import OrderedDict
from packaging import version
import fairseq
from fairseq.models.roberta import RobertaHubInterface
from pororo import Pororo
from pororo.tasks.utils.download_utils import download_or_load

# from dooly.hub_interface import HubInterface, TOKENIZER_USER_AGENT
from dooly.models import RobertaConfig, RobertaForCharNER
# from dooly.tasks.word_sense_disambiguation import KoWSDTokenizer


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")
    

def convert_fairseq_to_hf_transformers_ko():
    n_model = "charbert.base.ko.ner"
    model_path = os.path.join("bert", n_model)

    ckpt_dir = download_or_load(model_path, "ko")
    x = fairseq.hub_utils.from_pretrained(
        model_name_or_path=ckpt_dir,
        checkpoint_file="model.pt",
        data_name_or_path=ckpt_dir,
    )
    roberta = RobertaHubInterface(x["args"], x["task"], x["models"][0]).eval()

    sent_encoder = roberta.model.encoder.sentence_encoder

    config = RobertaConfig(
        vocab_size=sent_encoder.embed_tokens.num_embeddings,
        hidden_size=x["args"].encoder_embed_dim,
        num_hidden_layers=x["args"].encoder_layers,
        num_attention_heads=x["args"].encoder_attention_heads,
        intermediate_size=x["args"].encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )

    cls_heads = roberta.model.classification_heads
    task = "sequence_tagging_head"
    config.num_labels = cls_heads[task].out_proj.weight.shape[0]

    ner_model_new = RobertaForCharNER(config)
    ner_model_new.eval()

    # Now let's copy all the weights.
    # Embeddings
    ner_model_new.roberta.embeddings.word_embeddings.weight = sent_encoder.embed_tokens.weight
    ner_model_new.roberta.embeddings.position_embeddings.weight = sent_encoder.embed_positions.weight
    ner_model_new.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        ner_model_new.roberta.embeddings.token_type_embeddings.weight
        )  # just zero them out b/c RoBERTa doesn't use them.
    ner_model_new.roberta.embeddings.LayerNorm.weight = sent_encoder.emb_layer_norm.weight
    ner_model_new.roberta.embeddings.LayerNorm.bias = sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer = ner_model_new.roberta.encoder.layer[i]
        roberta_layer = sent_encoder.layers[i]

        # self attention
        self_attn = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # self-attention output
        self_output = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        # intermediate
        intermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # output
        bert_output = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        # end of layer

    ner_model_new.classifier.dense.weight = roberta.model.classification_heads[task].dense.weight
    ner_model_new.classifier.dense.bias = roberta.model.classification_heads[task].dense.bias
    ner_model_new.classifier.out_proj.weight = roberta.model.classification_heads[task].out_proj.weight
    ner_model_new.classifier.out_proj.bias = roberta.model.classification_heads[task].out_proj.bias

    input_ids = torch.LongTensor(1, 30).random_(sent_encoder.embed_tokens.num_embeddings,)
    our_logits = ner_model_new(input_ids).logits
    their_logits = roberta.model.classification_heads[task](roberta.extract_features(input_ids))

    max_absolute_diff = torch.max(torch.abs(our_logits - their_logits)).item()
    success = torch.allclose(our_logits, their_logits, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")
