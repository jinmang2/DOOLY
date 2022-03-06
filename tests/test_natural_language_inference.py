def convert_fairseq_to_hf_transformers(nli):

    sent_encoder = nli._model.model.encoder.sentence_encoder

    config = RobertaConfig(
        vocab_size=sent_encoder.embed_tokens.num_embeddings,
        hidden_size=nli._model.args.encoder_embed_dim,
        num_hidden_layers=nli._model.args.encoder_layers,
        num_attention_heads=nli._model.args.encoder_attention_heads,
        intermediate_size=nli._model.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )

    cls_heads = nli._model.model.classification_heads
    task = "sentence_classification_head"
    config.num_labels = cls_heads[task].out_proj.weight.shape[0]

    nli_model_new = RobertaForSequenceClassification(config)
    nli_model_new.eval()

# Now let's copy all the weights.
    # Embeddings
    nli_model_new.roberta.embeddings.word_embeddings.weight = sent_encoder.embed_tokens.weight
    nli_model_new.roberta.embeddings.position_embeddings.weight = sent_encoder.embed_positions.weight
    nli_model_new.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        nli_model_new.roberta.embeddings.token_type_embeddings.weight
        )  # just zero them out b/c RoBERTa doesn't use them.
    nli_model_new.roberta.embeddings.LayerNorm.weight = sent_encoder.emb_layer_norm.weight
    nli_model_new.roberta.embeddings.LayerNorm.bias = sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer = nli_model_new.roberta.encoder.layer[i]
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

    nli_model_new.classifier.dense.weight = nli._model.model.classification_heads[task].dense.weight
    nli_model_new.classifier.dense.bias = nli._model.model.classification_heads[task].dense.bias
    nli_model_new.classifier.out_proj.weight = nli._model.model.classification_heads[task].out_proj.weight
    nli_model_new.classifier.out_proj.bias = nli._model.model.classification_heads[task].out_proj.bias

    input_ids = torch.LongTensor(1, 30).random_(sent_encoder.embed_tokens.num_embeddings,)
    our_logits = nli_model_new(input_ids).logits
    their_logits = nli._model.model.classification_heads[task](nli._model.extract_features(input_ids))

    max_absolute_diff = torch.max(torch.abs(our_logits - their_logits)).item()
    success = torch.allclose(our_logits, their_logits, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    return nli_model_new


nli_ko_new.config.id2label = {
    0: "contradiction",
    1: "neutral",
    2: "entailment",
}
nli_ko_new.config.label2id = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
}
nli_ko_new.config._name_or_path = "jinmang2/dooly-hub/nli/ko/brainbert.base"


nli_en_new.config.id2label = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}
nli_en_new.config.label2id = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2,
}
nli_en_new.config._name_or_path = "jinmang2/dooly-hub/nli/en/roberta.base"

nli_ja_new.config.id2label = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}
nli_ja_new.config.label2id = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
nli_ja_new.config._name_or_path = "jinmang2/dooly-hub/nli/ja/jaberta.base"

nli_zh_new.config.id2label = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}
nli_zh_new.config.label2id = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
nli_zh_new.config._name_or_path = "jinmang2/dooly-hub/nli/zh/zhberta.base"

# !mv ../root/.pororo/misc/encoder.json ./dooly-hub/nli/en/roberta.base/
# !mv ../root/.pororo/misc/vocab.bpe ./dooly-hub/nli/en/roberta.base/

with open("dooly-hub/nli/en/roberta.base/vocab.json", "w", encoding="utf-8") as f:
    json.dump(nli_en._model.task.source_dictionary.indices, f)

with open("dooly-hub/nli/ja/jaberta.base/vocab.json", "w", encoding="utf-8") as f:
    json.dump(nli_ja._model.task.source_dictionary.indices, f)

with open("dooly-hub/nli/zh/zhberta.base/vocab.json", "w", encoding="utf-8") as f:
    json.dump(nli_zh._model.task.source_dictionary.indices, f)


input_ids = nli_ko._model.encode(
    "저는, 그냥 알아내려고 거기 있었어요",
    "나는 처음부터 그것을 잘 이해했다",
    no_separator=True
).view(1, -1)
prediction = nli_ko._model.predict(
    "sentence_classification_head",
    input_ids,
    return_logits=nli_ko._model.args.regression_target,
)
prediction

torch.nn.functional.softmax(torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)
