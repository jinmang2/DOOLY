def convert_fairseq_to_hf_transformers(mrc):

    sent_encoder = mrc._model.model.encoder.sentence_encoder

    config = RobertaConfig(
        vocab_size=sent_encoder.embed_tokens.num_embeddings,
        hidden_size=mrc._model.args.encoder_embed_dim,
        num_hidden_layers=mrc._model.args.encoder_layers,
        num_attention_heads=mrc._model.args.encoder_attention_heads,
        intermediate_size=mrc._model.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    config.span_head_dropout = 0.0
    config.span_head_inner_dim = 768

    cls_heads = mrc._model.model.classification_heads
    task = "span_prediction_head"
    config.num_labels = cls_heads[task].out_proj.weight.shape[0]

    mrc_model_new = RobertaForSpanPrediction(config)
    mrc_model_new.eval()

    # Now let's copy all the weights.
    # Embeddings
    mrc_model_new.roberta.embeddings.word_embeddings.weight = sent_encoder.embed_tokens.weight
    mrc_model_new.roberta.embeddings.position_embeddings.weight = sent_encoder.embed_positions.weight
    mrc_model_new.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        mrc_model_new.roberta.embeddings.token_type_embeddings.weight
        )  # just zero them out b/c RoBERTa doesn't use them.
    mrc_model_new.roberta.embeddings.LayerNorm.weight = sent_encoder.emb_layer_norm.weight
    mrc_model_new.roberta.embeddings.LayerNorm.bias = sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer = mrc_model_new.roberta.encoder.layer[i]
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

    mrc_model_new.qa_outputs.dense.weight = mrc._model.model.classification_heads[task].dense.weight
    mrc_model_new.qa_outputs.dense.bias = mrc._model.model.classification_heads[task].dense.bias
    mrc_model_new.qa_outputs.out_proj.weight = mrc._model.model.classification_heads[task].out_proj.weight
    mrc_model_new.qa_outputs.out_proj.bias = mrc._model.model.classification_heads[task].out_proj.bias

    input_ids = torch.LongTensor(1, 30).random_(sent_encoder.embed_tokens.num_embeddings,)
    our_logits = mrc_model_new(input_ids)
    their_logits = mrc._model.model.classification_heads[task](mrc._model.extract_features(input_ids))

    # start logits
    max_absolute_diff = torch.max(
        torch.abs(our_logits.start_logits - their_logits[:, :, 0])
    ).item()
    print(f"max_absolute_diff(start_logits) = {max_absolute_diff}")
    # end logits
    max_absolute_diff = torch.max(
        torch.abs(our_logits.end_logits - their_logits[:, :, 1])
    ).item()
    print(f"max_absolute_diff(end_logits) = {max_absolute_diff}")
    success = torch.allclose(
        torch.cat([our_logits.start_logits.unsqueeze(-1),
                   our_logits.end_logits.unsqueeze(-1)], dim=-1),
        their_logits,
        atol=1e-3
    )
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    return mrc_model_new
