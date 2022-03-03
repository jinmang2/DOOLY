import math
import random
import torch
import torch.nn as nn
from transformers.models.fsmt.configuration_fsmt import FSMTConfig
from transformers.models.fsmt.modeling_fsmt import (
    SinusoidalPositionalEmbedding,
    EncoderLayer,
    DecoderLayer,
    FSMTModel as _FSMTModel,
    FSMTForConditionalGeneration as _FSMTForConditionalGeneration,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    # Seq2SeqLMOutput,
    # Seq2SeqModelOutput,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


class FSMTConfig(FSMTConfig):
    def __init__(
        self,
        encoder_pre_layernorm=False,
        decoder_pre_layernorm=False,
        **kwargs,
    ):
        self.encoder_pre_layernorm = encoder_pre_layernorm
        self.decoder_pre_layernorm = decoder_pre_layernorm
        super().__init__(**kwargs)


class FSMTEncoderLayer(EncoderLayer):

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        self.pre_layernorm = config.encoder_pre_layernorm

    def forward(self, x, encoder_padding_mask, layer_head_mask, output_attentions=False):
        """
        Args:
            x (`torch.Tensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            encoder_padding_mask (`torch.ByteTensor`): binary ByteTensor of shape
                *(batch, src_len)* where padding elements are indicated by `1`.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(config.encoder_attention_heads,)*.
        Returns:
            encoded output of shape *(seq_len, batch, embed_dim)*
        """
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=encoder_padding_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn_weights


class FSMTDecoderLayer(DecoderLayer):

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        self.pre_layernorm = config.decoder_pre_layernorm

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}

        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        # Self Attention
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.pre_layernorm:
            x = self.encoder_attn_layer_norm(x)
        x, cross_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
            layer_head_mask=cross_attn_layer_head_mask,
            output_attentions=output_attentions,
        )
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
            cross_attn_weights,
        )  # layer_state = cache for decoding


class FSMTEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`EncoderLayer`].
    Args:
        config: FSMTConfig
    """

    def __init__(self, config: FSMTConfig, embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [FSMTEncoderLayer(config) for _ in range(config.encoder_layers)]
        )  # type: List[EncoderLayer]
        self.pre_layernorm = config.encoder_pre_layernorm
        if self.pre_layernorm:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Args:
            input_ids (`torch.LongTensor`): tokens in the source language of shape
                *(batch, src_len)*
            attention_mask (`torch.LongTensor`): indicating which indices are padding tokens
            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        Returns:
            BaseModelOutput or Tuple comprised of:
                - **x** (`torch.Tensor`): the last encoder layer's output of shape *(src_len, batch, embed_dim)*
                - **encoder_states** (`Tuple(torch.FloatTensor`)): all intermediate hidden states of shape *(src_len,
                  batch, embed_dim)*. Only populated if *output_hidden_states:* is True.
                - **all_attentions** (`Tuple(torch.FloatTensor`)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                x = x.transpose(0, 1)  # T x B x C -> B x T x C
                encoder_states += (x,)
                x = x.transpose(0, 1)  # B x T x C -> T x B x C
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(
                    x,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if output_hidden_states:
            encoder_states += (x,)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class FSMTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DecoderLayer`]
    Args:
        config: FSMTConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        embed_dim = embed_tokens.embedding_dim
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [FSMTDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.pre_layernorm = config.decoder_pre_layernorm
        if self.pre_layernorm:
            self.layer_norm = nn.LayerNorm(embed_dim)

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.embed_tokens.weight, modifier_rank=None):
                embed_tokens_weight_shape = self.embed_tokens.weight.shape
        else:
            embed_tokens_weight_shape = self.embed_tokens.weight.shape
        self.output_projection = nn.Linear(embed_tokens_weight_shape[1], embed_tokens_weight_shape[0], bias=False)
        self.output_projection.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Includes several features from "Jointly Learning to Align and Translate with Transformer Models" (Garg et al.,
        EMNLP 2019).
        Args:
            input_ids (`torch.LongTensor` of shape `(batch, tgt_len)`):
                previous decoder outputs for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            past_key_values (dict or None): dictionary used for storing state during generation
            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape *(batch, tgt_len, embed_dim)*
                - the cache
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids)  # , use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Convert to FSMT output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        next_decoder_cache = []

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                x = x.transpose(0, 1)
                all_hidden_states += (x,)
                x = x.transpose(0, 1)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = past_key_values[idx] if past_key_values is not None else None

            x, layer_self_attn, layer_past, layer_cross_attn = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                output_attentions=output_attentions,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if output_attentions:
                all_self_attns += (layer_self_attn,)
                all_cross_attns += (layer_cross_attn,)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            x = x.transpose(0, 1)
            all_hidden_states += (x,)
            x = x.transpose(0, 1)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        x = self.output_projection(x)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [x, next_cache, all_hidden_states, all_self_attns, all_cross_attns] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )


class FSMTModel(_FSMTModel):
    def __init__(self, config: FSMTConfig):
        super(_FSMTModel, self).__init__(config)

        padding_idx = config.pad_token_id
        encoder_embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx)
        decoder_embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx)

        self.encoder = FSMTEncoder(config, encoder_embed_tokens)
        self.decoder = FSMTDecoder(config, decoder_embed_tokens)

        # Initialize weights and apply final processing
        self.post_init()


class FSMTForConditionalGeneration(_FSMTForConditionalGeneration):
    def __init__(self, config: FSMTConfig):
        super(_FSMTForConditionalGeneration, self).__init__(config)
        self.model = FSMTModel(config)
