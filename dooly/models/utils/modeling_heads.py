from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn


class SpanPredictionHead(nn.Module):
    """Head for span prediction tasks.
    Can be viewed as a 2-class output layer that is applied to every position.
    """

    def __init__(self, config):
        assert config.num_labels == 2
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.span_head_inner_dim)
        self.dropout = nn.Dropout(p=config.span_head_dropout)
        self.out_proj = nn.Linear(config.span_head_inner_dim, config.num_labels)

    def forward(self, features, **kwargs):
        x = features  # take features across ALL positions
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x  # B x T x C, but softmax should be taken over T


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DependencyParseHead(nn.Module):
    """Head for sequence tagging tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.head_attn_pre = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.classifier_num_attention_heads,
            dropout=classifier_dropout,
        )
        self.head_attn_post = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=1,
            dropout=classifier_dropout,
        )

    def forward(self, features, masks=None, **kwargs):
        x = features
        x = self.dropout(x)

        x2 = x.permute(1, 0, 2)
        # https://github.com/kakaobrain/pororo/blob/master/pororo/models/brainbert/PoSLaBERTa.py#L110
        # deprecated! attention_mask vs masks
        # attention_mask = [[1,1,1,1],[1,1,1,0]]
        # masks = [[0,0,0,0],[0,0,0,1]]
        masks = (masks == 0).to(torch.bool)
        x3, _ = self.head_attn_pre(x2, x2, x2, key_padding_mask=masks)
        # x3 = [max_len, bsz, hidden_dim]
        _, attn = self.head_attn_post(x3, x3, x3, key_padding_mask=masks)
        # attn = [bsz, max_len, max_len]

        x = self.dense(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        label = self.out_proj(x)

        return attn, label


class SlotGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.parallel_decoding = config.parallel_decoding

        self.embed = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_token_id
        )  # shared with encoder

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )

        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2, "yes":3, "no": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.w_gate = nn.Linear(self.hidden_size, self.num_gates)

    @property
    def gating2id(self):
        return self._gating2id

    @gating2id.setter
    def gating2id(self, val: Dict[str, int]):
        self._gating2id = val
        self.num_gates = len(self._gating2id.keys())

    def set_slot_idx(self, slot_vocab_idx: List[List[int]]):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_token_id] * gap)
            whole.append(idx)
        self.slot_embed_idx: List[List[int]] = whole

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        hidden: torch.Tensor,
        input_masks: torch.Tensor,
        max_len: int,
        teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.slot_embed_idx is not None, (
            "`slot_embed_idx` is required for forward pass. Use `set_slot_idx` method."
        )

        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)
        # slot_embedding
        slot_e = torch.sum(self.embed(slot), 1)  # J, d
        J = slot_e.size(0)

        if self.parallel_decoding:
            all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(input_ids.device)
            all_gate_outputs = torch.zeros(batch_size, J, self.num_gates).to(input_ids.device)

            w = slot_e.repeat(batch_size, 1).unsqueeze(1)
            hidden = hidden.repeat_interleave(J, dim=1)
            encoder_output = encoder_output.repeat_interleave(J, dim=0)
            input_ids = input_ids.repeat_interleave(J, dim=0)
            input_masks = input_masks.repeat_interleave(J, dim=0)
            num_decoding = 1

        else:
            # Seperate Decoding
            all_point_outputs = torch.zeros(J, batch_size, max_len, self.vocab_size).to(input_ids.device)
            all_gate_outputs = torch.zeros(J, batch_size, self.num_gates).to(input_ids.device)
            num_decoding = J

        for j in range(num_decoding):

            if not self.parallel_decoding:
                w = slot_e[j].expand(batch_size, 1, self.hidden_size)

            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D

                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                MASKED_VALUE = (2 ** 15) if attn_e.dtype == torch.half else 1e9
                attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -MASKED_VALUE)
                attn_history = torch.nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = torch.nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
                p_gen = torch.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
                p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)

                if teacher is not None:
                    if self.parallel_decoding:
                        w = self.embed(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
                    else:
                        w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D

                if k == 0:
                    gated_logit = self.w_gate(context.squeeze(1))  # B,3
                    if self.parallel_decoding:
                        all_gate_outputs = gated_logit.view(batch_size, J, self.num_gates)
                    else:
                        _, gated = gated_logit.max(1)  # maybe `-1` would be more clear
                        all_gate_outputs[j] = gated_logit

                if self.parallel_decoding:
                    all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)
                else:
                    all_point_outputs[j, :, k, :] = p_final

        if not self.parallel_decoding:
            all_point_outputs = all_point_outputs.transpose(0, 1)
            all_gate_outputs = all_gate_outputs.transpose(0, 1)

        return all_point_outputs, all_gate_outputs
