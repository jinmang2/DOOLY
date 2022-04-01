import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import (
    ModelOutput,
    TokenClassifierOutput,  # noqa
    BaseModelOutputWithPoolingAndCrossAttentions,  # noqa
)


@dataclass
class DependencyParsingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    classifier_attention: Optional[torch.FloatTensor] = None


@dataclass
class DialogueStateTrackingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    point_outputs: Optional[torch.FloatTensor] = None
    gate_outputs: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
