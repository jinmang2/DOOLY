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
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
