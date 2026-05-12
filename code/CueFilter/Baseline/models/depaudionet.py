from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepAudioNetConfig:
    """
    Unified DepAudioNet configuration.

    The original project used a compact sequence representation obtained by
    compressing waveform segments to a short temporal sequence. To keep the
    unified experiment runner on raw audio while preserving the original
    CNN-LSTM prediction logic, this implementation performs that projection
    inside the model.
    """

    input_features: int = 1
    sequence_length: int = 100
    cnn_channels: Optional[Tuple[int, ...]] = None
    cnn_kernel_sizes: Optional[Tuple[int, ...]] = None
    cnn_stride: int = 1
    cnn_padding: int = 1
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    fc_hidden_sizes: Optional[Tuple[int, ...]] = None
    dropout_rate: float = 0.5
    num_classes: int = 5

    def __post_init__(self) -> None:
        if self.cnn_channels is None:
            self.cnn_channels = (32, 64) if self.input_features >= 10 else (16, 32)
        if self.cnn_kernel_sizes is None:
            if self.input_features < 10:
                self.cnn_kernel_sizes = (1, 1)
            elif self.input_features < 30:
                self.cnn_kernel_sizes = (2, 2)
            else:
                self.cnn_kernel_sizes = (3, 3)
        if self.fc_hidden_sizes is None:
            self.fc_hidden_sizes = (128, 64)
        if self.input_features < 10:
            self.cnn_padding = 0


class DepAudioNet(nn.Module):
    """
    Self-contained DepAudioNet implementation for the unified experiment stack.

    Raw audio input is first adaptively compressed to a short sequence and then
    processed with the original 1D-CNN + LSTM prediction pathway.
    """

    def __init__(self, config: DepAudioNetConfig):
        super().__init__()
        self.config = config
        self.cnn_layers = self._build_cnn_layers()
        self.feature_dim = self._calculate_cnn_output_size()
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_layers = self._build_fc_layers()
        self.regression_head = nn.Linear(config.fc_hidden_sizes[-1], 1)
        self.classification_head = nn.Linear(config.fc_hidden_sizes[-1], config.num_classes)
        self._initialize_weights()

    def _build_cnn_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        in_channels = 1
        current_feature_size = self.config.input_features

        for i, out_channels in enumerate(self.config.cnn_channels):
            if current_feature_size < 1:
                break

            kernel_size = int(min(self.config.cnn_kernel_sizes[i], max(1, current_feature_size)))
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=self.config.cnn_stride,
                padding=self.config.cnn_padding,
            )
            bn = nn.BatchNorm1d(out_channels)

            conv_output_size = (
                current_feature_size
                + 2 * self.config.cnn_padding
                - (kernel_size - 1)
            )

            if conv_output_size >= 2:
                pool_kernel = min(2, conv_output_size)
                pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
                layers.append(nn.Sequential(conv, bn, nn.ReLU(), pool))
                current_feature_size = max(1, conv_output_size // pool_kernel)
            else:
                layers.append(nn.Sequential(conv, bn, nn.ReLU()))
                current_feature_size = max(1, conv_output_size)

            in_channels = out_channels

        return layers

    def _calculate_cnn_output_size(self) -> int:
        dummy_input = torch.randn(2, 1, self.config.input_features)
        with torch.no_grad():
            x = dummy_input
            for layer in self.cnn_layers:
                x = layer(x)
            return int(x.shape[1] * x.shape[2])

    def _build_fc_layers(self) -> nn.Sequential:
        layers = []
        input_size = self.config.lstm_hidden_size

        for hidden_size in self.config.fc_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout_rate),
                ]
            )
            input_size = hidden_size

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_normal_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def _prepare_feature_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert either raw waveform or precomputed compact features to the
        [B, seq_len, input_features] format expected by the temporal encoder.
        """
        if x.dim() == 3 and x.shape[1] == self.config.sequence_length and x.shape[2] == self.config.input_features:
            return x

        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"DepAudioNet expects raw audio [B,1,T] or compact features [B,S,F], got {tuple(x.shape)}")

        if x.shape[1] != 1:
            raise ValueError(f"DepAudioNet raw audio input must be mono [B,1,T], got {tuple(x.shape)}")

        target_length = self.config.sequence_length * self.config.input_features
        pooled = F.adaptive_avg_pool1d(x, target_length).squeeze(1)
        return pooled.view(x.shape[0], self.config.sequence_length, self.config.input_features)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.config.input_features:
            raise ValueError(
                f"Expected input feature size {self.config.input_features}, got {input_dim}"
            )

        x = x.reshape(batch_size * seq_len, 1, input_dim)
        for layer in self.cnn_layers:
            x = layer(x)

        if x.shape[2] == 0:
            x = torch.zeros(batch_size * seq_len, x.shape[1], 1, device=x.device, dtype=x.dtype)

        return x.reshape(batch_size, seq_len, -1)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        feature_sequence = self._prepare_feature_sequence(audio)
        return self.encode_sequence(feature_sequence)

    def predict_from_sequence(
        self,
        sequence_features: torch.Tensor,
        return_dict: bool = False,
        return_features: bool = False,
    ):
        lstm_out, _state = self.lstm(sequence_features)
        final_hidden = lstm_out[:, -1, :]
        features = self.fc_layers(final_hidden)
        regression_output = self.regression_head(features).squeeze(-1)
        classification_output = self.classification_head(features)

        if return_dict or return_features:
            outputs: Dict[str, torch.Tensor] = {
                "regression_output": regression_output.unsqueeze(-1),
                "classification_output": classification_output,
            }
            if return_features:
                outputs["features"] = features
                outputs["sequence_features"] = sequence_features
            return outputs

        return regression_output

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_dict: bool = False,
    ):
        sequence_features = self.encode(x)
        return self.predict_from_sequence(
            sequence_features,
            return_dict=return_dict,
            return_features=return_features,
        )

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sequence_features = self.encode(x)
            lstm_out, _state = self.lstm(sequence_features)
            return torch.softmax(torch.sum(lstm_out.pow(2), dim=2), dim=1)


class DepAudioNetBackbone(DepAudioNet):
    """
    Canonical regression backbone used by the unified baseline and mitigation
    runners. It keeps the full DepAudioNet implementation but defaults to the
    raw-audio compact-sequence setting used in this repository.
    """

    def __init__(self, config: Optional[DepAudioNetConfig] = None):
        super().__init__(config or DepAudioNetConfig(input_features=1, sequence_length=100))


class DepAudioNetLoss(nn.Module):
    def __init__(self, regression_weight: float = 1.0, classification_weight: float = 1.0):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        regression_loss = self.mse_loss(
            outputs["regression_output"].squeeze(-1),
            targets["regression_target"].float(),
        )
        classification_loss = self.ce_loss(
            outputs["classification_output"],
            targets["classification_target"].long(),
        )
        total_loss = (
            self.regression_weight * regression_loss
            + self.classification_weight * classification_loss
        )
        return total_loss, {
            "regression_loss": float(regression_loss.item()),
            "classification_loss": float(classification_loss.item()),
            "total_loss": float(total_loss.item()),
        }


def create_depaudionet_model(config: Optional[DepAudioNetConfig] = None) -> DepAudioNet:
    return DepAudioNet(config or DepAudioNetConfig())


__all__ = [
    "DepAudioNet",
    "DepAudioNetBackbone",
    "DepAudioNetConfig",
    "DepAudioNetLoss",
    "create_depaudionet_model",
]
