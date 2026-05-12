from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


def _frame_audio(audio: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"_frame_audio expects (B, T), got {audio.shape}")
    if audio.shape[1] < frame_length:
        pad = frame_length - audio.shape[1]
        audio = F.pad(audio, (0, pad))
    return audio.unfold(1, frame_length, hop_length)


def _resample_sequence(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    if seq.shape[1] == target_len:
        return seq
    seq_t = seq.transpose(1, 2)
    seq_t = F.interpolate(seq_t, size=target_len, mode="linear", align_corners=False)
    return seq_t.transpose(1, 2)


class VoiceprintBranch(nn.Module):
    def __init__(self, seq_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=321, stride=32, padding=160),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True, dropout=0.1)
        self.proj = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, seq_dim),
            nn.GELU(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.conv(audio).transpose(1, 2)
        x, _ = self.attn(x, x, x, need_weights=False)
        return self.proj(x)


class EmotionBranch(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        seq_dim: int = 128,
        transformer_dim: int = 128,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer(
            "mel_basis",
            torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)).float(),
        )
        self.input_proj = nn.Linear(n_mels, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=4,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, seq_dim),
            nn.GELU(),
        )

    def _log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        waveform = audio.squeeze(1)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2)
        mel = torch.einsum("mf,bft->bmt", self.mel_basis.to(audio.device), power)
        return torch.log(mel.clamp_min(1e-6)).transpose(1, 2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self._log_mel(audio)
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.proj(x)


class SequenceFeatureBranch(nn.Module):
    def __init__(self, input_dim: int, seq_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, seq_dim),
            nn.GELU(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(seq)
        return self.proj(x)


class PerspectivePooler(nn.Module):
    def __init__(self, seq_dim: int, out_dim: int = 256):
        super().__init__()
        self.attn = nn.Linear(seq_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(seq).squeeze(-1), dim=1)
        pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return self.proj(pooled)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        n_nodes = h.shape[1]
        hi = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        hj = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        logits = self.attn(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        logits = self.activation(logits)
        mask = adjacency.to(dtype=torch.bool, device=x.device).unsqueeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, h)


class GATFusion(nn.Module):
    def __init__(self, node_dim: int = 128, hidden_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.gat1(nodes, adjacency)
        x = self.gat2(x, adjacency)
        pooled = x.mean(dim=1)
        return self.head(pooled)


@dataclass
class DMPFConfig:
    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    common_time_steps: int = 64
    perspective_seq_dim: int = 128
    perspective_out_dim: int = 256
    common_dim: int = 128
    fusion_hidden_dim: int = 64
    fusion_out_dim: int = 16
    lambda_rec: float = 0.4
    lambda_align: float = 0.6
    lambda_branch: float = 0.7


class DMPFRegressor(nn.Module):
    """
    Practical regression-oriented reproduction of DMPF.

    It keeps the paper's main ideas:
    1. five interpretable acoustic perspectives,
    2. decoupled common/private feature learning,
    3. graph-attention fusion across perspectives.
    """

    perspective_names = ("voiceprint", "emotion", "pause", "energy", "tremor")

    def __init__(self, config: DMPFConfig | None = None):
        super().__init__()
        self.config = config or DMPFConfig()
        self.feature_dim = self.config.perspective_seq_dim * len(self.perspective_names)
        self.frame_length = int(round(self.config.sample_rate * self.config.frame_length_ms / 1000.0))
        self.hop_length = int(round(self.config.sample_rate * self.config.hop_length_ms / 1000.0))

        self.voiceprint_branch = VoiceprintBranch(seq_dim=self.config.perspective_seq_dim)
        self.emotion_branch = EmotionBranch(
            sample_rate=self.config.sample_rate,
            n_fft=400,
            win_length=self.frame_length,
            hop_length=self.hop_length,
            n_mels=128,
            seq_dim=self.config.perspective_seq_dim,
        )
        self.pause_branch = SequenceFeatureBranch(input_dim=6, seq_dim=self.config.perspective_seq_dim)
        self.energy_branch = SequenceFeatureBranch(input_dim=2, seq_dim=self.config.perspective_seq_dim)
        self.tremor_branch = SequenceFeatureBranch(input_dim=2, seq_dim=self.config.perspective_seq_dim)

        self.poolers = nn.ModuleDict(
            {
                name: PerspectivePooler(self.config.perspective_seq_dim, self.config.perspective_out_dim)
                for name in self.perspective_names
            }
        )
        self.branch_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(self.config.perspective_out_dim),
                    nn.Linear(self.config.perspective_out_dim, 64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )
                for name in self.perspective_names
            }
        )

        self.common_encoder = nn.Sequential(
            nn.Linear(self.config.perspective_out_dim, self.config.common_dim),
            nn.GELU(),
        )
        self.private_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.config.perspective_out_dim, self.config.common_dim),
                    nn.GELU(),
                )
                for name in self.perspective_names
            }
        )
        self.recovery_decoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.config.common_dim * 2, self.config.perspective_out_dim),
                    nn.GELU(),
                    nn.Linear(self.config.perspective_out_dim, self.config.perspective_out_dim),
                )
                for name in self.perspective_names
            }
        )

        common_adj = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        private_adj = torch.ones(5, 5, dtype=torch.float32)
        self.register_buffer("common_adjacency", common_adj)
        self.register_buffer("private_adjacency", private_adj)
        self.common_gat = GATFusion(
            node_dim=self.config.common_dim,
            hidden_dim=self.config.fusion_hidden_dim,
            out_dim=self.config.fusion_out_dim,
        )
        self.private_gat = GATFusion(
            node_dim=self.config.common_dim,
            hidden_dim=self.config.fusion_hidden_dim,
            out_dim=self.config.fusion_out_dim,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.config.fusion_out_dim * 2),
            nn.Linear(self.config.fusion_out_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def _energy_and_pause_sequences(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = _frame_audio(waveform, self.frame_length, self.hop_length)
        short_energy = frames.pow(2).mean(dim=-1)
        teo = frames[..., 1:-1].pow(2) - frames[..., :-2] * frames[..., 2:]
        teo = teo.mean(dim=-1)

        energy_seq = torch.stack(
            [
                torch.log1p(short_energy.clamp_min(0.0)),
                torch.log1p(teo.abs()),
            ],
            dim=-1,
        )

        energy_norm = short_energy / short_energy.mean(dim=1, keepdim=True).clamp_min(1e-6)
        silence_prob = torch.sigmoid((0.5 - energy_norm) * 8.0)
        voice_prob = 1.0 - silence_prob
        pause_count = F.relu(silence_prob[:, 1:] - silence_prob[:, :-1])
        pause_count = F.pad(pause_count, (1, 0))
        pause_count = torch.cumsum(pause_count, dim=1) / max(1, silence_prob.shape[1])
        pause_ratio = torch.cumsum(silence_prob, dim=1) / torch.arange(
            1,
            silence_prob.shape[1] + 1,
            device=waveform.device,
            dtype=waveform.dtype,
        ).view(1, -1)
        voice_pause_ratio = torch.cumsum(voice_prob, dim=1) / torch.cumsum(silence_prob + 1e-3, dim=1)
        running_mean = torch.cumsum(silence_prob, dim=1) / torch.arange(
            1,
            silence_prob.shape[1] + 1,
            device=waveform.device,
            dtype=waveform.dtype,
        ).view(1, -1)
        running_sq_mean = torch.cumsum(silence_prob.pow(2), dim=1) / torch.arange(
            1,
            silence_prob.shape[1] + 1,
            device=waveform.device,
            dtype=waveform.dtype,
        ).view(1, -1)
        running_std = (running_sq_mean - running_mean.pow(2)).clamp_min(0.0).sqrt()
        duration_feat = torch.linspace(
            0.0,
            1.0,
            silence_prob.shape[1],
            device=waveform.device,
            dtype=waveform.dtype,
        ).view(1, -1).expand_as(silence_prob)

        pause_seq = torch.stack(
            [
                duration_feat,
                silence_prob,
                pause_count,
                running_std,
                pause_ratio,
                torch.log1p(voice_pause_ratio),
            ],
            dim=-1,
        )
        return energy_seq, pause_seq

    def _tremor_sequence(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = _frame_audio(waveform, self.frame_length, self.hop_length)
        window = torch.hann_window(self.frame_length, device=waveform.device)
        frames = frames * window.view(1, 1, -1)
        spec = torch.fft.rfft(frames, dim=-1).abs()
        freqs = torch.fft.rfftfreq(self.frame_length, d=1.0 / self.config.sample_rate).to(waveform.device)
        valid = (freqs >= 50.0) & (freqs <= 400.0)
        valid_spec = spec[..., valid]
        valid_freqs = freqs[valid]
        peak_idx = valid_spec.argmax(dim=-1)
        peak_freq = valid_freqs[peak_idx]
        peak_amp = valid_spec.max(dim=-1).values
        return torch.stack(
            [
                peak_freq / 400.0,
                torch.log1p(peak_amp),
            ],
            dim=-1,
        )

    def _extract_branch_sequences(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        waveform = audio.squeeze(1)
        voice_seq = self.voiceprint_branch(audio)
        emotion_seq = self.emotion_branch(audio)
        energy_seq_raw, pause_seq_raw = self._energy_and_pause_sequences(waveform)
        tremor_seq_raw = self._tremor_sequence(waveform)

        seqs = {
            "voiceprint": voice_seq,
            "emotion": emotion_seq,
            "pause": self.pause_branch(pause_seq_raw),
            "energy": self.energy_branch(energy_seq_raw),
            "tremor": self.tremor_branch(tremor_seq_raw),
        }
        return {
            name: _resample_sequence(seq, self.config.common_time_steps)
            for name, seq in seqs.items()
        }

    def _build_sequence_features(self, branch_sequences: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([branch_sequences[name] for name in self.perspective_names], dim=-1)

    def _split_sequence_features(self, sequence_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        chunks = torch.chunk(sequence_features, chunks=len(self.perspective_names), dim=-1)
        return {name: chunk for name, chunk in zip(self.perspective_names, chunks)}

    def _pool_vectors(self, branch_sequences: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: self.poolers[name](branch_sequences[name])
            for name in self.perspective_names
        }

    def _decouple(self, branch_vectors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        common = []
        private = []
        rec_losses = []
        ortho_terms = []

        for name in self.perspective_names:
            h = branch_vectors[name]
            hco = self.common_encoder(h)
            hpr = self.private_encoders[name](h)
            recovered = self.recovery_decoders[name](torch.cat([hco, hpr], dim=-1))
            rec_losses.append(F.mse_loss(recovered, h))
            ortho_terms.append(F.relu(F.cosine_similarity(hco, hpr, dim=-1)).mean())
            common.append(hco)
            private.append(hpr)

        common_nodes = torch.stack(common, dim=1)
        private_nodes = torch.stack(private, dim=1)

        common_align_terms = []
        for i in range(len(self.perspective_names)):
            for j in range(i + 1, len(self.perspective_names)):
                common_align_terms.append(
                    1.0 - F.cosine_similarity(common_nodes[:, i], common_nodes[:, j], dim=-1).mean()
                )

        rec_loss = torch.stack(rec_losses).mean()
        align_loss = torch.stack(ortho_terms + common_align_terms).mean()
        return common_nodes, private_nodes, rec_loss, align_loss

    def _predict_from_branch_sequences(
        self,
        branch_sequences: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        branch_vectors = self._pool_vectors(branch_sequences)
        branch_preds = {
            name: self.branch_heads[name](branch_vectors[name]).squeeze(-1)
            for name in self.perspective_names
        }
        common_nodes, private_nodes, rec_loss, align_loss = self._decouple(branch_vectors)
        h_common = self.common_gat(common_nodes, self.common_adjacency)
        h_private = self.private_gat(private_nodes, self.private_adjacency)
        y_hat = self.output_head(torch.cat([h_common, h_private], dim=-1)).squeeze(-1)
        return {
            "y_hat": y_hat,
            "branch_preds": branch_preds,
            "branch_vectors": branch_vectors,
            "common_nodes": common_nodes,
            "private_nodes": private_nodes,
            "reconstruction_loss": rec_loss,
            "alignment_loss": align_loss,
        }

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        branch_sequences = self._extract_branch_sequences(audio)
        sequence_features = self._build_sequence_features(branch_sequences)
        return sequence_features, {"branch_sequences": branch_sequences}

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        branch_sequences = self._split_sequence_features(sequence_features)
        outputs = self._predict_from_branch_sequences(branch_sequences)
        return outputs["y_hat"]

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        branch_sequences = self._extract_branch_sequences(audio)
        sequence_features = self._build_sequence_features(branch_sequences)
        outputs = self._predict_from_branch_sequences(branch_sequences)
        outputs["sequence_features"] = sequence_features
        return outputs

    def compute_training_loss(self, audio: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = labels.squeeze(-1) if labels.ndim > 1 else labels
        outputs = self.forward(audio)
        fusion_loss = F.mse_loss(outputs["y_hat"], labels)
        branch_loss = torch.stack(
            [F.mse_loss(pred, labels) for pred in outputs["branch_preds"].values()]
        ).mean()
        total_loss = (
            fusion_loss
            + self.config.lambda_rec * outputs["reconstruction_loss"]
            + self.config.lambda_align * outputs["alignment_loss"]
            + self.config.lambda_branch * branch_loss
        )
        return total_loss, outputs["y_hat"]
