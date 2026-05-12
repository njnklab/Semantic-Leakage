import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio.transforms as T  # 未使用，移除以避免版本冲突
import numpy as np
import math
from typing import Tuple, Optional


class GatedResidualBlock(nn.Module):
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.swish = nn.SiLU()
        
    def forward(self, x):
        residual = x
        
        # 因果卷积 + 扩张卷积
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        
        # 门控机制
        gate = torch.sigmoid(self.gate_conv(x))
        x = x * gate
        
        # 残差连接
        return x + self.residual_conv(residual)


class VQWTNet(nn.Module):
    
    def __init__(self, input_dim: int = 1, output_dim: int = 512):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            # 第一层：模拟wav2vec的7层CNN
            nn.Conv1d(input_dim, 512, kernel_size=10, stride=5, padding=3),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(512, output_dim, kernel_size=2, stride=2, padding=0),
        ])
        
        # LayerNorm层（1D格式）
        self.layer_norms = nn.ModuleList([
            nn.GroupNorm(1, 512),
            nn.GroupNorm(1, 512),
            nn.GroupNorm(1, 512),
            nn.GroupNorm(1, 512),
            nn.GroupNorm(1, 512),
            nn.GroupNorm(1, 512),
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=2, 
            dim_feedforward=output_dim * 4, 
            activation='relu',
            batch_first=True,
            dropout=0.1,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.positional_encoding = PositionalEncoding(output_dim, max_len=5000)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms + [None])):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = F.gelu(x)
        
        x = x.transpose(1, 2) 
        
        x = self.positional_encoding(x)
        
        x = self.transformer(x)
        
        return x


class PositionalEncoding(nn.Module):
        
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class SFNet(nn.Module):

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        assert input_dim == hidden_dim == 512, "SFNet使用512通道"
        
        self.layer1 = nn.ModuleList([
            GatedResidualBlock(512, dilation=1), 
            GatedResidualBlock(512, dilation=5),
            GatedResidualBlock(512, dilation=9)
        ])

        self.layer2 = nn.ModuleList([
            GatedResidualBlock(512, dilation=1),
            GatedResidualBlock(512, dilation=5), 
            GatedResidualBlock(512, dilation=9)
        ])
        
        self.residual_projection = nn.Conv1d(512, 512, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        residual = x
        
        for block in self.layer1:
            x = block(x)
            
        x = x + self.residual_projection(residual)
        
        for block in self.layer2:
            x = block(x)
            
        x = x.transpose(1, 2)
        
        return x


class HCPCNet(nn.Module):

    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, prediction_steps: int = 1):
        super().__init__()
        assert input_dim == 512
        self.prediction_steps = prediction_steps
        self.hidden_dim = hidden_dim
    
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.context_network = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        self.prediction_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        z = self.encoder(x) 
        context, _ = self.context_network(z) 
        
        if seq_len > self.prediction_steps:
            context_current = context[:, :-self.prediction_steps, :]
            target_future = z[:, self.prediction_steps:, :]
            predictions = self.prediction_network(context_current)
            
            return context, predictions, target_future
        else:
            return context, None, None
    
    def compute_hcpc_loss(self, predictions, targets, negative_samples=None):

        if predictions is None or targets is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        
        batch_size, seq_len, hidden_dim = predictions.shape
        
        predictions_norm = F.normalize(predictions, dim=-1)
        targets_norm = F.normalize(targets, dim=-1)
        
        pos_scores = torch.sum(predictions_norm * targets_norm, dim=-1) / self.temperature

        neg_scores_list = []
        
        for t_offset in [-2, -1, 1, 2]:
            if 0 <= t_offset < seq_len:
                shifted_targets = torch.roll(targets_norm, shifts=t_offset, dims=1)
                neg_score = torch.sum(predictions_norm * shifted_targets, dim=-1) / self.temperature
                neg_scores_list.append(neg_score.unsqueeze(-1))
        
        for b_offset in range(1, min(batch_size, 8)):
            shifted_targets = torch.roll(targets_norm, shifts=b_offset, dims=0)
            neg_score = torch.sum(predictions_norm * shifted_targets, dim=-1) / self.temperature
            neg_scores_list.append(neg_score.unsqueeze(-1))
        
        if neg_scores_list:
            neg_scores = torch.cat(neg_scores_list, dim=-1)
            
            logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)

            labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                reduction='mean'
            )
        else:
            loss = F.mse_loss(predictions, targets)
        
        return loss


class PredictionNet(nn.Module):
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):

        x = self.fc_layers(x) 
        
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x) 
        x = x.squeeze(-1) 
        
        output = self.regression_head(x)
        
        return output


class STFNModel(nn.Module):
    
    def __init__(
        self, 
        input_dim: int = 1,
        dropout: float = 0.5,
        prediction_steps: int = 1 
    ):
        super().__init__()
        
        vqwt_output_dim = 512
        hcpc_hidden_dim = 256 
        pred_hidden_dim = 128 
        
        self.vqwtnet = VQWTNet(
            input_dim=input_dim, 
            output_dim=vqwt_output_dim
        )
        
        self.sfnet = SFNet(
            input_dim=vqwt_output_dim, 
            hidden_dim=vqwt_output_dim 
        )
        
        self.hcpcnet = HCPCNet(
            input_dim=vqwt_output_dim, 
            hidden_dim=hcpc_hidden_dim,
            prediction_steps=prediction_steps 
        )
        
        self.prediction_net = PredictionNet(
            input_dim=hcpc_hidden_dim, 
            hidden_dim=pred_hidden_dim, 
            dropout=dropout
        )
        
        self.training_config = {
            'batch_size': 20,      
            'max_epochs': 400,     
            'learning_rate': 1e-3,
            'hcpc_weight': 1.0,      
            'audio_length': 3.0    
        }
        
    def forward(self, x):

        x_vqwt = self.vqwtnet(x)
        x_sf = self.sfnet(x_vqwt) 
        context, predictions, targets = self.hcpcnet(x_sf)
        output = self.prediction_net(context) 
        hcpc_loss = self.hcpcnet.compute_hcpc_loss(predictions, targets)
        
        return output, hcpc_loss
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
            return output
            
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_config': self.training_config,
            'model_components': {
                'VQWTNet': sum(p.numel() for p in self.vqwtnet.parameters()),
                'SFNet': sum(p.numel() for p in self.sfnet.parameters()),
                'HCPCNet': sum(p.numel() for p in self.hcpcnet.parameters()),
                'PredictionNet': sum(p.numel() for p in self.prediction_net.parameters())
            }
        }


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 35):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        
    def forward(self, spec):
        if self.training:
            freq_mask_len = torch.randint(0, self.freq_mask_param, (1,)).item()
            freq_mask_start = torch.randint(0, spec.size(-2) - freq_mask_len + 1, (1,)).item()
            spec = spec.clone()
            spec[:, freq_mask_start:freq_mask_start + freq_mask_len, :] = 0
            
            time_mask_len = torch.randint(0, self.time_mask_param, (1,)).item()
            time_mask_start = torch.randint(0, spec.size(-1) - time_mask_len + 1, (1,)).item()
            spec[:, :, time_mask_start:time_mask_start + time_mask_len] = 0
            
        return spec 