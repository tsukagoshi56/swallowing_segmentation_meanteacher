"""
Model definitions for sound event detection
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional, Tuple, List
from transformers import WavLMModel, Wav2Vec2Model, HubertModel, PreTrainedModel

from ddp_utils import is_main_process
from exp_config import EPS

logger = logging.getLogger(__name__)

class EventDetector(nn.Module):
    def __init__(self,
                 config: Dict[str, Any],
                 ssl_model_name: Optional[str],
                 freeze_feature_extractor: bool,
                 freeze_transformer_layers: bool,
                 architecture: str,
                 feature_type: str,
                 local_rank: int = 0):
        super().__init__()
        self.feature_type = feature_type
        self.architecture = architecture
        self.ssl_model: Optional[PreTrainedModel] = None
        input_dimension: int
        self.local_rank = local_rank

        if feature_type == "raw" and ssl_model_name:
            if is_main_process(self.local_rank):
                logger.info(f"Loading SSL model: {ssl_model_name}")
            
            # Load SSL model with proper DDP synchronization
            def load_ssl_model(model_name: str) -> PreTrainedModel:
                """Helper function to load SSL model based on name"""
                if "wavlm" in model_name.lower():
                    return WavLMModel.from_pretrained(model_name)
                elif "wav2vec2" in model_name.lower():
                    return Wav2Vec2Model.from_pretrained(model_name)
                elif "hubert" in model_name.lower():
                    return HubertModel.from_pretrained(model_name)
                else:
                    raise ValueError(f"Unsupported SSL model type in name: {model_name}")
            
            # Ensure SSL model is downloaded/loaded by main process first, then synchronized
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Main process downloads the model first
                if is_main_process(self.local_rank):
                    self.ssl_model = load_ssl_model(ssl_model_name)
                    if is_main_process(self.local_rank):
                        logger.info(f"Main process loaded SSL model with {sum(p.numel() for p in self.ssl_model.parameters())} parameters")
                
                # Wait for main process to complete downloading
                dist.barrier()
                
                # Non-main processes load the model after barrier
                if not is_main_process(self.local_rank):
                    self.ssl_model = load_ssl_model(ssl_model_name)
                    logger.info(f"Rank {self.local_rank} loaded SSL model with {sum(p.numel() for p in self.ssl_model.parameters())} parameters")
                
                # Synchronize again to ensure all processes have loaded the model
                dist.barrier()
            else:
                # Single GPU/CPU case - no synchronization needed
                self.ssl_model = load_ssl_model(ssl_model_name)

            if freeze_feature_extractor:
                if is_main_process(self.local_rank):
                    logger.info("Freezing SSL feature extractor.")
                if hasattr(self.ssl_model, 'feature_extractor'):
                    for param in self.ssl_model.feature_extractor.parameters():
                        param.requires_grad = False
                if hasattr(self.ssl_model, 'feature_projection'):  # For Wav2Vec2
                    for param in self.ssl_model.feature_projection.parameters():
                        param.requires_grad = False
            
            if freeze_transformer_layers:
                if hasattr(self.ssl_model, 'encoder'):
                    if is_main_process(self.local_rank):
                        logger.info("Freezing SSL transformer encoder layers.")
                    for param in self.ssl_model.encoder.parameters():
                        param.requires_grad = False
                else:  # Hubert specific
                    if hasattr(self.ssl_model, 'encoder') and hasattr(self.ssl_model.encoder, 'layers'):
                         if is_main_process(self.local_rank):
                            logger.info("Freezing SSL transformer encoder layers (HuBERT style).")
                         for layer in self.ssl_model.encoder.layers:
                            for param in layer.parameters():
                                param.requires_grad = False
                    elif is_main_process(self.local_rank):
                        logger.warning(f"Model {ssl_model_name} does not seem to have standard 'encoder' or 'encoder.layers' for freezing.")

            input_dimension = self.ssl_model.config.hidden_size

        elif feature_type == "mfcc":
            input_dimension = 40
            if is_main_process(self.local_rank): logger.info("Using MFCC features (dim=40).")
        elif feature_type == "mel":
            input_dimension = 128
            if is_main_process(self.local_rank): logger.info("Using Mel Spectrogram features (dim=128).")
        elif feature_type == "raw" and ssl_model_name is None:
            if is_main_process(self.local_rank): logger.info("Using raw audio directly without SSL frontend.")
            input_dimension = 1
        else:
            raise ValueError(f"Invalid feature_type/ssl_model_name combination: {feature_type}, {ssl_model_name}")

        if architecture in ["gru", "lstm", "fc"]:
            self.projection = nn.Linear(input_dimension, 256)
            self.activation = nn.ReLU()
            self.rnn_or_fc: Optional[nn.Module] = None
            rnn_hidden_size = 128
            if architecture == "gru":
                if is_main_process(self.local_rank): logger.info("Using GRU backend.")
                self.rnn_or_fc = nn.GRU(input_size=256, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
                output_dimension = rnn_hidden_size * 2
            elif architecture == "lstm":
                if is_main_process(self.local_rank): logger.info("Using LSTM backend.")
                self.rnn_or_fc = nn.LSTM(input_size=256, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
                output_dimension = rnn_hidden_size * 2
            elif architecture == "fc":
                if is_main_process(self.local_rank): logger.info("Using FC backend (applied frame-wise).")
                self.rnn_or_fc = None  # Projection output is used directly
                output_dimension = 256
            self.dropout = nn.Dropout(0.3)
            # 動的クラス数の出力層（3クラス or 4クラス+blank対応）
            num_classes = len(config["classes"])
            self.output_layer = nn.Linear(output_dimension, num_classes)

        elif architecture == "crnn":
            if is_main_process(self.local_rank): logger.info("Using CRNN backend (with SSL features).")
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dimension, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            rnn_input_size = 128
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.dropout = nn.Dropout(0.3)
            # 動的クラス数の出力層（3クラス or 4クラス+blank対応）
            num_classes = len(config["classes"])
            self.output_layer = nn.Linear(256, num_classes)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        elif architecture == "crnn_direct":
            if is_main_process(self.local_rank): logger.info(f"Using CRNN_DIRECT backend (directly from {feature_type}).")
            if feature_type == "raw":
                self.raw_frontend = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38),  # Adjusted padding for stride 4
                    nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                    nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4)
                )
                cnn_input_dim = 128
            else:  # MFCC/Mel
                cnn_input_dim = input_dimension
            
            self.cnn = nn.Sequential(
                nn.Conv1d(cnn_input_dim, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.dropout = nn.Dropout(0.3)
            # 動的クラス数の出力層（3クラス or 4クラス+blank対応）
            num_classes = len(config["classes"])
            self.output_layer = nn.Linear(256, num_classes)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ssl_model and self.feature_type == "raw":
            x = x.to(torch.float32)  # SSL models expect float32
            # WavLM etc. expect raw waveform (B, T_samples)
            if x.dim() == 2:  # If (B, T_samples)
                 # Some SSL models might internally handle normalization. Verify if needed.
                 # x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + EPS)
                 pass
            elif x.dim() == 3 and x.size(1) == 1:  # If (B, 1, T_samples)
                x = x.squeeze(1)
            else:  # Should not happen if collate is correct for raw audio
                # This might need adjustment based on how raw audio is batched by pad_collate
                pass
            output = self.ssl_model(x)
            h = output.last_hidden_state
        elif self.feature_type == "raw" and self.architecture == "crnn_direct":
            x = x.unsqueeze(1)  # (B, 1, T_wav)
            h = self.raw_frontend(x)  # (B, 128, T_feat)
            # Note: raw_frontend output h is (B, C_out, T_feat_raw).
            # This h will then go to self.cnn which expects (B, F, T) or (B, C, T)
        else:  # MFCC/Mel features, or raw features for GRU/LSTM/FC (if input_dimension=1 for raw)
            h = x  # (B, T, F) for MFCC/Mel from dataloader, or (B,T,1) for raw (non-crnn_direct)

        if self.architecture in ["crnn", "crnn_direct"]:
            # Input to cnn should be (B, Channels/Features, Time)
            if not (hasattr(self, "raw_frontend") and self.feature_type == "raw" and self.architecture == "crnn_direct"):
                 # If not raw_frontend output (which is already B,C,T), then transpose
                 # SSL output h is (B, T_feat, hidden_size)
                 # MFCC/Mel input x is (B, T_feat, feature_dim)
                 h = h.transpose(1, 2)  # (B, F, T)
            
            if hasattr(self, "cnn"):  # All CRNN types have self.cnn
                 h = self.cnn(h)      # (B, 128, T_cnn)
            
            h = h.transpose(1, 2)  # (B, T_cnn, 128) for RNN
            self.rnn.flatten_parameters()
            h, _ = self.rnn(h)     # (B, T_cnn, 256)
            h = self.dropout(h)
            logits = self.output_layer(h)  # (B, T_cnn, C)
            logits = logits.transpose(1, 2)  # (B, C, T_cnn)
            logits = self.upsample(logits)   # (B, C, T_original_ish)
            return logits
        else:  # GRU, LSTM, FC
            # Input h is (B, T, F)
            if hasattr(self, "projection"):
                h = self.activation(self.projection(h))  # (B, T, 256)

            if isinstance(self.rnn_or_fc, (nn.GRU, nn.LSTM)):
                self.rnn_or_fc.flatten_parameters()
                h, _ = self.rnn_or_fc(h)
            # If FC, h is already (B, T, 256) from projection
            
            h = self.dropout(h)
            logits = self.output_layer(h)  # (B, T, C)
            return logits.permute(0, 2, 1)  # (B, C, T)
