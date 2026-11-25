from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from src.models.bilstm_crf.vocab import PAD_IDX


class CharCNN(nn.Module):
    """Trích xuất đặc trưng hình thái ký tự bằng Conv1D + Max-Pooling (Ma & Hovy, 2016)."""

    def __init__(
        self,
        char_vocab_size: int,
        char_emb_dim: int = 30,
        num_filters: int = 50,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=PAD_IDX,
        )
        self.dropout = nn.Dropout(dropout)
        # padding = kernel_size // 2 giúp giữ nguyên độ dài W ngay cả với từ ngắn (1-2 ký tự)
        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.out_dim = num_filters

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, T, W)
        batch_size, seq_len, word_len = char_ids.shape
        word_mask = (char_ids != PAD_IDX).any(dim=-1, keepdim=True)  # (B, T, 1)

        flat_chars = char_ids.reshape(batch_size * seq_len, word_len)  # (N, W)
        emb = self.dropout(self.char_embedding(flat_chars))            # (N, W, C_emb)
        emb = emb.transpose(1, 2)                                      # (N, C_emb, W) cho Conv1d

        conv_out = torch.relu(self.conv(emb))                          # (N, num_filters, W)
        pooled = conv_out.max(dim=-1).values                           # (N, num_filters)

        out = pooled.reshape(batch_size, seq_len, self.out_dim)        # (B, T, num_filters)
        return out * word_mask.to(out.dtype)


class CharLSTM(nn.Module):
    """Trích xuất đặc trưng hình thái ký tự bằng BiLSTM cấp ký tự (Lample et al., 2016)."""

    def __init__(
        self,
        char_vocab_size: int,
        char_emb_dim: int = 30,
        char_hidden_dim: int = 50,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=PAD_IDX,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=char_emb_dim,
            hidden_size=char_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = char_hidden_dim * 2

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, T, W)
        batch_size, seq_len, word_len = char_ids.shape
        word_mask = (char_ids != PAD_IDX).any(dim=-1, keepdim=True)  # (B, T, 1)

        flat_chars = char_ids.reshape(batch_size * seq_len, word_len)  # (N, W)
        char_lengths = (flat_chars != PAD_IDX).sum(dim=-1).clamp(min=1)  # Tránh độ dài 0 với token <PAD>

        emb = self.dropout(self.char_embedding(flat_chars))            # (N, W, C_emb)
        packed = pack_padded_sequence(
            emb, char_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)                                # h_n: (2, N, char_hidden_dim)
        # Ghép trạng thái ẩn cuối của chiều xuôi (h_n[0]) và chiều ngược (h_n[1])
        char_repr = torch.cat([h_n[0], h_n[1]], dim=-1)                # (N, 2 * char_hidden_dim)

        out = char_repr.reshape(batch_size, seq_len, self.out_dim)     # (B, T, 2 * char_hidden_dim)
        return out * word_mask.to(out.dtype)


def build_char_encoder(
    model_cfg: Dict, char_vocab_size: int
) -> Tuple[nn.Module, int]:
    """Khởi tạo CharCNN hoặc CharLSTM theo cấu hình YAML và trả về (module, out_dim)."""
    enc_type = model_cfg.get("char_encoder_type", "cnn").lower()
    char_emb_dim = model_cfg.get("char_emb_dim", 30)
    char_hidden_dim = model_cfg.get("char_hidden_dim", 50)
    dropout = model_cfg.get("dropout", 0.5)

    if enc_type == "cnn":
        encoder = CharCNN(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            num_filters=char_hidden_dim,
            kernel_size=model_cfg.get("char_kernel_size", 3),
            dropout=dropout,
        )
    elif enc_type == "lstm":
        encoder = CharLSTM(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"char_encoder_type không hợp lệ: {enc_type!r} (chỉ hỗ trợ 'cnn' hoặc 'lstm').")

    return encoder, encoder.out_dim