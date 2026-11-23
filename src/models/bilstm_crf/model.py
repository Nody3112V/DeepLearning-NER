from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models.bilstm_crf.vocab import PAD_IDX


class BiLSTMCRF(nn.Module):
    """Kiến trúc BiLSTM cơ sở (Tuần 2) có sẵn cổng tích hợp CharEncoder và CRF (Tuần 3)."""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        word_emb_dim: int = 100,
        bilstm_hidden_dim: int = 256,
        bilstm_num_layers: int = 1,
        dropout: float = 0.5,
        char_encoder: Optional[nn.Module] = None,
        char_out_dim: int = 0,
        crf: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if bilstm_hidden_dim % 2 != 0:
            raise ValueError("bilstm_hidden_dim phải là số chẵn (chia đều cho 2 chiều).")

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_emb_dim,
            padding_idx=PAD_IDX,
        )
        self.char_encoder = char_encoder
        self.crf = crf
        self.num_tags = num_tags

        lstm_input_dim = word_emb_dim + (char_out_dim if char_encoder is not None else 0)
        self.input_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=bilstm_hidden_dim // 2,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(bilstm_hidden_dim, num_tags)

    def get_emissions(
        self,
        word_ids: torch.Tensor,
        lengths: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính ma trận điểm phát xạ (emission scores) có kích thước (B, T, num_tags)."""
        max_seq_len = word_ids.size(1)
        word_emb = self.word_embedding(word_ids)  # (B, T, word_emb_dim)

        if self.char_encoder is not None and char_ids is not None:
            char_emb = self.char_encoder(char_ids)  # (B, T, char_out_dim)
            x = torch.cat([word_emb, char_emb], dim=-1)
        else:
            x = word_emb

        x = self.input_dropout(x)

        # Nén chuỗi để BiLSTM bỏ qua hoàn toàn các vị trí <PAD> ở cả chiều xuôi và ngược
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_seq_len
        )

        lstm_out = self.output_dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)  # (B, T, num_tags)
        return emissions

    def compute_loss(
        self,
        word_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính hàm mất mát: dùng NLL của CRF nếu có, ngược lại dùng Masked CrossEntropyLoss."""
        emissions = self.get_emissions(word_ids, lengths, char_ids=char_ids)

        if self.crf is not None:
            return self.crf(emissions, tag_ids, mask)

        # Masked CrossEntropyLoss: gán -100 cho mọi vị trí padding (~mask) để bỏ qua khi tính loss
        masked_tags = tag_ids.masked_fill(~mask, -100)
        loss = F.cross_entropy(
            emissions.reshape(-1, self.num_tags),
            masked_tags.reshape(-1),
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def decode(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Giải mã chuỗi nhãn dự đoán cho từng câu (loại bỏ phần padding)."""
        emissions = self.get_emissions(word_ids, lengths, char_ids=char_ids)

        if self.crf is not None:
            return self.crf.decode(emissions, mask)

        pred_ids = emissions.argmax(dim=-1)  # (B, T)
        results: List[List[int]] = []
        for i, seq_len in enumerate(lengths.tolist()):
            results.append(pred_ids[i, :seq_len].tolist())
        return results
