"""
model.py - Kiến trúc mô hình Transformer Token Classification tùy biến cho PhoBERT & XLM-R.

Hỗ trợ:
  * Nạp pretrained backbone (PhoBERT, XLM-R) qua HuggingFace AutoModel.
  * Các chiến lược pooling/kết hợp các tầng ẩn cuối (Ablation study ở Tuần 4):
      - "last": Tầng ẩn cuối cùng [B, L, H]
      - "concat_4": Ghép nối 4 tầng ẩn cuối [B, L, 4*H]
      - "mean_4": Trung bình cộng 4 tầng ẩn cuối [B, L, H]
      - "weighted": Tổng có trọng số học được của tất cả các tầng ẩn [B, L, H]
  * Chiến lược đóng băng tham số (Freeze embeddings, freeze bottom N encoder layers).
  * Chú thích tensor shape chi tiết tại mọi biến đổi quan trọng theo quy chuẩn dl-engineer.md.
"""
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class PretrainedNERModel(nn.Module):
    """
    Mô hình nhận dạng thực thể tiếng Việt dựa trên Transformer (PhoBERT / XLM-R).
    """

    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base-v2",
        num_labels: int = 9,
        pooling_strategy: str = "last",
        freeze_embeddings: bool = False,
        freeze_encoder_layers: int = 0,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy.lower()

        # Nạp cấu hình và mô hình cơ sở
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.hidden_dim = self.config.hidden_size  # Thường là 768

        # Xác định chiều đầu vào cho classifier dựa trên chiến lược pooling
        if self.pooling_strategy == "concat_4":
            self.classifier_in_dim = self.hidden_dim * 4
        elif self.pooling_strategy in ("last", "mean_4", "weighted"):
            self.classifier_in_dim = self.hidden_dim
        else:
            raise ValueError(f"Chiến lược pooling không hợp lệ: {pooling_strategy}. "
                             f"Chọn trong: ['last', 'concat_4', 'mean_4', 'weighted']")

        # Nếu dùng weighted pooling: Tạo vector trọng số học được cho từng tầng
        if self.pooling_strategy == "weighted":
            num_layers = self.config.num_hidden_layers + 1  # Gồm cả embedding layer
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Đầu phân loại thực thể
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.classifier_in_dim, num_labels)

        # Hàm mất mát CrossEntropyLoss với ignore_index=-100
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Áp dụng chiến lược đóng băng tầng
        self.apply_freeze_strategy(freeze_embeddings, freeze_encoder_layers)

    def apply_freeze_strategy(self, freeze_embeddings: bool, freeze_encoder_layers: int) -> None:
        """Đóng băng embeddings và một số tầng encoder dưới cùng."""
        # 1. Đóng băng embeddings
        if freeze_embeddings:
            embeddings = getattr(self.encoder, "embeddings", None)
            if embeddings is not None:
                for param in embeddings.parameters():
                    param.requires_grad = False

        # 2. Đóng băng N tầng encoder dưới cùng
        if freeze_encoder_layers > 0:
            # RoBERTa / PhoBERT / XLM-R đều có encoder.layer
            encoder_layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            if encoder_layers is not None:
                for i in range(min(freeze_encoder_layers, len(encoder_layers))):
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        labels: Optional[torch.Tensor] = None, # [B, L]
    ) -> Dict[str, Any]:
        """
        Forward pass mô hình Transformer Token Classification.
        
        Args:
            input_ids: Tensor [B, L]
            attention_mask: Tensor [B, L]
            labels: Tensor nhãn mục tiêu [B, L] (hoặc None khi inference)
            
        Returns:
            Dict chứa:
                - "loss": Scalar loss nếu có labels, ngược lại None
                - "logits": Tensor dự đoán [B, L, num_labels]
                - "hidden_states": Tuple các tầng ẩn
                - "attentions": Tuple attention weights từ các tầng
        """
        # [B, L] -> Backbone -> outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        all_hidden_states = outputs.hidden_states  # Tuple của num_layers + 1 tensor [B, L, H]

        # Áp dụng chiến lược pooling tầng ẩn
        if self.pooling_strategy == "last":
            # Lấy tầng ẩn cuối cùng
            features = all_hidden_states[-1]  # [B, L, H]

        elif self.pooling_strategy == "concat_4":
            # Ghép nối 4 tầng ẩn cuối
            # all_hidden_states[-4:] mỗi phần tử là [B, L, H]
            features = torch.cat(all_hidden_states[-4:], dim=-1)  # [B, L, 4 * H]

        elif self.pooling_strategy == "mean_4":
            # Trung bình cộng 4 tầng ẩn cuối
            stacked = torch.stack(all_hidden_states[-4:], dim=0)  # [4, B, L, H]
            features = torch.mean(stacked, dim=0)                 # [B, L, H]

        elif self.pooling_strategy == "weighted":
            # Tổng có trọng số softmax của tất cả các tầng ẩn
            normalized_weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers]
            stacked = torch.stack(all_hidden_states, dim=0)                # [num_layers, B, L, H]
            # [num_layers, 1, 1, 1] * [num_layers, B, L, H] -> sum -> [B, L, H]
            weight_view = normalized_weights.view(-1, 1, 1, 1)            # [num_layers, 1, 1, 1]
            features = torch.sum(stacked * weight_view, dim=0)             # [B, L, H]

        # Đi qua tầng Dropout và Linear Classifier
        features = self.dropout(features)            # [B, L, classifier_in_dim]
        logits = self.classifier(features)           # [B, L, num_labels]

        loss = None
        if labels is not None:
            # Flatten để đưa vào CrossEntropyLoss
            # logits: [B, L, num_labels] -> [B * L, num_labels]
            # labels: [B, L] -> [B * L]
            active_logits = logits.view(-1, self.num_labels)  # [B * L, num_labels]
            active_labels = labels.view(-1)                   # [B * L]
            loss = self.loss_fn(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits,                                 # [B, L, num_labels]
            "hidden_states": all_hidden_states,
            "attentions": outputs.attentions,
        }
