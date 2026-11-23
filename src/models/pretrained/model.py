"""
model.py - Kiến trúc mô hình Transformer Token Classification tùy biến cho PhoBERT & XLM-R.

Hỗ trợ đầy đủ:
  * Nạp pretrained backbone (PhoBERT, XLM-R) qua HuggingFace AutoModel hoặc custom backbone.
  * Các chiến lược pooling/kết hợp các tầng ẩn cuối (Ablation study ở Tuần 4):
      - "last": Tầng ẩn cuối cùng [B, L, H]
      - "concat_4": Ghép nối 4 tầng ẩn cuối [B, L, 4*H]
      - "mean_4": Trung bình cộng 4 tầng ẩn cuối [B, L, H]
      - "weighted": Tổng có trọng số softmax học được của tất cả các tầng ẩn [B, L, H]
  * Các chiến lược đóng băng tham số (Freezing strategies):
      - Đóng băng embeddings (freeze_embeddings=True)
      - Đóng băng N tầng encoder dưới cùng (freeze_encoder_layers=N)
      - Đóng băng toàn bộ encoder (freeze_entire_encoder=True), chỉ huấn luyện classifier head
      - Fine-tune toàn bộ encoder (mặc định)
      - Phương thức unfreeze_all() hỗ trợ progressive unfreezing.
  * Thống kê chi tiết số lượng tham số trainable / tổng số tham số (get_trainable_parameters).
  * Khởi tạo linh hoạt từ YAML config qua `PretrainedNERModel.from_config(cfg)`.
  * Chú thích tensor shape chi tiết tại mọi biến đổi theo quy chuẩn dl-engineer.md.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import yaml
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PretrainedConfig


class PretrainedNERModel(nn.Module):
    """
    Mô hình nhận dạng thực thể tiếng Việt dựa trên Transformer (PhoBERT / XLM-R).

    Kiến trúc tổng thể:
        Input: input_ids [B, L], attention_mask [B, L], (token_type_ids [B, L])
          │
          ▼
        Backbone Transformer (PhoBERT / XLM-R)
          │  ───> hidden_states: tuple of (num_layers + 1) tensors, mỗi tensor [B, L, H]
          ▼
        Pooling Module ("last" | "concat_4" | "mean_4" | "weighted")
          │  ───> features: [B, L, classifier_in_dim]
          ▼
        Classifier Dropout (p=classifier_dropout)
          │
          ▼
        Linear Classification Head (classifier_in_dim -> num_labels)
          │  ───> logits: [B, L, num_labels]
          ▼
        CrossEntropyLoss(ignore_index=-100)
    """

    VALID_POOLING_STRATEGIES = ("last", "concat_4", "mean_4", "weighted")

    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base-v2",
        num_labels: int = 9,
        pooling_strategy: str = "last",
        freeze_embeddings: bool = False,
        freeze_encoder_layers: int = 0,
        freeze_entire_encoder: bool = False,
        classifier_dropout: float = 0.2,
        hidden_dropout_prob: Optional[float] = None,
        ignore_index: int = -100,
        config: Optional[PretrainedConfig] = None,
        pretrained_backbone: Optional[nn.Module] = None,
    ):
        """
        Khởi tạo mô hình Transformer Token Classification.

        Args:
            model_name_or_path: Định danh checkpoint Hugging Face (ví dụ 'vinai/phobert-base-v2' hoặc 'xlm-roberta-base').
            num_labels: Số lượng nhãn BIO mục tiêu (ví dụ 9 cho VLSP, 21 cho PhoNER).
            pooling_strategy: Chiến lược pooling tầng ẩn: 'last', 'concat_4', 'mean_4', 'weighted'.
            freeze_embeddings: Nếu True, đóng băng toàn bộ tham số của embedding layer.
            freeze_encoder_layers: Số tầng encoder dưới cùng bị đóng băng (0 đến num_layers).
            freeze_entire_encoder: Nếu True, đóng băng toàn bộ backbone encoder (chỉ train classifier head).
            classifier_dropout: Tỷ lệ dropout trước Linear Classification Head.
            hidden_dropout_prob: Nếu được chỉ định, ghi đè dropout nội tại của backbone encoder.
            ignore_index: Giá trị nhãn bỏ qua khi tính loss (mặc định -100).
            config: Đối tượng PretrainedConfig tùy biến (nếu có).
            pretrained_backbone: Mô hình encoder được khởi tạo sẵn (dùng cho mock/unit test nhanh không cần tải weights).
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy.lower()
        self.freeze_embeddings_flag = freeze_embeddings
        self.freeze_encoder_layers_count = freeze_encoder_layers
        self.freeze_entire_encoder_flag = freeze_entire_encoder
        self.classifier_dropout_prob = classifier_dropout
        self.ignore_index = ignore_index

        if self.pooling_strategy not in self.VALID_POOLING_STRATEGIES:
            raise ValueError(
                f"Chiến lược pooling không hợp lệ: '{pooling_strategy}'. "
                f"Các chiến lược hợp lệ: {list(self.VALID_POOLING_STRATEGIES)}"
            )

        # 1. Cấu hình và Backbone Encoder
        if pretrained_backbone is not None:
            self.encoder = pretrained_backbone
            if config is not None:
                self.config = config
            elif hasattr(pretrained_backbone, "config"):
                self.config = pretrained_backbone.config
            else:
                self.config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    output_hidden_states=True,
                    output_attentions=True,
                )
        else:
            if config is not None:
                self.config = config
            else:
                try:
                    self.config = AutoConfig.from_pretrained(
                        model_name_or_path,
                        output_hidden_states=True,
                        output_attentions=True,
                        local_files_only=True,
                    )
                except Exception:
                    self.config = AutoConfig.from_pretrained(
                        model_name_or_path,
                        output_hidden_states=True,
                        output_attentions=True,
                    )

            if hidden_dropout_prob is not None:
                self.config.hidden_dropout_prob = hidden_dropout_prob
                self.config.attention_probs_dropout_prob = hidden_dropout_prob

            # Luôn đảm bảo xuất hidden_states và attentions
            self.config.output_hidden_states = True
            self.config.output_attentions = True

            try:
                self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config, local_files_only=True)
            except Exception:
                self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        self.hidden_dim = getattr(self.config, "hidden_size", 768)
        self.num_hidden_layers = getattr(self.config, "num_hidden_layers", 12)

        # 2. Xác định chiều đầu vào cho classifier dựa trên pooling_strategy
        if self.pooling_strategy == "concat_4":
            self.classifier_in_dim = self.hidden_dim * 4
        elif self.pooling_strategy in ("last", "mean_4", "weighted"):
            self.classifier_in_dim = self.hidden_dim

        # 3. Trọng số học được cho chiến lược "weighted" pooling
        if self.pooling_strategy == "weighted":
            # Gồm embedding output + num_hidden_layers
            total_hidden_tensors = self.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(total_hidden_tensors))
        else:
            self.register_parameter("layer_weights", None)

        # 4. Classification Head & Loss
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.classifier_in_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # 5. Áp dụng chiến lược đóng băng tham số
        self.apply_freeze_strategy(
            freeze_embeddings=self.freeze_embeddings_flag,
            freeze_encoder_layers=self.freeze_encoder_layers_count,
            freeze_entire_encoder=self.freeze_entire_encoder_flag,
        )

    def _get_encoder_layers(self) -> Optional[nn.ModuleList]:
        """Trích xuất ModuleList chứa các tầng encoder theo kiến trúc RoBERTa/BERT."""
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            return self.encoder.encoder.layer
        elif hasattr(self.encoder, "layer"):
            return self.encoder.layer
        elif hasattr(self.encoder, "layers"):
            return self.encoder.layers
        return None

    def apply_freeze_strategy(
        self,
        freeze_embeddings: bool = False,
        freeze_encoder_layers: int = 0,
        freeze_entire_encoder: bool = False,
    ) -> None:
        """
        Áp dụng chiến lược đóng băng các tham số trong backbone encoder.

        Args:
            freeze_embeddings: Đóng băng module embedding.
            freeze_encoder_layers: Số tầng encoder dưới cùng bị đóng băng (0 = không đóng băng tầng nào).
            freeze_entire_encoder: Đóng băng toàn bộ encoder backbone (chỉ classifier head trainable).
        """
        self.freeze_embeddings_flag = freeze_embeddings
        self.freeze_encoder_layers_count = freeze_encoder_layers
        self.freeze_entire_encoder_flag = freeze_entire_encoder

        # Trường hợp 1: Đóng băng toàn bộ encoder backbone
        if freeze_entire_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            return

        # Trường hợp 2: Đóng băng embeddings
        embeddings = getattr(self.encoder, "embeddings", None)
        if embeddings is not None:
            for param in embeddings.parameters():
                param.requires_grad = not freeze_embeddings

        # Trường hợp 3: Đóng băng N tầng encoder dưới cùng
        layers = self._get_encoder_layers()
        if layers is not None:
            num_layers = len(layers)
            for idx, layer in enumerate(layers):
                should_freeze = (idx < freeze_encoder_layers)
                for param in layer.parameters():
                    param.requires_grad = not should_freeze

    def unfreeze_all(self) -> None:
        """Mở khóa toàn bộ tham số của mô hình (phục vụ progressive unfreezing)."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_embeddings_flag = False
        self.freeze_encoder_layers_count = 0
        self.freeze_entire_encoder_flag = False

    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ encoder backbone."""
        self.apply_freeze_strategy(freeze_entire_encoder=True)

    def get_trainable_parameters(self) -> Dict[str, Any]:
        """
        Thống kê chi tiết số lượng tham số trainable và tổng số tham số.

        Returns:
            Dict chứa:
                - trainable_params (int)
                - total_params (int)
                - trainable_ratio (float, % từ 0.0 đến 100.0)
                - encoder_trainable (int)
                - encoder_total (int)
                - classifier_trainable (int)
                - classifier_total (int)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        encoder_total = sum(p.numel() for p in self.encoder.parameters())

        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        classifier_total = sum(p.numel() for p in self.classifier.parameters())

        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_ratio,
            "encoder_trainable": encoder_trainable,
            "encoder_total": encoder_total,
            "classifier_trainable": classifier_trainable,
            "classifier_total": classifier_total,
        }

    def forward(
        self,
        input_ids: torch.Tensor,                                # [B, L]
        attention_mask: torch.Tensor,                           # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,          # [B, L] (tùy chọn)
        labels: Optional[torch.Tensor] = None,                  # [B, L]
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass mô hình Transformer Token Classification.

        Args:
            input_ids: Tensor chỉ số subwords [B, L].
            attention_mask: Tensor mặt nạ chú ý [B, L] (1: subword thực, 0: padding).
            token_type_ids: Tensor segment id [B, L] (nếu backbone hỗ trợ).
            labels: Tensor nhãn mục tiêu [B, L] (giá trị trong [0, num_labels-1] hoặc ignore_index).
            output_hidden_states: Có trả về hidden states hay không (mặc định lấy theo config).
            output_attentions: Có trả về attention weights hay không (mặc định lấy theo config).
            return_dict: Trả về dict chuẩn.

        Returns:
            Dict chứa:
                - 'loss': Scalar CrossEntropyLoss nếu labels được cung cấp, ngược lại None.
                - 'logits': Tensor phân phối điểm số nhãn [B, L, num_labels].
                - 'hidden_states': Tuple các tầng ẩn từ embedding đến layer cuối.
                - 'attentions': Tuple attention weights của các tầng.
        """
        # Chuẩn bị kwargs truyền vào encoder backbone
        encoder_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if output_hidden_states is not None:
            encoder_kwargs["output_hidden_states"] = output_hidden_states
        if output_attentions is not None:
            encoder_kwargs["output_attentions"] = output_attentions

        # Truyền token_type_ids nếu được cung cấp và backbone có tham số này
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        # Forward qua Transformer Backbone
        try:
            outputs = self.encoder(**encoder_kwargs)
        except TypeError:
            # Nếu backbone không hỗ trợ token_type_ids (ví dụ XLM-RoBERTa), tự động loại bỏ
            encoder_kwargs.pop("token_type_ids", None)
            outputs = self.encoder(**encoder_kwargs)

        all_hidden_states = outputs.hidden_states  # Tuple của num_layers + 1 tensors, mỗi tensor [B, L, H]

        if all_hidden_states is None:
            raise RuntimeError(
                "Encoder không xuất `hidden_states`! Hãy đảm bảo output_hidden_states=True trong config."
            )

        # Áp dụng chiến lược pooling tầng ẩn (Multi-layer hidden states combination)
        if self.pooling_strategy == "last":
            # Tầng ẩn cuối cùng: [B, L, H]
            features = all_hidden_states[-1]

        elif self.pooling_strategy == "concat_4":
            # Ghép nối 4 tầng ẩn cuối: [B, L, 4 * H]
            features = torch.cat(all_hidden_states[-4:], dim=-1)

        elif self.pooling_strategy == "mean_4":
            # Trung bình cộng 4 tầng ẩn cuối: [B, L, H]
            stacked = torch.stack(all_hidden_states[-4:], dim=0)   # [4, B, L, H]
            features = torch.mean(stacked, dim=0)                  # [B, L, H]

        elif self.pooling_strategy == "weighted":
            # Tổng có trọng số softmax học được của tất cả các tầng ẩn: [B, L, H]
            normalized_weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers + 1]
            stacked = torch.stack(all_hidden_states, dim=0)                # [num_layers + 1, B, L, H]
            weight_view = normalized_weights.view(-1, 1, 1, 1)            # [num_layers + 1, 1, 1, 1]
            features = torch.sum(stacked * weight_view, dim=0)             # [B, L, H]

        # Áp dụng Dropout và Linear Classification Head
        features = self.dropout(features)            # [B, L, classifier_in_dim]
        logits = self.classifier(features)           # [B, L, num_labels]

        loss = None
        if labels is not None:
            # Flatten để đưa vào CrossEntropyLoss
            # active_logits: [B * L, num_labels]
            # active_labels: [B * L]
            active_logits = logits.view(-1, self.num_labels)
            active_labels = labels.view(-1)
            loss = self.loss_fn(active_logits, active_labels)

        return {
            "loss": loss,
            "logits": logits,                                 # [B, L, num_labels]
            "hidden_states": all_hidden_states,
            "attentions": outputs.attentions,
        }

    @classmethod
    def from_config(
        cls,
        cfg: Union[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> "PretrainedNERModel":
        """
        Khởi tạo PretrainedNERModel từ tệp YAML hoặc dictionary cấu hình.

        Args:
            cfg: Đường dẫn tệp .yaml hoặc dict cấu hình (vd: configs/pretrained_plm.yaml).
            kwargs: Các tham số ghi đè thêm (nếu có).

        Returns:
            PretrainedNERModel
        """
        config_dict: Dict[str, Any] = {}
        if isinstance(cfg, str):
            if not os.path.exists(cfg):
                raise FileNotFoundError(f"Không tìm thấy file cấu hình: {cfg}")
            with open(cfg, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(cfg, dict):
            config_dict = cfg
        else:
            raise TypeError(f"cfg phải là str (đường dẫn file) hoặc dict, nhận được: {type(cfg)}")

        # Lấy phần cấu hình model
        model_cfg = config_dict.get("model", {})

        init_args = {
            "model_name_or_path": model_cfg.get("model_name_or_path", "vinai/phobert-base-v2"),
            "num_labels": model_cfg.get("num_labels", 9),
            "pooling_strategy": model_cfg.get("pooling_strategy", "last"),
            "freeze_embeddings": model_cfg.get("freeze_embeddings", False),
            "freeze_encoder_layers": model_cfg.get("freeze_encoder_layers", 0),
            "freeze_entire_encoder": model_cfg.get("freeze_entire_encoder", False),
            "classifier_dropout": model_cfg.get("classifier_dropout", 0.2),
            "hidden_dropout_prob": model_cfg.get("hidden_dropout_prob", None),
            "ignore_index": model_cfg.get("ignore_index", -100),
        }

        # Ghi đè bằng kwargs nếu người dùng truyền thêm
        init_args.update(kwargs)

        return cls(**init_args)

    def get_architecture_diagram(self) -> str:
        """Trả về sơ đồ khối kiến trúc dạng văn bản trực quan."""
        strategy_desc = {
            "last": f"Last Hidden State Layer [-1] -> [B, L, {self.hidden_dim}]",
            "concat_4": f"Concat 4 Last Hidden Layers [-4:] -> [B, L, {self.hidden_dim * 4}]",
            "mean_4": f"Mean 4 Last Hidden Layers [-4:] -> [B, L, {self.hidden_dim}]",
            "weighted": f"Softmax Learnable Weighted Sum of all {self.num_hidden_layers + 1} Hidden States -> [B, L, {self.hidden_dim}]",
        }
        diagram = [
            "=======================================================================",
            f"  PretrainedNERModel Architecture: {self.model_name_or_path}",
            "=======================================================================",
            " [Input Tokens: B x L]  ──>  [Subword Tokenizer & Alignment]",
            "          │",
            "          ▼",
            " [Input IDs & Attention Mask: B x L]",
            "          │",
            "          ▼",
            f" [Transformer Backbone: {self.model_name_or_path}]",
            f"    ├── Embeddings Layer: {'[FROZEN]' if self.freeze_embeddings_flag or self.freeze_entire_encoder_flag else '[TRAINABLE]'}",
            f"    ├── Encoder Layers (0..{self.num_hidden_layers - 1}):",
        ]
        for l in range(self.num_hidden_layers):
            is_frozen = self.freeze_entire_encoder_flag or (l < self.freeze_encoder_layers_count)
            status = "[FROZEN]" if is_frozen else "[TRAINABLE]"
            diagram.append(f"    │    Layer {l:02d}: {status}")

        diagram.extend([
            "          │ (All Hidden States: 13 tensors of [B, L, H])",
            "          ▼",
            f" [Hidden States Pooling: '{self.pooling_strategy}']",
            f"    └── {strategy_desc.get(self.pooling_strategy, 'Unknown strategy')}",
            "          │",
            "          ▼",
            f" [Dropout Layer: p = {self.classifier_dropout_prob}]",
            "          │",
            "          ▼",
            f" [Linear Classifier: {self.classifier_in_dim} -> {self.num_labels}] [TRAINABLE]",
            "          │",
            "          ▼",
            f" [Logits Output: B x L x {self.num_labels}]",
            "          │ (When labels are provided)",
            "          ▼",
            f" [CrossEntropyLoss (ignore_index={self.ignore_index})]",
            "=======================================================================",
        ])
        return "\n".join(diagram)
