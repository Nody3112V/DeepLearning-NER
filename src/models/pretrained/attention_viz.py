"""
attention_viz.py - Trích xuất và trực quan hóa bản đồ chú ý (Attention Map) của Transformer.

Phục vụ phần phân tích lỗi và trực quan hóa (Analysis) trong báo cáo ACL.
"""
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def extract_attention_weights(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Trích xuất ma trận chú ý (attention matrix) từ tầng chỉ định.
    
    Args:
        model: Mô hình PretrainedNERModel
        input_ids: Tensor [1, L]
        attention_mask: Tensor [1, L]
        layer_idx: Chỉ số tầng chú ý (-1 là tầng cuối cùng)
        
    Returns:
        Tensor [num_heads, L, L] chứa ma trận chú ý của tất cả các attention heads.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        attentions = outputs.get("attentions")
        if attentions is None:
            raise ValueError("Mô hình không trả về attentions. Kiểm tra output_attentions=True trong AutoConfig.")
        
        # attentions là tuple chứa (num_layers) tensor có shape [B, num_heads, L, L]
        selected_layer_attention = attentions[layer_idx]  # [B, num_heads, L, L]
        # Lấy mẫu đầu tiên [num_heads, L, L]
        return selected_layer_attention[0]


def plot_attention_heatmap(
    tokens: List[str],
    attention_matrix: torch.Tensor,
    head_idx: int = 0,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Vẽ heatmap ma trận chú ý giữa các tokens cho một attention head cụ thể.
    
    Args:
        tokens: Danh sách subwords hoặc tokens có độ dài L
        attention_matrix: Tensor [num_heads, L, L] hoặc [L, L]
        head_idx: Chỉ số attention head cần vẽ
        output_path: Đường dẫn lưu file ảnh (ví dụ: outputs/figures/attention/sample_01_head_0.png)
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
    """
    if attention_matrix.dim() == 3:
        attn = attention_matrix[head_idx].cpu().numpy()  # [L, L]
    else:
        attn = attention_matrix.cpu().numpy()            # [L, L]

    seq_len = min(len(tokens), attn.shape[0])
    attn = attn[:seq_len, :seq_len]
    display_tokens = tokens[:seq_len]

    plt.figure(figsize=figsize)
    sns.heatmap(
        attn,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        cmap="Blues",
        annot=False,
        square=True,
        cbar_kws={"label": "Attention Weight"},
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    chart_title = title or f"Transformer Attention Map (Head {head_idx})"
    plt.title(chart_title, fontsize=12, pad=12)
    plt.tight_layout()

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ Attention Map vào: {out_file}")
    plt.close()
