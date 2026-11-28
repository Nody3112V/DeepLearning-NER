"""
tests_pretrained_plm.py - Kiểm thử Sanity Check cho nhánh Pretrained PLM.

Kiểm tra:
  1. Căn chỉnh subword & nhãn (subword_align.py).
  2. Dynamic padding collate function.
  3. Forward pass và kiểm tra tensor shape ([B, L, H], [B, L, num_labels]).
  4. Backward pass và Gradient flow (không có layer nào bị None/NaN/Inf).
  5. Overfit trên một batch nhỏ để loss giảm.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

from src.models.pretrained.subword_align import align_labels_with_tokens, create_ner_collate_fn
from src.models.pretrained.model import PretrainedNERModel


class MockEncoder(nn.Module):
    """Mô hình giả lập kiến trúc Transformer để chạy test nhanh không cần tải weights nặng từ mạng."""
    def __init__(self, hidden_size=64, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2, dim_feedforward=128, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        x = self.embeddings(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        from collections import namedtuple
        Output = namedtuple("Output", ["last_hidden_state", "hidden_states", "attentions"])
        return Output(last_hidden_state=x, hidden_states=tuple(hidden_states), attentions=None)


def test_subword_alignment():
    print("[1/4] Kiểm tra căn chỉnh Subword và Nhãn...")
    class DummyTokenizer:
        bos_token_id = 0
        eos_token_id = 2
        cls_token_id = 0
        sep_token_id = 2
        def encode(self, word, add_special_tokens=False):
            # Giả lập: từ có dấu gạch dưới tách thành 2 subword
            if "_" in word:
                return [10, 11]
            return [12]

    tok = DummyTokenizer()
    tokens = ["Tôi", "ở", "Hà_Nội", "."]
    tags = ["O", "O", "B-LOC", "O"]
    tag2id = {"O": 0, "B-LOC": 1, "I-LOC": 2}

    input_ids, mask, labels = align_labels_with_tokens(tok, tokens, tags, tag2id, max_seq_len=20)
    # bos: -100, Tôi: 0, ở: 0, Hà: B-LOC (1), Nội: -100, .: 0, eos: -100
    assert labels[0] == -100, "BOS token phải là -100"
    assert labels[-1] == -100, "EOS token phải là -100"
    assert labels[3] == 1, "Subword đầu tiên của 'Hà_Nội' phải nhận B-LOC (1)"
    assert labels[4] == -100, "Subword thứ hai của 'Hà_Nội' phải nhận -100"
    print("  -> Căn chỉnh subword & nhãn: ĐẠT")


def test_collate_fn():
    print("[2/4] Kiểm tra Dynamic Padding trong DataLoader Collate...")
    collate = create_ner_collate_fn(pad_token_id=1)
    batch = [
        {"id": "s1", "input_ids": [0, 10, 11, 2], "attention_mask": [1, 1, 1, 1], "labels": [-100, 0, 1, -100], "tokens": ["a", "b"], "tags": ["O", "B-LOC"]},
        {"id": "s2", "input_ids": [0, 12, 2], "attention_mask": [1, 1, 1], "labels": [-100, 0, -100], "tokens": ["c"], "tags": ["O"]},
    ]
    padded = collate(batch)
    assert padded["input_ids"].shape == (2, 4), f"Lỗi shape input_ids: {padded['input_ids'].shape}"
    assert padded["attention_mask"].shape == (2, 4)
    assert padded["labels"].shape == (2, 4)
    assert padded["labels"][1, -1] == -100, "Pad label phải là -100"
    print("  -> Dynamic padding: ĐẠT")


def test_model_forward_and_shapes():
    print("[3/4] Kiểm tra Forward Pass và Tensor Shapes...")
    # Khởi tạo model với MockEncoder
    model = PretrainedNERModel.__new__(PretrainedNERModel)
    super(PretrainedNERModel, model).__init__()
    model.hidden_dim = 64
    model.num_labels = 9
    model.pooling_strategy = "concat_4"
    model.classifier_in_dim = 64 * 4
    model.encoder = MockEncoder(hidden_size=64, num_layers=4)
    model.dropout = nn.Dropout(0.1)
    model.classifier = nn.Linear(model.classifier_in_dim, model.num_labels)
    model.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    B, L = 2, 8
    input_ids = torch.randint(1, 100, (B, L))
    attention_mask = torch.ones((B, L), dtype=torch.long)
    labels = torch.randint(0, 9, (B, L))
    labels[0, 5:] = -100

    out = model(input_ids, attention_mask, labels=labels)
    logits = out["logits"]
    loss = out["loss"]

    assert logits.shape == (B, L, 9), f"Logits shape không đúng: {logits.shape}"
    assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss), "Loss bị NaN hoặc Inf!"
    print(f"  -> Forward shape [B={B}, L={L}, num_labels=9], Loss: {loss.item():.4f}: ĐẠT")


def test_backward_and_overfit():
    print("[4/4] Kiểm tra Backward Pass và Overfit trên 1 Batch nhỏ...")
    torch.manual_seed(42)
    model = PretrainedNERModel.__new__(PretrainedNERModel)
    super(PretrainedNERModel, model).__init__()
    model.hidden_dim = 32
    model.num_labels = 5
    model.pooling_strategy = "last"
    model.classifier_in_dim = 32
    model.encoder = MockEncoder(hidden_size=32, num_layers=2)
    model.dropout = nn.Dropout(0.0)
    model.classifier = nn.Linear(32, 5)
    model.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    input_ids = torch.randint(1, 100, (2, 6))
    attention_mask = torch.ones((2, 6), dtype=torch.long)
    labels = torch.tensor([[0, 1, 2, 0, -100, -100], [0, 0, 3, 4, 0, -100]], dtype=torch.long)

    # 1. Kiểm tra gradient flow
    model.train()
    optimizer.zero_grad()
    loss = model(input_ids, attention_mask, labels=labels)["loss"]
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient bị None tại: {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient bị NaN tại: {name}"

    # 2. Overfit 60 steps
    initial_loss = loss.item()
    for _ in range(60):
        optimizer.zero_grad()
        l = model(input_ids, attention_mask, labels=labels)["loss"]
        l.backward()
        optimizer.step()

    final_loss = l.item()
    assert final_loss < initial_loss * 0.2, f"Loss không giảm đủ: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"  -> Gradient Flow & Overfit: ĐẠT ({initial_loss:.4f} -> {final_loss:.4f})")


def run_all_tests():
    print("=== BẮT ĐẦU SANITY CHECK MÔ HÌNH PRETRAINED PLM ===")
    test_subword_alignment()
    test_collate_fn()
    test_model_forward_and_shapes()
    test_backward_and_overfit()
    print("=== TẤT CẢ SANITY CHECK ĐẠT CHUẨN KỸ THUẬT dl-engineer.md ===")


if __name__ == "__main__":
    run_all_tests()
