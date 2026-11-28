"""
examples_subword_align.py - Minh họa thực tế quy trình căn chỉnh subword với PhoBERT và XLM-R.
"""
from transformers import AutoTokenizer
from src.models.pretrained.subword_align import (
    align_tokens_and_labels,
    align_batch_tokens_and_labels,
)

LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def demo_tokenizer(model_name: str, tokens: list, tags: list):
    print(f"\n=======================================================")
    print(f"MÔ HÌNH: {model_name}")
    print(f"=======================================================")
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    print(f"Tokenizer class: {type(tok).__name__} | is_fast: {tok.is_fast}")

    res = align_tokens_and_labels(
        tokens=tokens,
        tags=tags,
        tokenizer=tok,
        label2id=LABEL2ID,
        max_length=64,
    )

    subwords = tok.convert_ids_to_tokens(res["input_ids"])

    print(f"\nCâu gốc tokens: {tokens}")
    print(f"Nhãn gốc tags : {tags}\n")
    print(f"{'Idx':<4}{'Subword':<18}{'Input ID':<10}{'Word ID':<10}{'Label ID':<10}{'Label Tag':<10}")
    print("-" * 62)

    for i, (sw, iid, wid, lid) in enumerate(zip(subwords, res["input_ids"], res["word_ids"], res["labels"])):
        ltag = ID2LABEL.get(lid, "-100 (Bỏ qua)") if lid != -100 else "-100 (Bỏ qua)"
        wid_str = str(wid) if wid is not None else "None"
        print(f"{i:<4}{sw:<18}{iid:<10}{wid_str:<10}{lid:<10}{ltag:<10}")


def demo_batch(model_name: str):
    print(f"\n=======================================================")
    print(f"DEMO BATCH VỚI DYNAMIC PADDING: {model_name}")
    print(f"=======================================================")
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    batch_tokens = [
        ["Hà_Nội", "đẹp", "lắm", "."],
        ["Giáo_sư", "Ngô_Bảo_Châu", "giảng_dạy", "tại", "Đại_học", "Chicago", "ở", "Mỹ", "."],
    ]
    batch_tags = [
        ["B-LOC", "O", "O", "O"],
        ["O", "B-PER", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O"],
    ]

    batch = align_batch_tokens_and_labels(
        batch_tokens=batch_tokens,
        batch_tags=batch_tags,
        tokenizer=tok,
        label2id=LABEL2ID,
        max_length=64,
        padding=True,
    )

    print(f"Input IDs shape     : {batch['input_ids'].shape} -> [Batch=2, Max_Len={batch['input_ids'].shape[1]}]")
    print(f"Attention Mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape        : {batch['labels'].shape}")
    print("\nCâu 1 labels sau padding:\n", batch['labels'][0].tolist())
    print("\nCâu 2 labels sau padding:\n", batch['labels'][1].tolist())


if __name__ == "__main__":
    sample_tokens = ["Giáo_sư", "Ngô_Bảo_Châu", "sinh", "tại", "Hà_Nội", "."]
    sample_tags = ["O", "B-PER", "O", "O", "B-LOC", "O"]

    demo_tokenizer("vinai/phobert-base-v2", sample_tokens, sample_tags)
    demo_tokenizer("xlm-roberta-base", sample_tokens, sample_tags)
    demo_batch("vinai/phobert-base-v2")
