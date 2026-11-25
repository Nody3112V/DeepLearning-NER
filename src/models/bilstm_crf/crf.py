from typing import List
import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain CRF tự cài đặt từ đầu (Forward log-sum-exp & Viterbi decoding)."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        if num_tags <= 0:
            raise ValueError(f"num_tags phải > 0, nhận được: {num_tags}")
        self.num_tags = num_tags

        # start_transitions[j]: điểm chuyển từ <START> sang nhãn j
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        # end_transitions[i]: điểm chuyển từ nhãn i sang <END>
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        # transitions[i, j]: điểm chuyển từ nhãn i (bước t-1) sang nhãn j (bước t)
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tính tổng điểm s(x, y) cho chuỗi nhãn đúng của từng câu trong batch.

        Args:
            emissions: (B, T, C)
            tags:      (B, T)
            mask:      (B, T) kiểu bool (True tại token thật, False tại <PAD>)
        Returns:
            score:     (B,)
        """
        batch_size, seq_len = tags.shape
        batch_idx = torch.arange(batch_size, device=emissions.device)

        # Bước t = 0: A[START, y_0] + P[0, y_0]
        score = self.start_transitions[tags[:, 0]] + emissions[batch_idx, 0, tags[:, 0]]

        # Bước t = 1 .. T-1: chỉ cộng tại các vị trí token thật (mask[:, t] == True)
        for t in range(1, seq_len):
            emit_t = emissions[batch_idx, t, tags[:, t]]
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit_t + trans_t) * mask[:, t].to(emissions.dtype)

        # Điểm kết thúc tại vị trí token thật cuối cùng của mỗi câu: A[y_{L-1}, END]
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags[batch_idx, last_idx]
        score = score + self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Thuật toán quy hoạch động Forward tính log Z(x) = log sum_y exp(s(x, y)).

        Args:
            emissions: (B, T, C)
            mask:      (B, T) kiểu bool
        Returns:
            log_z:     (B,)
        """
        seq_len = emissions.size(1)

        # Bước t = 0: alpha_0(j) = A[START, j] + P[0, j], shape (B, C)
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for t in range(1, seq_len):
            # broadcast_alpha: (B, C, 1) -> nhãn i ở bước t-1
            # transitions:     (1, C, C) -> A[i, j]
            # broadcast_emit:  (B, 1, C) -> P[t, j] ở bước t
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )  # (B, C, C)
            next_alpha = torch.logsumexp(scores, dim=1)  # (B, C)

            # Giữ nguyên alpha cũ tại các câu đã kết thúc (mask[:, t] == False)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)

        # Cộng điểm chuyển sang <END> và gộp log-sum-exp trên toàn bộ nhãn cuối
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Tính hàm mất mát Negative Log-Likelihood: log Z(x) - s(x, y)."""
        mask_bool = mask.bool()
        if not mask_bool[:, 0].all():
            raise ValueError("Token đầu tiên của mọi câu trong batch phải có mask=True.")

        log_partition = self._compute_log_partition(emissions, mask_bool)
        gold_score = self._compute_gold_score(emissions, tags, mask_bool)
        nll = log_partition - gold_score  # (B,), luôn >= 0

        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction == "none":
            return nll
        raise ValueError(f"reduction không hợp lệ: {reduction!r}")

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Thuật toán quy hoạch động Viterbi tìm chuỗi nhãn tối ưu cho từng câu.

        Returns:
            List[List[int]]: B chuỗi nhãn, mỗi chuỗi có độ dài đúng bằng độ dài thật của câu.
        """
        mask_bool = mask.bool()
        batch_size, seq_len, _ = emissions.shape
        lengths = mask_bool.long().sum(dim=1)  # (B,)

        # Bước t = 0: viterbi_0(j) = A[START, j] + P[0, j], shape (B, C)
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            # scores[b, i, j] = viterbi[b, i] + A[i, j]
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, C, C)
            best_prev_scores, best_prev_tags = scores.max(dim=1)           # (B, C), (B, C)
            next_viterbi = best_prev_scores + emissions[:, t]

            viterbi = torch.where(mask_bool[:, t].unsqueeze(1), next_viterbi, viterbi)
            backpointers.append(best_prev_tags)

        # Cộng điểm chuyển sang <END>
        viterbi = viterbi + self.end_transitions.unsqueeze(0)

        # Truy vết ngược (backtracking) theo đúng độ dài thật của từng câu
        best_paths: List[List[int]] = []
        for b in range(batch_size):
            sent_len = int(lengths[b].item())
            best_tag = int(viterbi[b].argmax().item())
            path = [best_tag]

            # backpointers[k] lưu nhãn tốt nhất ở bước k dẫn đến bước k + 1
            for k in range(sent_len - 2, -1, -1):
                best_tag = int(backpointers[k][b, best_tag].item())
                path.append(best_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths
