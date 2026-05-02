"""Soft-target knowledge distillation loss for language models."""

import torch
import torch.nn.functional as F


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Combined hard-label + soft-label KD loss.

    Args:
        student_logits: [batch, seq_len, vocab_size] from spiking student
        teacher_logits: [batch, seq_len, vocab_size] from ANN teacher
        labels: [batch, seq_len] ground truth token IDs
        alpha: Weight for hard label loss (1-alpha for soft label loss)
        temperature: Softmax temperature for soft targets

    The student's output logits come from a linear readout of spike counts,
    making them directly comparable to teacher logits.
    """
    # Flatten for loss computation
    s_flat = student_logits.reshape(-1, student_logits.size(-1))
    t_flat = teacher_logits.reshape(-1, teacher_logits.size(-1))
    labels_flat = labels.reshape(-1)

    # Hard label loss (cross-entropy with ground truth)
    hard_loss = F.cross_entropy(s_flat, labels_flat)

    # Soft label loss (KL divergence with teacher's soft predictions)
    student_soft = F.log_softmax(s_flat / temperature, dim=-1)
    teacher_soft = F.softmax(t_flat / temperature, dim=-1)
    soft_loss = F.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss
