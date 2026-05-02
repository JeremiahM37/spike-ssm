"""Knowledge distillation pipeline: ANN teacher -> Spiking student.

Supports three KD methods:
1. Logit matching (soft-target KD)
2. Feature matching (hidden state alignment)
3. Spike-rate matching (novel: match SNN firing rates to ANN activations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .logit_matching import logit_kd_loss
from .feature_matching import FeatureProjector, feature_kd_loss
from .spike_encoding import spike_rate_kd_loss
from .advanced_kd import SAMDLoss, NLDLoss, TemporalSeparationKD, CutoffRegularization


@dataclass
class KDConfig:
    alpha: float = 0.5       # Hard label loss weight
    beta: float = 0.3        # Feature KD weight
    gamma: float = 0.2       # Spike-rate KD weight
    temperature: float = 4.0
    use_feature_kd: bool = True
    use_spike_rate_kd: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    # Advanced KD options
    use_samd: bool = False        # Saliency-scaled Activation Map Distillation
    samd_weight: float = 0.2
    use_nld: bool = False         # Noise-smoothed Logits Distillation
    nld_weight: float = 0.3
    nld_noise_std: float = 0.1
    use_temporal_sep: bool = False # Per-timestep KD (CVPR 2025)
    temporal_sep_weight: float = 0.3
    temporal_sep_timesteps: int = 4
    use_cutoff_reg: bool = False  # Early-exit cutoff regularization
    cutoff_weight: float = 0.1
    cutoff_timesteps: int = 4


class KDPipeline:
    """Knowledge distillation training pipeline.

    Distills from a frozen ANN teacher into a trainable spiking student.
    Uses backprop with surrogate gradients during pre-deployment training.
    (MF replaces backprop only post-deployment, in Phase 4.)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: KDConfig,
        device: str = "cpu",
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Feature projectors to align student/teacher hidden dimensions
        self.projector = None
        if config.use_feature_kd:
            student_dim = self._get_dim(student)
            teacher_dim = self._get_dim(teacher)
            if student_dim != teacher_dim:
                self.projector = FeatureProjector(student_dim, teacher_dim).to(device)
                self.optimizer.add_param_group({
                    "params": self.projector.parameters(),
                    "lr": config.learning_rate,
                })

        # Advanced KD modules
        self.samd = SAMDLoss() if config.use_samd else None
        self.nld = NLDLoss(noise_std=config.nld_noise_std, temperature=config.temperature) if config.use_nld else None
        self.temporal_sep = TemporalSeparationKD(
            timesteps=config.temporal_sep_timesteps, temperature=config.temperature
        ) if config.use_temporal_sep else None
        self.cutoff_reg = CutoffRegularization(
            timesteps=config.cutoff_timesteps, cutoff_weight=config.cutoff_weight
        ) if config.use_cutoff_reg else None

        self.step_count = 0

    def _get_dim(self, model) -> int:
        if hasattr(model, "get_hidden_dim"):
            return model.get_hidden_dim()
        if hasattr(model, "config"):
            return getattr(model.config, "n_embd", 768)
        return 768

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """One KD training step.

        Returns dict of loss components for logging.
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Teacher forward (frozen, no grad)
        teacher_out = self.teacher(input_ids)

        # Student forward (spiking, with surrogate gradients)
        student_out = self.student(
            input_ids,
            targets=labels,
            return_hidden_states=self.config.use_feature_kd,
            return_spike_rates=self.config.use_spike_rate_kd,
        )

        # 1. Combined logit + hard label loss
        loss_logit = logit_kd_loss(
            student_out.logits,
            teacher_out.logits,
            labels,
            alpha=self.config.alpha,
            temperature=self.config.temperature,
        )

        metrics = {"loss_logit": loss_logit.item()}
        total_loss = loss_logit

        # 2. Feature matching loss
        if self.config.use_feature_kd and student_out.hidden_states and teacher_out.hidden_states:
            loss_feature = feature_kd_loss(
                student_out.hidden_states,
                teacher_out.hidden_states,
                projector=self.projector,
            )
            total_loss = total_loss + self.config.beta * loss_feature
            metrics["loss_feature"] = loss_feature.item()

        # 3. Spike-rate matching loss (novel)
        if self.config.use_spike_rate_kd and student_out.spike_rates and teacher_out.hidden_states:
            loss_spike = spike_rate_kd_loss(
                student_out.spike_rates,
                teacher_out.hidden_states,
            )
            total_loss = total_loss + self.config.gamma * loss_spike
            metrics["loss_spike_rate"] = loss_spike.item()

        # 4. SAMD: saliency-weighted feature alignment
        if self.samd and student_out.hidden_states and teacher_out.hidden_states:
            loss_samd = self.samd(
                student_out.hidden_states[-1],
                teacher_out.hidden_states[-1],
                teacher_out.logits,
            )
            total_loss = total_loss + self.config.samd_weight * loss_samd
            metrics["loss_samd"] = loss_samd.item()

        # 5. NLD: noise-smoothed logit distillation
        if self.nld:
            loss_nld = self.nld(student_out.logits, teacher_out.logits)
            total_loss = total_loss + self.config.nld_weight * loss_nld
            metrics["loss_nld"] = loss_nld.item()

        # 6. Temporal Separation KD (requires per-timestep outputs)
        if self.temporal_sep and hasattr(student_out, 'logits_per_timestep') and student_out.logits_per_timestep:
            loss_ts = self.temporal_sep(student_out.logits_per_timestep, teacher_out.logits)
            total_loss = total_loss + self.config.temporal_sep_weight * loss_ts
            metrics["loss_temporal_sep"] = loss_ts.item()

        # 7. Cutoff regularization (requires per-timestep outputs)
        if self.cutoff_reg and hasattr(student_out, 'logits_per_timestep') and student_out.logits_per_timestep:
            loss_cutoff = self.cutoff_reg(student_out.logits_per_timestep, labels)
            total_loss = total_loss + loss_cutoff
            metrics["loss_cutoff"] = loss_cutoff.item()

        # Backward + update
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.student.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        self.step_count += 1
        metrics["total_loss"] = total_loss.item()
        metrics["step"] = self.step_count
        return metrics

    def evaluate(
        self,
        dataloader,
        max_batches: Optional[int] = None,
    ) -> dict[str, float]:
        """Evaluate student perplexity on a dataset."""
        self.student.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                out = self.student(input_ids, targets=labels)
                if out.loss is not None:
                    total_loss += out.loss.item() * labels.numel()
                    total_tokens += labels.numel()

        self.student.train()

        avg_loss = total_loss / max(total_tokens, 1)
        import math
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": math.exp(avg_loss),
            "eval_bpc": avg_loss / math.log(2),
        }
