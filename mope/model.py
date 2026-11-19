import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class MultiSlotMorphologyExtractor(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, num_slots: int):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(num_experts, num_slots, input_dim) / math.sqrt(input_dim))
        self.key_projection = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.key_projection(patch_features)
        slot_attention = F.softmax(torch.einsum("ksd,bpd->bksp", self.slot_queries, keys), dim=-1)
        slot_embeddings = torch.einsum("bksp,bpd->bksd", slot_attention, patch_features)
        return slot_embeddings, slot_attention


class LowRankMultiSlotExpert(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank_dim: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.shared_down_projection = nn.Linear(input_dim, rank_dim, bias=False)
        self.expert_up_projections = nn.Parameter(torch.randn(num_experts, rank_dim, output_dim) * 0.02)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.slot_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )

    def forward(self, slot_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_rank = self.shared_down_projection(slot_embeddings)
        slot_tokens = torch.einsum("bksr,krd->bksd", low_rank, self.expert_up_projections)
        slot_tokens = self.output_dropout(self.output_norm(F.relu(slot_tokens)))
        slot_gate_weights = F.softmax(self.slot_gate(slot_tokens).squeeze(-1), dim=-1)
        expert_tokens = torch.sum(slot_gate_weights.unsqueeze(-1) * slot_tokens, dim=2)
        return slot_tokens, slot_gate_weights, expert_tokens


class PathwayAutoencoder(nn.Module):
    def __init__(self, num_genes: int, token_dim: int):
        super().__init__()
        self.encoder = nn.Linear(num_genes, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.decoder = nn.Linear(token_dim, num_genes)

    @staticmethod
    def _size_normalize(pathways: torch.Tensor) -> torch.Tensor:
        sizes = pathways.ne(0).float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return pathways / sizes.sqrt()

    def encode(self, pathways: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self._size_normalize(pathways)))

    def forward(self, pathways: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode(pathways)
        membership = pathways.ne(0)
        reconstruction_loss = F.mse_loss(self.decoder(tokens)[membership], pathways[membership])
        return tokens, reconstruction_loss


class MoPE(nn.Module):
    """Mixture of Pathway Experts: RNA-guided training, H&E-only inference."""

    def __init__(
        self,
        num_pathways: int = 50,
        num_genes: int = 4188,
        wsi_input_dim: int = 1536,
        token_dim: int = 64,
        rank_dim: int = 64,
        num_memory: int = 24,
        num_slots: int = 16,
        task_type: str = "binary",
        n_classes: int = 2,
        num_time_bins: int = 4,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_memory = num_memory
        self.num_slots = num_slots
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature

        self.slot_extractor = MultiSlotMorphologyExtractor(wsi_input_dim, num_pathways, num_slots)
        self.expert_transform = LowRankMultiSlotExpert(wsi_input_dim, token_dim, rank_dim, num_pathways, dropout)
        self.expert_gate = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 1),
        )
        self.memory = VectorQuantize(
            dim=token_dim,
            codebook_size=num_memory,
            accept_image_fmap=False,
            orthogonal_reg_weight=10,
            orthogonal_reg_max_codes=128,
            orthogonal_reg_active_codes_only=False,
            ema_update=False,
        )
        self.rna_autoencoder = PathwayAutoencoder(num_genes, token_dim)
        self.head = nn.Linear(token_dim, num_time_bins if task_type == "survival" else n_classes)

    @staticmethod
    def prepare_wsi(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(f"Expected WSI features with shape (P, D) or (1, P, D), got {tuple(features.shape)}.")
        return features.float().unsqueeze(0)

    @staticmethod
    def prepare_rna(rna: torch.Tensor) -> torch.Tensor:
        rna = rna.unsqueeze(0) if rna.dim() == 2 else rna
        if rna.dim() != 3:
            raise ValueError(f"Expected RNA pathways with shape (B, K, G), got {tuple(rna.shape)}.")
        return rna.float()

    def memory_basis(self) -> torch.Tensor:
        return F.normalize(self.memory.codebook, p=2, dim=-1, eps=1e-6)

    @torch.no_grad()
    def pathway_tokens(self, rna: torch.Tensor) -> torch.Tensor:
        tokens = self.rna_autoencoder.encode(self.prepare_rna(rna))
        return F.normalize(tokens, p=2, dim=-1, eps=1e-6)

    def expert_tokens(self, features: torch.Tensor) -> torch.Tensor:
        slot_embeddings, _ = self.slot_extractor(features)
        return self.expert_transform(slot_embeddings)[2]

    def encode_wsi(self, features: torch.Tensor, update_memory: bool = True) -> Dict[str, torch.Tensor]:
        slot_embeddings, slot_attention = self.slot_extractor(features)
        _, slot_gate_weights, expert_tokens = self.expert_transform(slot_embeddings)
        _, _, memory_loss = self.memory(expert_tokens, freeze_codebook=not update_memory)
        basis = self.memory_basis()
        memory_logits = torch.matmul(expert_tokens, basis.T) / self.student_temperature

        normalized_experts = F.normalize(expert_tokens, p=2, dim=-1, eps=1e-6)
        expert_gates = F.softmax(self.expert_gate(normalized_experts).squeeze(-1), dim=1)
        slide_repr = F.normalize(torch.sum(expert_gates.unsqueeze(-1) * normalized_experts, dim=1), p=2, dim=-1)

        expert_attention = torch.sum(slot_gate_weights.unsqueeze(-1) * slot_attention, dim=2)
        patch_attention = F.softmax(torch.sum(expert_gates.unsqueeze(-1) * expert_attention, dim=1), dim=-1)
        return {
            "expert_tokens": expert_tokens,
            "expert_gates": expert_gates,
            "slot_attention": slot_attention,
            "slot_gates": slot_gate_weights,
            "expert_attention": expert_attention,
            "patch_attention": patch_attention,
            "slide_repr": slide_repr,
            "memory_logits": memory_logits,
            "memory_weights": F.softmax(memory_logits, dim=-1),
            "memory_loss": memory_loss,
            "basis": basis,
        }

    def predict(self, slide_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.task_type == "survival":
            hazards = torch.sigmoid(self.head(slide_repr))
            return {"hazards": hazards, "survival": torch.cumprod(1 - hazards, dim=1)}
        return {"logits": self.head(slide_repr)}

    def distillation_loss(self, memory_logits, rna_tokens, basis) -> torch.Tensor:
        with torch.no_grad():
            teacher = F.softmax(torch.matmul(rna_tokens, basis.T) / self.teacher_temperature, dim=-1)
        student = F.log_softmax(memory_logits, dim=-1)
        return F.kl_div(student, teacher, reduction="none").sum(dim=-1).mean()

    @staticmethod
    def alignment_loss(expert_tokens, rna_tokens) -> torch.Tensor:
        experts = F.normalize(expert_tokens, p=2, dim=-1, eps=1e-6)
        return (1 - F.cosine_similarity(experts, rna_tokens, dim=-1)).mean()

    def diversity_loss(self, slot_attention: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(slot_attention, p=2, dim=-1, eps=1e-6)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diagonal = 1 - torch.eye(self.num_slots, device=similarity.device)
        return (similarity * off_diagonal).pow(2).mean()

    def forward_train(self, rna: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        wsi = self.encode_wsi(self.prepare_wsi(features), update_memory=True)
        tokens, reconstruction_loss = self.rna_autoencoder(self.prepare_rna(rna))
        rna_tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6).detach()
        outputs = self.predict(wsi["slide_repr"])
        outputs.update(
            distillation_loss=self.distillation_loss(wsi["memory_logits"], rna_tokens, wsi["basis"].detach()),
            alignment_loss=self.alignment_loss(wsi["expert_tokens"], rna_tokens),
            memory_loss=wsi["memory_loss"],
            reconstruction_loss=reconstruction_loss,
            diversity_loss=self.diversity_loss(wsi["slot_attention"]),
        )
        return outputs

    @torch.no_grad()
    def forward_inference(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        wsi = self.encode_wsi(self.prepare_wsi(features), update_memory=False)
        outputs = self.predict(wsi["slide_repr"])
        outputs.update({k: wsi[k] for k in ("expert_gates", "expert_attention", "patch_attention", "memory_weights")})
        outputs.update(slot_attention=wsi["slot_attention"], slot_gates=wsi["slot_gates"])
        return outputs
