import torch
import torch.nn.functional as F

from mope.engine import nll_survival_loss
from mope.model import MoPE


def _inputs(num_patches=200, wsi_dim=1536, num_pathways=50, num_genes=4188):
    features = torch.randn(1, num_patches, wsi_dim)
    rna = torch.randn(1, num_pathways, num_genes) * (torch.rand(1, num_pathways, num_genes) < 0.04)
    return features, rna


def _objective(out, task_loss):
    return (
        task_loss
        + out["distillation_loss"]
        + out["alignment_loss"]
        + 0.05 * out["memory_loss"]
        + 0.01 * out["reconstruction_loss"]
        + 0.1 * out["diversity_loss"]
    )


def _assert_gradients(model):
    for name in ("slot_extractor.slot_queries", "expert_transform.expert_up_projections", "memory._codebook.embed"):
        assert model.get_parameter(name).grad.abs().sum() > 0, name
    for name, parameter in model.rna_autoencoder.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_classification_forward_backward():
    torch.manual_seed(0)
    model = MoPE(task_type="binary", n_classes=2)
    features, rna = _inputs()
    out = model.forward_train(rna, features)
    assert out["logits"].shape == (1, 2)
    loss = _objective(out, F.cross_entropy(out["logits"], torch.tensor([1])))
    loss.backward()
    assert torch.isfinite(loss)
    _assert_gradients(model)
    inference = model.eval().forward_inference(features)
    assert inference["logits"].shape == (1, 2)
    assert inference["expert_attention"].shape == (1, 50, 200)
    assert inference["memory_weights"].shape == (1, 50, 24)


def test_survival_forward_backward():
    torch.manual_seed(0)
    model = MoPE(task_type="survival", num_time_bins=4)
    features, rna = _inputs()
    out = model.forward_train(rna, features[0])
    assert out["hazards"].shape == (1, 4)
    loss = _objective(out, nll_survival_loss(out["hazards"], out["survival"], torch.tensor([2]), torch.tensor([0.0])))
    loss.backward()
    assert torch.isfinite(loss)
    _assert_gradients(model)


if __name__ == "__main__":
    test_classification_forward_backward()
    test_survival_forward_backward()
    print("smoke tests passed")
