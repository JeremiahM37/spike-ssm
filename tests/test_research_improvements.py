"""Tests for research improvements (Contrastive FF, Advanced KD, Continual Learning, SpikeRWKV-7)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn


# ---- Contrastive FF Goodness ----

def test_contrastive_goodness():
    from src.mono_forward.goodness import contrastive_goodness
    x = torch.rand(4, 16)
    g = contrastive_goodness(x)
    assert g.dim() == 0
    assert 0 <= g.item() <= 1.0  # cosine similarity range

def test_deeper_forward_goodness():
    from src.mono_forward.goodness import deeper_forward_goodness
    x = torch.randn(4, 32)
    g = deeper_forward_goodness(x)
    assert g.dim() == 0
    assert g.item() >= 0  # L2 norm is non-negative

def test_marginal_contrastive_goodness():
    from src.mono_forward.goodness import marginal_contrastive_goodness
    pos = torch.rand(4, 16) + 1.0  # Shift up for distinct clusters
    neg = torch.rand(4, 16) - 1.0
    g = marginal_contrastive_goodness(pos, neg)
    assert g.dim() == 0

def test_contrastive_ff_loss():
    from src.mono_forward.local_loss import contrastive_ff_loss
    pos = torch.randn(4, 16, requires_grad=True)
    neg = torch.randn(4, 16, requires_grad=True)
    loss = contrastive_ff_loss(pos, neg)
    assert loss.item() > 0
    assert loss.requires_grad

def test_get_goodness_fn_includes_new():
    from src.mono_forward.goodness import get_goodness_fn
    for name in ["contrastive", "deeper_forward"]:
        fn = get_goodness_fn(name)
        assert callable(fn)


# ---- Advanced KD ----

def test_samd_loss():
    from src.distillation.advanced_kd import SAMDLoss
    samd = SAMDLoss()
    s = torch.randn(2, 8, 64)
    t = torch.randn(2, 8, 64)
    logits = torch.randn(2, 8, 100)
    loss = samd(s, t, logits)
    assert loss.item() > 0

def test_nld_loss():
    from src.distillation.advanced_kd import NLDLoss
    nld = NLDLoss(noise_std=0.1, temperature=4.0)
    s = torch.randn(2, 8, 100)
    t = torch.randn(2, 8, 100)
    loss = nld(s, t)
    assert loss.item() >= 0

def test_temporal_separation_kd():
    from src.distillation.advanced_kd import TemporalSeparationKD
    tskd = TemporalSeparationKD(timesteps=4, temperature=4.0)
    # Simulate 4 timesteps of student outputs
    per_t = [torch.randn(2, 8, 100) for _ in range(4)]
    teacher = torch.randn(2, 8, 100)
    loss = tskd(per_t, teacher)
    assert isinstance(loss.item(), float)

def test_cutoff_regularization():
    from src.distillation.advanced_kd import CutoffRegularization
    cutoff = CutoffRegularization(timesteps=4)
    logits_per_t = [torch.randn(2, 8, 100) for _ in range(4)]
    labels = torch.randint(0, 100, (2, 8))
    loss = cutoff(logits_per_t, labels)
    assert loss.item() > 0

def test_early_exit_inference():
    from src.distillation.advanced_kd import EarlyExitInference
    eei = EarlyExitInference(confidence_threshold=0.5, min_timesteps=2)
    # Low confidence: should not exit
    logits = torch.randn(2, 8, 100)
    assert not eei.should_exit(logits, timestep=1)  # Below min_timesteps
    # High confidence
    logits_confident = torch.zeros(2, 8, 100)
    logits_confident[:, :, 0] = 100.0  # Very confident
    assert eei.should_exit(logits_confident, timestep=3)
    assert eei.compute_savings(3, 8) == pytest.approx(0.625)


# ---- Continual Learning ----

def test_neurogenesis_pool():
    from src.mono_forward.continual_learning import NeurogenesisPool
    pool = NeurogenesisPool(32, 8, threshold_novelty=0.3)
    x = torch.randn(2, 32)
    out = pool(x)
    assert out.shape == (2, 32)
    # First input establishes baseline, second should show some novelty
    pool.compute_novelty(x)
    x2 = torch.randn(2, 32) * 5  # Very different input
    novelty = pool.compute_novelty(x2)
    assert novelty >= 0

def test_metaplastic_weights():
    from src.mono_forward.continual_learning import MetaplasticWeights
    model = nn.Linear(16, 16)
    mp = MetaplasticWeights(model)
    # Record some updates
    for name, _ in model.named_parameters():
        mp.record_update(name)
        mp.record_update(name)
    stats = mp.get_stiffness_stats()
    assert len(stats) > 0
    for name, s in stats.items():
        assert s['mean'] > 1.0  # Should increase after updates

def test_three_factor_mf_learner():
    from src.mono_forward.continual_learning import ThreeFactorMFLearner
    learner = ThreeFactorMFLearner(goodness_name='rate_squared', lr=0.01, epsilon=0.01)
    layer = nn.Linear(16, 16)
    pos = torch.randn(2, 16) * 2
    neg = torch.randn(2, 16) * 0.5
    metrics = learner.update_layer(layer, pos, neg)
    assert 'baseline_gap' in metrics
    assert 'modulation' in metrics
    assert 0 <= metrics['modulation'] <= 1

def test_stdp_learner():
    from src.mono_forward.continual_learning import STDPLearner
    learner = STDPLearner(lr=0.01)
    layer = nn.Linear(8, 8)
    # 3D spike trains: [batch, time, neurons]
    pre = (torch.rand(2, 10, 8) > 0.5).float()
    post = (torch.rand(2, 10, 8) > 0.5).float()
    metrics = learner.update_layer(layer, pre, post)
    assert 'stdp_update' in metrics
    assert metrics['stdp_update'] > 0

def test_hybrid_stdp_mf():
    from src.mono_forward.continual_learning import HybridSTDPMFLearner
    learner = HybridSTDPMFLearner(mf_lr=0.01, stdp_lr=0.01)
    # Simple block with ffn attribute
    block = nn.Module()
    block.ffn = nn.Linear(16, 16)
    pos = torch.randn(2, 16)
    neg = torch.randn(2, 16)
    metrics = learner.update_block(block, pos, neg)
    assert any('mf_' in k for k in metrics)


# ---- SpikeRWKV-7 ----

def test_spike_rwkv7_forward():
    from src.models.spike_rwkv7 import SpikeRWKV7Model, SpikeRWKV7Config
    cfg = SpikeRWKV7Config(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    model = SpikeRWKV7Model(cfg)
    ids = torch.randint(0, 50, (2, 8))
    out = model(ids)
    assert out.logits.shape == (2, 8, 50)

def test_spike_rwkv7_with_targets():
    from src.models.spike_rwkv7 import SpikeRWKV7Model, SpikeRWKV7Config
    cfg = SpikeRWKV7Config(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    model = SpikeRWKV7Model(cfg)
    ids = torch.randint(0, 50, (2, 8))
    out = model(ids, targets=ids)
    assert out.loss is not None
    assert out.loss.item() > 0

def test_spike_rwkv7_hidden_states():
    from src.models.spike_rwkv7 import SpikeRWKV7Model, SpikeRWKV7Config
    cfg = SpikeRWKV7Config(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    model = SpikeRWKV7Model(cfg)
    ids = torch.randint(0, 50, (2, 8))
    out = model(ids, return_hidden_states=True, return_spike_rates=True)
    assert len(out.hidden_states) == 2
    assert len(out.spike_rates) == 2

def test_adaptive_threshold_lif():
    from src.models.spike_rwkv7 import AdaptiveThresholdLIF
    lif = AdaptiveThresholdLIF(16, beta=0.9, base_threshold=1.0)
    x = torch.randn(2, 16)
    spikes, mem = lif(x)
    assert spikes.shape == x.shape
    assert mem.shape == x.shape
    # Thresholds should be learnable
    assert lif.threshold_offset.requires_grad

def test_spike_rwkv7_generalized_delta():
    from src.models.spike_rwkv7 import SpikeRWKV7TimeMix
    tm = SpikeRWKV7TimeMix(32, head_size=16, use_generalized_delta=True)
    x = torch.randn(2, 8, 32)
    out = tm(x)
    assert out.shape == x.shape


# ---- NIR Export ----

def test_nir_graph_creation():
    from src.models.nir_export import NIRGraph, NIRNode, NIREdge
    g = NIRGraph()
    g.add_node(NIRNode("input", "input"))
    g.add_node(NIRNode("lif1", "lif", params={"threshold": 1.0}))
    g.add_edge(NIREdge("input", "lif1"))
    d = g.to_dict()
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1

def test_nir_export_model():
    from src.models.nir_export import export_spikegpt_to_nir
    from src.models.spikegpt_wrapper import SpikeGPTWrapper, SpikeGPTConfig
    cfg = SpikeGPTConfig(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    model = SpikeGPTWrapper(cfg)
    graph = export_spikegpt_to_nir(model, cfg)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    summary = graph.summary()
    assert "NIR Graph" in summary


# ---- Integrated Advanced KD Pipeline ----

def test_kd_pipeline_with_samd_nld():
    """Test KDPipeline with SAMD and NLD enabled."""
    from src.distillation.kd_pipeline import KDPipeline, KDConfig
    from src.models.spikegpt_wrapper import SpikeGPTWrapper, SpikeGPTConfig

    cfg = SpikeGPTConfig(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    student = SpikeGPTWrapper(cfg)
    teacher = SpikeGPTWrapper(cfg)

    kd_config = KDConfig(
        use_feature_kd=True,
        use_spike_rate_kd=False,
        use_samd=True,
        samd_weight=0.2,
        use_nld=True,
        nld_weight=0.3,
        nld_noise_std=0.1,
    )
    pipeline = KDPipeline(student, teacher, kd_config)

    ids = torch.randint(0, 50, (2, 8))
    metrics = pipeline.train_step(ids, ids)
    assert "total_loss" in metrics
    assert "loss_nld" in metrics
    assert metrics["loss_nld"] > 0


def test_pretrained_teacher_interface():
    """Verify PretrainedRWKVTeacher has correct interface without loading model."""
    from src.models.pretrained_teacher import PretrainedRWKVTeacher
    teacher = PretrainedRWKVTeacher()
    # Check it has the expected methods without loading
    assert hasattr(teacher, 'forward')
    assert hasattr(teacher, 'get_hidden_dim')
    assert hasattr(teacher, 'get_num_params')
    assert hasattr(teacher, 'generate')
    assert teacher._model is None  # Not loaded yet


def test_convertible_teacher_has_gelu():
    """StubTeacher should have explicit nn.GELU modules for ANN-to-SNN conversion."""
    import torch.nn as nn
    from src.models.teacher import StubTeacher
    teacher = StubTeacher(vocab_size=100, n_embd=64, n_layers=2, n_head=2)
    gelu_count = sum(1 for _, m in teacher.named_modules() if isinstance(m, nn.GELU))
    assert gelu_count == 2, f"Expected 2 GELU modules (one per layer), got {gelu_count}"


def test_kd_pipeline_with_spike_rwkv7():
    """Test KDPipeline with SpikeRWKV-7 as student."""
    from src.distillation.kd_pipeline import KDPipeline, KDConfig
    from src.models.spike_rwkv7 import SpikeRWKV7Model, SpikeRWKV7Config
    from src.models.spikegpt_wrapper import SpikeGPTWrapper, SpikeGPTConfig

    student_cfg = SpikeRWKV7Config(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    student = SpikeRWKV7Model(student_cfg)

    teacher_cfg = SpikeGPTConfig(n_layers=2, n_embd=32, vocab_size=50, ctx_len=16)
    teacher = SpikeGPTWrapper(teacher_cfg)

    kd_config = KDConfig(use_feature_kd=True, use_spike_rate_kd=False, use_nld=True)
    pipeline = KDPipeline(student, teacher, kd_config)

    ids = torch.randint(0, 50, (2, 8))
    metrics = pipeline.train_step(ids, ids)
    assert "total_loss" in metrics
    assert metrics["total_loss"] > 0
