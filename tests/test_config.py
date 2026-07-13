from mamba_hybrid.config import MambaHybridConfig


def test_config_defaults() -> None:
    config = MambaHybridConfig()
    assert config.d_model == 512
    assert config.n_meta == 128
    assert config.l_ans == 64
    assert config.n_steps == 6
    assert config.t_cycles == 5
    assert config.use_cuda_kernels is False
    assert config.M_min == 1
    assert config.M_max == 6


def test_config_allows_halting_horizon_beyond_cycle_length() -> None:
    config = MambaHybridConfig(n_steps=2, M_max=3)
    assert config.M_max == 3
