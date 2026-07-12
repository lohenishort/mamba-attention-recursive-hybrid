from dataclasses import dataclass


@dataclass
class MambaHybridConfig:
    d_model: int = 512
    n_meta: int = 128
    l_ans: int = 64
    n_steps: int = 6
    t_cycles: int = 5
    max_noise_step: int = 20
    sigma_base: float = 0.05
    use_cuda_kernels: bool = False
    M_min: int = 1
    M_max: int = 6
    use_moe: bool = False
    num_experts: int = 4
    moe_top_k: int = 1
