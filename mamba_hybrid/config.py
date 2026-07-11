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
