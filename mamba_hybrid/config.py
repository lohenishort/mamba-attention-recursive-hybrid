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
    activation_checkpointing: bool = False
    M_min: int = 1
    M_max: int = 6
    use_moe: bool = False
    num_experts: int = 4
    moe_top_k: int = 1
    vocab_size: int = 512
    halt_threshold: float = 0.5

    def __post_init__(self) -> None:
        positive = {
            "d_model": self.d_model,
            "n_meta": self.n_meta,
            "l_ans": self.l_ans,
            "n_steps": self.n_steps,
            "t_cycles": self.t_cycles,
            "vocab_size": self.vocab_size,
            "M_min": self.M_min,
            "M_max": self.M_max,
            "num_experts": self.num_experts,
            "moe_top_k": self.moe_top_k,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % 8 != 0:
            raise ValueError("d_model must be divisible by 8")
        if self.M_min > self.M_max:
            raise ValueError("M_min must not exceed M_max")
        if not 0.0 <= self.halt_threshold <= 1.0:
            raise ValueError("halt_threshold must be in [0, 1]")
        if self.sigma_base < 0.0:
            raise ValueError("sigma_base must be non-negative")
        if self.max_noise_step < 0:
            raise ValueError("max_noise_step must be non-negative")
        if self.moe_top_k > self.num_experts:
            raise ValueError("moe_top_k must not exceed num_experts")
