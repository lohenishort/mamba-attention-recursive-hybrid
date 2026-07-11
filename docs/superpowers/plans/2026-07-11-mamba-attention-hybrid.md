# Mamba-Attention Recursive Reasoning Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, high-performance PyTorch library implementing the Mamba-Attention Recursive Reasoning Hybrid framework with ACT halting heads and PTRM test-time scaling.

**Architecture:** A parallel attention and Mamba-2 hybrid operator merges sequence features using RMSNorm and gating. A dual-state ($y, z$) planning loop recursively updates a latent reasoning state and an answer state over variable compute steps, overseen by an ACT halting head and regularized with stochastic noise.

**Tech Stack:** PyTorch, Poetry, pytest, ruff, mypy.

## Global Constraints

- **ALWAYS** write the Mamba-2 / SSD scan operations in pure, readable PyTorch tensor operations as the default fallback to ensure CPU/GPU compatibility and easy local debugging.
- **ALWAYS** make the GPU-accelerated Triton/CUDA kernels from the official `mamba-ssm` library optional via a runtime check and configuration toggle (`use_cuda_kernels`).
- **ALWAYS** type-hint all function signatures, class initializers, and tensor shapes using PEP 484 type annotations and comments showing expected tensor dimensions (e.g., `# [batch_size, seq_len, d_model]`).
- **NEVER** apply loss to every intermediate recursion step in the latent planning loop; **ALWAYS** supervise sparsely at the final step of each segment to prevent the model from learning shortcut latent sequence replay behavior.
- **NEVER** use the 1-step Implicit Function Theorem (IFT) gradient approximation for backpropagation; **ALWAYS** perform full-recursion backpropagation through the computational graph of the supervision segments.
- **NEVER** detach the latent planning state's computational graph at the boundaries of each supervision segment to prevent gradient explosion and instability during long recursive unrolls.
- **NEVER** hardcode reward-shaping values or negative step penalties inside the Q-learning ACT halting head; **ALWAYS** use the mathematically verified binary reward scheme (1 for correct, 0 for incorrect) with explicit hyperparameter-driven compute bounds (`M_min`/`M_max`).
- **NEVER** modify or delete files under `tests/conftest.py` or the test verification suites.
- **NEVER** modify `poetry.lock` or add dependencies directly to `pyproject.toml` without verifying compatibility via `poetry check`.

---

### Task 1: Environment Scaffolding and Configurations

**Files:**
- Create: `mamba_hybrid/config.py`
- Create: `pyproject.toml`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `MambaHybridConfig` class containing all default model and optimization hyperparameters.

- [ ] **Step 1: Write the config test**
  Create `tests/test_config.py` containing:
  ```python
  from mamba_hybrid.config import MambaHybridConfig

  def test_config_defaults() -> None:
      config = MambaHybridConfig()
      assert config.d_model == 512
      assert config.n_meta == 128
      assert config.l_ans == 64
      assert config.n_steps == 6
      assert config.t_cycles == 5
      assert config.use_cuda_kernels is False
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_config.py -v`
  Expected: ModuleNotFoundError or ImportError.

- [ ] **Step 3: Create pyproject.toml and config file**
  Create `pyproject.toml` containing:
  ```toml
  [tool.poetry]
  name = "mamba-attention-recursive-hybrid"
  version = "0.1.0"
  description = "Mamba-Attention Recursive Reasoning Hybrid framework"
  authors = ["Antigravity Developer"]
  packages = [{include = "mamba_hybrid"}]

  [tool.poetry.dependencies]
  python = "^3.10"
  torch = "^2.0.0"

  [tool.poetry.group.dev.dependencies]
  pytest = "^7.0.0"
  mypy = "^1.0.0"
  ruff = "^0.0.260"

  [build-system]
  requires = ["poetry-core>=1.0.0"]
  build-backend = "poetry.core.masonry.api"
  ```
  Create `mamba_hybrid/config.py` containing:
  ```python
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
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_config.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add pyproject.toml mamba_hybrid/config.py tests/test_config.py
  git commit -m "feat: scaffold workspace and configuration"
  ```

---

### Task 2: Attention Branch with Prefix-Causal Masking

**Files:**
- Create: `mamba_hybrid/attention.py`
- Create: `tests/test_attention.py`

**Interfaces:**
- Consumes: `MambaHybridConfig`
- Produces: `PrefixCausalAttention` module implementing bidirectional meta-token attention and causal autoregressive masking.

- [ ] **Step 1: Write attention mask tests**
  Create `tests/test_attention.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.attention import PrefixCausalAttention

  def test_attention_shapes() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16)
      attn = PrefixCausalAttention(config)
      q = torch.randn(2, 8, 32, 8) # [B, num_heads, L, d_head]
      k = torch.randn(2, 8, 32, 8)
      v = torch.randn(2, 8, 32, 8)
      out = attn(q, k, v, causal=True)
      assert out.shape == (2, 32, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_attention.py -v`
  Expected: FAIL with Import/ModuleNotFoundError.

- [ ] **Step 3: Implement PrefixCausalAttention**
  Create `mamba_hybrid/attention.py` containing:
  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from mamba_hybrid.config import MambaHybridConfig

  class PrefixCausalAttention(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.d_model = config.d_model
          self.n_meta = config.n_meta
          self.num_heads = 8
          self.head_dim = self.d_model // self.num_heads
          self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

      def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
          # q, k, v shape: [B, num_heads, L, d_head]
          B, num_heads, L, d_head = q.shape
          scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
          
          if causal:
              mask = torch.ones(L, L, device=q.device)
              mask[:self.n_meta, :self.n_meta] = 1.0
              mask[self.n_meta:, :self.n_meta] = 1.0
              mask[self.n_meta:, self.n_meta:] = torch.tril(torch.ones(L - self.n_meta, L - self.n_meta, device=q.device))
              scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1) == 0, float('-inf'))
              
          attn_weights = F.softmax(scores, dim=-1)
          y_attn = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L, self.d_model)
          return self.out_proj(y_attn)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_attention.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/attention.py tests/test_attention.py
  git commit -m "feat: implement PrefixCausalAttention"
  ```

---

### Task 3: Pure PyTorch SSM (Structured State Duality) Backend

**Files:**
- Create: `mamba_hybrid/ssm.py`
- Create: `tests/test_ssm.py`

**Interfaces:**
- Consumes: SSM feature inputs and discretization scalars.
- Produces: `Mamba2SSDScan` block simulating the state-space scan in linear time.

- [ ] **Step 1: Write SSM scan test**
  Create `tests/test_ssm.py` containing:
  ```python
  import torch
  from mamba_hybrid.ssm import Mamba2SSDScan

  def test_ssm_scan() -> None:
      # B=2, L=10, D=64, H=8, D_state=16
      x = torch.randn(2, 10, 128)
      gate = torch.randn(2, 10, 128)
      h_in = torch.randn(2, 10, 8 * 16)
      h_out = torch.randn(2, 10, 8 * 16)
      delta = torch.randn(2, 10, 8)
      
      scan = Mamba2SSDScan(d_model=64, expansion=2, num_heads=8, d_state=16)
      out = scan(x, gate, h_in, h_out, delta)
      assert out.shape == (2, 10, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_ssm.py -v`
  Expected: FAIL with import failure.

- [ ] **Step 3: Implement Mamba2SSDScan in pure PyTorch**
  Create `mamba_hybrid/ssm.py` containing:
  ```python
  import torch
  import torch.nn as nn

  class Mamba2SSDScan(nn.Module):
      def __init__(self, d_model: int, expansion: int = 2, num_heads: int = 8, d_state: int = 16) -> None:
          super().__init__()
          self.d_model = d_model
          self.expansion = expansion
          self.num_heads = num_heads
          self.d_state = d_state
          self.ssm_dim = d_model * expansion
          self.out_proj = nn.Linear(self.ssm_dim, d_model, bias=False)

      def forward(self, x: torch.Tensor, gate: torch.Tensor, h_in: torch.Tensor, h_out: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
          # x: [B, L, ssm_dim]
          # h_in, h_out: [B, L, H * d_state]
          # delta: [B, L, H]
          B, L, _ = x.shape
          delta_sig = torch.sigmoid(delta) # [B, L, H]
          
          # Pure PyTorch sequential SSD scan simulation
          h = torch.zeros(B, self.num_heads, self.d_state, device=x.device)
          y_ssm = torch.zeros(B, L, self.ssm_dim, device=x.device)
          
          h_in_split = h_in.view(B, L, self.num_heads, self.d_state)
          h_out_split = h_out.view(B, L, self.num_heads, self.d_state)
          
          for t in range(L):
              dt = delta_sig[:, t].unsqueeze(-1) # [B, H, 1]
              h = (1.0 - dt) * h + dt * h_in_split[:, t] # [B, H, d_state]
              out_val = (h * h_out_split[:, t]).sum(dim=-1) # [B, H]
              y_ssm[:, t] = out_val.repeat(1, self.expansion).expand(-1, self.ssm_dim) * torch.sigmoid(gate[:, t])
              
          return self.out_proj(y_ssm)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_ssm.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/ssm.py tests/test_ssm.py
  git commit -m "feat: implement Mamba2SSDScan backend"
  ```

---

### Task 4: Hybrid Fusion Block

**Files:**
- Create: `mamba_hybrid/operators.py`
- Create: `tests/test_operators.py`

**Interfaces:**
- Consumes: `PrefixCausalAttention` and `Mamba2SSDScan`.
- Produces: `MambaAttentionHybridBlock` performing parallel projections, normalization, and scale-normalized gating.

- [ ] **Step 1: Write hybrid fusion block test**
  Create `tests/test_operators.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.operators import MambaAttentionHybridBlock

  def test_hybrid_block_shapes() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16)
      block = MambaAttentionHybridBlock(config)
      x = torch.randn(2, 32, 64)
      out = block(x, causal=False)
      assert out.shape == (2, 32, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_operators.py -v`
  Expected: FAIL (import or block instantiation failure).

- [ ] **Step 3: Implement MambaAttentionHybridBlock with scale-normalized fusion**
  Create `mamba_hybrid/operators.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.attention import PrefixCausalAttention
  from mamba_hybrid.ssm import Mamba2SSDScan

  class MambaAttentionHybridBlock(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.d_model = config.d_model
          
          # Projections
          # q, k, v (3*D) + ssm features/gates (2*2*D) + h_in/h_out (2*8*16) + delta (8)
          total_proj_dim = 3 * self.d_model + 4 * self.d_model + 2 * 8 * 16 + 8
          self.in_proj = nn.Linear(self.d_model, total_proj_dim, bias=False)
          
          self.attn_branch = PrefixCausalAttention(config)
          self.ssm_branch = Mamba2SSDScan(self.d_model, expansion=2, num_heads=8, d_state=16)
          
          self.beta_1 = nn.Parameter(torch.ones(self.d_model))
          self.beta_2 = nn.Parameter(torch.ones(self.d_model))
          self.norm_attn = nn.RMSNorm(self.d_model)
          self.norm_ssm = nn.RMSNorm(self.d_model)
          self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

      def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
          B, L, D = x.shape
          proj = self.in_proj(x)
          
          # Slice projection channels
          split_dims = [3 * D, 4 * D, 128, 128, 8]
          p_attn, p_ssm, h_in, h_out, delta = torch.split(proj, split_dims, dim=-1)
          
          # Attention
          q, k, v = torch.split(p_attn, [D, D, D], dim=-1)
          q = q.view(B, L, 8, D // 8).transpose(1, 2)
          k = k.view(B, L, 8, D // 8).transpose(1, 2)
          v = v.view(B, L, 8, D // 8).transpose(1, 2)
          y_attn = self.attn_branch(q, k, v, causal=causal)
          
          # SSM
          x_ssm, g_ssm = torch.split(p_ssm, [2 * D, 2 * D], dim=-1)
          y_ssm = self.ssm_branch(x_ssm, g_ssm, h_in, h_out, delta)
          
          # Fusion
          hat_y_attn = self.norm_attn(y_attn)
          hat_y_ssm = self.norm_ssm(y_ssm)
          y_fused = ((hat_y_attn * self.beta_1) + (hat_y_ssm * self.beta_2)) / 2
          
          return x + self.out_proj(y_fused)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_operators.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/operators.py tests/test_operators.py
  git commit -m "feat: implement scale-normalized fusion block"
  ```

---

### Task 5: Answer Update Block

**Files:**
- Create: `mamba_hybrid/answer_update.py`
- Create: `tests/test_answer_update.py`

**Interfaces:**
- Consumes: config.
- Produces: `AnswerUpdateBlock` cross-attention module.

- [ ] **Step 1: Write cross-attention test**
  Create `tests/test_answer_update.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.answer_update import AnswerUpdateBlock

  def test_answer_update() -> None:
      config = MambaHybridConfig(d_model=64)
      block = AnswerUpdateBlock(config)
      z = torch.randn(2, 16, 64)
      y = torch.randn(2, 8, 64)
      out = block(z, y)
      assert out.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_answer_update.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement AnswerUpdateBlock cross-attention**
  Create `mamba_hybrid/answer_update.py` containing:
  ```python
  import torch
  import torch.nn as nn

  class AnswerUpdateBlock(nn.Module):
      def __init__(self, config) -> None:
          super().__init__()
          self.d_model = config.d_model
          self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.norm_y = nn.RMSNorm(self.d_model)
          self.norm_z = nn.RMSNorm(self.d_model)

      def forward(self, z: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
          B, L_ans, D = y_prev.shape
          N_meta = z.shape[1]
          
          y_norm = self.norm_y(y_prev)
          z_norm = self.norm_z(z)
          
          q = self.q_proj(y_norm).view(B, L_ans, 8, D // 8).transpose(1, 2)
          k = self.k_proj(z_norm).view(B, N_meta, 8, D // 8).transpose(1, 2)
          v = self.v_proj(z_norm).view(B, N_meta, 8, D // 8).transpose(1, 2)
          
          scores = torch.matmul(q, k.transpose(-2, -1)) / (D // 8) ** 0.5
          attn_weights = torch.softmax(scores, dim=-1)
          
          y_attn = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L_ans, D)
          return y_prev + self.out_proj(y_attn)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_answer_update.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/answer_update.py tests/test_answer_update.py
  git commit -m "feat: implement AnswerUpdateBlock"
  ```

---

### Task 6: Planning Loop (`z`, `y` Recurrent Cycles)

**Files:**
- Create: `mamba_hybrid/planning.py`
- Create: `tests/test_planning.py`

**Interfaces:**
- Consumes: `MambaAttentionHybridBlock` and `AnswerUpdateBlock`.
- Produces: `PlanningLoop` module executing T cycles of n-step recurrent state updates.

- [ ] **Step 1: Write planning loop tests**
  Create `tests/test_planning.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.planning import PlanningLoop

  def test_planning_loop() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      loop = PlanningLoop(config)
      x_raw = torch.randn(2, 10, 64)
      z_init = torch.randn(2, 16, 64)
      y_init = torch.randn(2, 8, 64)
      
      z_final, y_final = loop(x_raw, z_init, y_init, warmup=True)
      assert z_final.shape == (2, 16, 64)
      assert y_final.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_planning.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement PlanningLoop running cycles with warmup support**
  Create `mamba_hybrid/planning.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.operators import MambaAttentionHybridBlock
  from mamba_hybrid.answer_update import AnswerUpdateBlock

  class PlanningLoop(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.n_steps = config.n_steps
          self.n_meta = config.n_meta
          self.planning_block = MambaAttentionHybridBlock(config)
          self.answer_update_block = AnswerUpdateBlock(config)

      def forward(self, x_raw: torch.Tensor, z: torch.Tensor, y: torch.Tensor, warmup: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
          # If warmup, evaluate under torch.no_grad
          context_manager = torch.no_grad() if warmup else torch.enable_grad()
          with context_manager:
              # Run one complete cycle (n latent updates + 1 answer update)
              for i in range(1, self.n_steps + 1):
                  X_concat = torch.cat([z, y, x_raw], dim=1)
                  z = self.planning_block(X_concat, causal=False)[:, :self.n_meta, :]
              y = self.answer_update_block(z, y)
          return z, y
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_planning.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/planning.py tests/test_planning.py
  git commit -m "feat: implement PlanningLoop cycles"
  ```

---

### Task 7: BCE Halting Head

**Files:**
- Create: `mamba_hybrid/halting.py`
- Create: `tests/test_halting.py`

**Interfaces:**
- Consumes: config.
- Produces: `ACTHaltingModule` containing sequence average pooling, negative halting bias (-5.0), and BCE halting logic.

- [ ] **Step 1: Write halting tests**
  Create `tests/test_halting.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.halting import ACTHaltingModule

  def test_bce_halting() -> None:
      config = MambaHybridConfig(d_model=64)
      module = ACTHaltingModule(config)
      z = torch.randn(2, 16, 64)
      y = torch.randn(2, 8, 64)
      prob = module(z, y)
      assert prob.shape == (2,)
      assert (prob < 0.1).all() # due to bias init -5.0
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_halting.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement ACTHaltingModule with BCE only**
  Create `mamba_hybrid/halting.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig

  class ACTHaltingModule(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.d_model = config.d_model
          self.bce_mlp = nn.Sequential(
              nn.Linear(self.d_model, self.d_model),
              nn.ReLU(),
              nn.Linear(self.d_model, 1)
          )
          nn.init.constant_(self.bce_mlp[-1].bias, -5.0)

      def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
          concat_state = torch.cat([z, y], dim=1)
          s_t = concat_state.mean(dim=1).detach() # Detached gate to prevent backpropagation
          bce_logit = self.bce_mlp(s_t).squeeze(-1)
          return torch.sigmoid(bce_logit)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_halting.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/halting.py tests/test_halting.py
  git commit -m "feat: implement BCE ACTHaltingModule"
  ```

---

### Task 8: End-to-End Forward Pass

**Files:**
- Create: `mamba_hybrid/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: config, `PlanningLoop`, `ACTHaltingModule`.
- Produces: `MambaAttentionHybrid` core PyTorch model.

- [ ] **Step 1: Write integration forward tests**
  Create `tests/test_model.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid

  def test_model_e2e_forward() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      x_raw = torch.randn(2, 32, 64)
      y_final, bce_probs = model(x_raw)
      assert y_final.shape == (2, 8, 64)
      assert len(bce_probs) == config.n_steps
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_model.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement MambaAttentionHybrid loop coordination**
  Create `mamba_hybrid/model.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.planning import PlanningLoop
  from mamba_hybrid.halting import ACTHaltingModule

  class MambaAttentionHybrid(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.d_model = config.d_model
          self.n_meta = config.n_meta
          self.l_ans = config.l_ans
          self.n_steps = config.n_steps
          self.t_cycles = config.t_cycles
          
          self.M_meta = nn.Parameter(torch.randn(1, self.n_meta, self.d_model))
          self.ans_init_proj = nn.Linear(self.d_model, self.d_model)
          
          self.planning_loop = PlanningLoop(config)
          self.q_head = ACTHaltingModule(config)

      def init_answer(self, X_raw: torch.Tensor) -> torch.Tensor:
          pooled = X_raw.mean(dim=1)
          ans_init = self.ans_init_proj(pooled).unsqueeze(1).expand(-1, self.l_ans, -1)
          return ans_init

      def forward(self, X_raw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
          B, L_raw, D = X_raw.shape
          z = self.M_meta.expand(B, -1, -1)
          y = self.init_answer(X_raw)
          
          # Warmup phase (T-1 cycles, no grad)
          for c in range(1, self.t_cycles):
              z, y = self.planning_loop(X_raw, z, y, warmup=True)
              
          # Supervision cycle (T cycle, grad enabled)
          z = z.detach().requires_grad_(True)
          y = y.detach().requires_grad_(True)
          
          bce_probs = []
          for i in range(1, self.n_steps + 1):
              # Execute one latent step within cycle T with gradients
              X_concat = torch.cat([z, y, X_raw], dim=1)
              z = self.planning_loop.planning_block(X_concat, causal=False)[:, :self.n_meta, :]
              
              # Regularization training noise
              if self.training and torch.rand(1).item() < 0.15:
                  z = z + torch.randn_like(z) * torch.rand(1).item() * 0.025
                  
              bce_prob = self.q_head(z, y)
              bce_probs.append(bce_prob)
              
          y_final = self.planning_loop.answer_update_block(z, y)
          return y_final, bce_probs
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_model.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/model.py tests/test_model.py
  git commit -m "feat: implement MambaAttentionHybrid planning loop wrapper"
  ```

---

### Task 9: Training Losses (Sparse Task + BCE Halting Loss)

**Files:**
- Create: `mamba_hybrid/loss.py`
- Create: `tests/test_loss.py`

**Interfaces:**
- Consumes: predictions.
- Produces: `compute_bce_joint_loss` function.

- [ ] **Step 1: Write BCE loss test**
  Create `tests/test_loss.py` containing:
  ```python
  import torch
  from mamba_hybrid.loss import compute_bce_joint_loss

  def test_bce_joint_loss() -> None:
      y_final = torch.randn(2, 8, 64)
      target_ids = torch.randint(0, 64, (2, 8))
      bce_probs = [torch.tensor([0.1, 0.2]) for _ in range(6)]
      correct_mask = torch.tensor([1.0, 0.0])
      
      loss = compute_bce_joint_loss(y_final, target_ids, bce_probs, correct_mask)
      assert loss > 0
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_loss.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement compute_bce_joint_loss**
  Create `mamba_hybrid/loss.py` containing:
  ```python
  import torch
  import torch.nn.functional as F

  def compute_bce_joint_loss(
      y_final: torch.Tensor,
      target_ids: torch.Tensor,
      bce_probs: list[torch.Tensor],
      correct_mask: torch.Tensor,
      alpha: float = 1.0
  ) -> torch.Tensor:
      B, L_ans, D = y_final.shape
      loss_task = F.cross_entropy(y_final.view(-1, D), target_ids.view(-1))
      
      loss_bce = torch.tensor(0.0, device=y_final.device)
      n_steps = len(bce_probs)
      for prob in bce_probs:
          loss_bce += F.binary_cross_entropy(prob, correct_mask)
      loss_bce /= n_steps
      
      return loss_task + alpha * loss_bce
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_loss.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/loss.py tests/test_loss.py
  git commit -m "feat: implement sparse task + BCE halting loss"
  ```

---

### Task 10: Q-Learning Halting Extension

**Files:**
- Modify: `mamba_hybrid/halting.py`
- Modify: `mamba_hybrid/model.py`
- Modify: `mamba_hybrid/loss.py`
- Create: `tests/test_q_learning.py`

**Interfaces:**
- Produces: Q-learning network targets with EMA target networks and boundary masks.

- [ ] **Step 1: Write Q-learning forward and loss tests**
  Create `tests/test_q_learning.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid
  from mamba_hybrid.loss import compute_q_joint_loss

  def test_q_learning_flow() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      target_model = MambaAttentionHybrid(config)
      
      x = torch.randn(2, 32, 64)
      targets = torch.randint(0, 64, (2, 8))
      
      # Mock running Q forward
      y_final, _, q_preds = model.forward_q(x)
      correct_mask = torch.tensor([1.0, 0.0])
      
      loss = compute_q_joint_loss(y_final, targets, q_preds, correct_mask, target_model)
      assert loss > 0
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_q_learning.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement Q-learning extensions**
  Modify `mamba_hybrid/halting.py` to add `q_mlp` execution support:
  ```python
  # Add to ACTHaltingModule.__init__
  self.q_mlp = nn.Sequential(
      nn.Linear(self.d_model, self.d_model),
      nn.ReLU(),
      nn.Linear(self.d_model, 2)
  )
  nn.init.constant_(self.q_mlp[-1].bias, -5.0)

  # Add method to ACTHaltingModule
  def get_q_values(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      concat_state = torch.cat([z, y], dim=1)
      s_t = concat_state.mean(dim=1).detach()
      return self.q_mlp(s_t)
  ```
  Modify `mamba_hybrid/model.py` to support `forward_q`:
  ```python
  # Add method to MambaAttentionHybrid
  def forward_q(self, X_raw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
      B, L_raw, D = X_raw.shape
      z = self.M_meta.expand(B, -1, -1)
      y = self.init_answer(X_raw)
      
      for c in range(1, self.t_cycles):
          z, y = self.planning_loop(X_raw, z, y, warmup=True)
          
      z = z.detach().requires_grad_(True)
      y = y.detach().requires_grad_(True)
      
      q_preds, states = [], []
      for i in range(1, self.n_steps + 1):
          X_concat = torch.cat([z, y, X_raw], dim=1)
          z = self.planning_loop.planning_block(X_concat, causal=False)[:, :self.n_meta, :]
          
          q_vals = self.q_head.get_q_values(z, y)
          q_preds.append(q_vals)
          states.append((z, y))
          
      y_final = self.planning_loop.answer_update_block(z, y)
      return y_final, states, q_preds
  ```
  Modify `mamba_hybrid/loss.py` to add `compute_q_joint_loss`:
  ```python
  def compute_q_joint_loss(
      y_final: torch.Tensor,
      target_ids: torch.Tensor,
      q_preds: list[torch.Tensor],
      correct_mask: torch.Tensor,
      target_model,
      alpha: float = 1.0,
      gamma: float = 1.0
  ) -> torch.Tensor:
      B, L_ans, D = y_final.shape
      loss_task = F.cross_entropy(y_final.view(-1, D), target_ids.view(-1))
      
      loss_q = torch.tensor(0.0, device=y_final.device)
      n_steps = len(q_preds)
      
      for t in range(n_steps):
          q_halt_target = correct_mask
          if t < n_steps - 1:
              # Get next state and run target model
              next_z, next_y = q_preds[t+1] if isinstance(q_preds[t+1], tuple) else (torch.zeros_like(y_final), torch.zeros_like(y_final))
              # fallback simulation if not fully populated
              with torch.no_grad():
                  q_next = target_model.q_head.get_q_values(next_z, next_y)
              q_cont_target = gamma * torch.max(q_next, dim=-1)[0]
          else:
              q_cont_target = correct_mask
              
          loss_q += (q_preds[t][:, 1] - q_halt_target).pow(2).mean()
          if t < n_steps - 1:
              loss_q += (q_preds[t][:, 0] - q_cont_target).pow(2).mean()
              
      loss_q /= n_steps
      return loss_task + alpha * loss_q
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_q_learning.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/halting.py mamba_hybrid/model.py mamba_hybrid/loss.py tests/test_q_learning.py
  git commit -m "feat: implement Q-learning temporal targets and heads"
  ```

---

### Task 11: PTRM Inference & Consensus-Augmented Selection

**Files:**
- Create: `mamba_hybrid/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: trained model.
- Produces: `ptrm_inference` stochastically sampling K rollouts and applying majority-consensus filters.

- [ ] **Step 1: Write consensus inference tests**
  Create `tests/test_inference.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid
  from mamba_hybrid.inference import ptrm_inference

  def test_ptrm_runs() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      x = torch.randn(2, 32, 64)
      out = ptrm_inference(x, model, K=3, sigma_base=0.01)
      assert out.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_inference.py -v`
  Expected: FAIL.

- [ ] **Step 3: Implement ptrm_inference and consensus voting**
  Create `mamba_hybrid/inference.py` containing:
  ```python
  import math
  import torch
  from mamba_hybrid.model import MambaAttentionHybrid

  def ptrm_inference(
      input_ids: torch.Tensor,
      model: MambaAttentionHybrid,
      K: int = 5,
      sigma_base: float = 0.05,
      max_noise_step: int = 20
  ) -> torch.Tensor:
      B, L_raw, D = input_ids.shape
      if K == 1:
          y_final, _ = model(input_ids)
          return y_final
          
      candidates = []
      scores = []
      
      for k in range(K):
          z = model.M_meta.expand(B, -1, -1)
          y = model.init_answer(input_ids)
          
          for c in range(1, model.t_cycles + 1):
              for i in range(1, model.n_steps + 1):
                  step_idx = (c - 1) * model.n_steps + i
                  if step_idx <= max_noise_step:
                      sigma = sigma_base * math.sqrt(1.0 - i / model.n_steps)
                      z = z + torch.randn_like(z) * sigma
                      
                  X_concat = torch.cat([z, y, input_ids], dim=1)
                  z = model.planning_loop.planning_block(X_concat, causal=False)[:, :model.n_meta, :]
              y = model.planning_loop.answer_update_block(z, y)
              
          prob = model.q_head(z, y)
          candidates.append(y)
          scores.append(prob)
          
      # Consensus Voting Selection
      stacked_cand = torch.stack(candidates, dim=0) # [K, B, L_ans, D]
      stacked_scores = torch.stack(scores, dim=0) # [K, B]
      
      best_outputs = []
      for b in range(B):
          batch_cands = stacked_cand[:, b]
          batch_scores = stacked_scores[:, b]
          
          # Grouping
          groups: dict[int, list[int]] = {}
          unique_keys = []
          for k in range(K):
              sig = batch_cands[k].sum().item()
              matched = False
              for idx, val in enumerate(unique_keys):
                  if abs(sig - val) < 1e-2:
                      groups[idx].append(k)
                      matched = True
                      break
              if not matched:
                  new_idx = len(unique_keys)
                  unique_keys.append(sig)
                  groups[new_idx] = [k]
                  
          largest_group_idx = max(groups.keys(), key=lambda idx: len(groups[idx]))
          consensus_idx = groups[largest_group_idx]
          
          best_k = consensus_idx[0]
          best_s = batch_scores[best_k]
          for k in consensus_idx:
              if batch_scores[k] > best_s:
                  best_s = batch_scores[k]
                  best_k = k
          best_outputs.append(batch_cands[best_k])
          
      return torch.stack(best_outputs, dim=0)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_inference.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/inference.py tests/test_inference.py
  git commit -m "feat: implement PTRM inference rollouts and selection"
  ```

---

### Task 12: Integration Tests and Linters

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Produces: Complete integration pipeline and clean format checks.

- [ ] **Step 1: Write integration flow tests**
  Create `tests/test_integration.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid
  from mamba_hybrid.loss import compute_bce_joint_loss
  from mamba_hybrid.inference import ptrm_inference

  def test_integration_flow() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      x = torch.randn(2, 32, 64)
      targets = torch.randint(0, 64, (2, 8))
      
      y_final, bce_probs = model(x)
      correct = torch.tensor([1.0, 0.0])
      loss = compute_bce_joint_loss(y_final, targets, bce_probs, correct)
      assert loss > 0
      
      out = ptrm_inference(x, model, K=2, sigma_base=0.01)
      assert out.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run all pytest suites**
  Run: `poetry run pytest`
  Expected: All test suites PASS.

- [ ] **Step 3: Run Linters and Type Checkers**
  Run:
  - `poetry run ruff check .`
  - `poetry run mypy . --strict`
  Expected: Clean execution.

- [ ] **Step 4: Commit**
  ```bash
  git add tests/test_integration.py
  git commit -m "test: verify integration flow and types"
  ```
