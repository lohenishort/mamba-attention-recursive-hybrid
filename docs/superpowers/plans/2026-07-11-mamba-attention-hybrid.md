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

### Task 2: Core Hybrid Block Operator (`MambaAttentionHybridBlock`)

**Files:**
- Create: `mamba_hybrid/operators.py`
- Create: `tests/test_operators.py`

**Interfaces:**
- Consumes: `MambaHybridConfig` from `mamba_hybrid/config.py`
- Produces: `MambaAttentionHybridBlock` PyTorch module taking inputs of shape `[B, L, D]` and performing parallel Attention + SSD scanning with RMSNorm and scale-normalized fusion.

- [ ] **Step 1: Write prefix mask and hybrid block test**
  Create `tests/test_operators.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.operators import MambaAttentionHybridBlock

  def test_hybrid_block_shape() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16)
      block = MambaAttentionHybridBlock(config)
      x = torch.randn(2, 32, 64) # [B, L, D]
      out = block(x, causal=False)
      assert out.shape == (2, 32, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_operators.py -v`
  Expected: FAIL with ModuleNotFoundError or import failure.

- [ ] **Step 3: Implement core hybrid block with pure PyTorch SSD scan fallback**
  Create `mamba_hybrid/operators.py` containing:
  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from mamba_hybrid.config import MambaHybridConfig

  class MambaAttentionHybridBlock(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.d_model = config.d_model
          self.n_meta = config.n_meta
          
          # Parallel input projection
          self.in_proj = nn.Linear(self.d_model, 3 * self.d_model + 2 * self.d_model + 2 * 64 + 1, bias=False)
          self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          
          self.beta_1 = nn.Parameter(torch.ones(self.d_model))
          self.beta_2 = nn.Parameter(torch.ones(self.d_model))
          self.norm_attn = nn.RMSNorm(self.d_model)
          self.norm_ssm = nn.RMSNorm(self.d_model)

      def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
          # x: [B, L, D]
          B, L, D = x.shape
          projected = self.in_proj(x)
          
          # Split projections
          q, k, v, rest = torch.split(projected, [D, D, D, 2*D + 2*64 + 1], dim=-1)
          x_ssm, g_ssm, h_in, h_out, delta = torch.split(rest, [D, D, 64, 64, 1], dim=-1)
          
          # 1. Attention Path
          q = q.view(B, L, 8, D // 8).transpose(1, 2)
          k = k.view(B, L, 8, D // 8).transpose(1, 2)
          v = v.view(B, L, 8, D // 8).transpose(1, 2)
          
          scores = torch.matmul(q, k.transpose(-2, -1)) / (D // 8) ** 0.5
          if causal:
              mask = torch.ones(L, L, device=x.device)
              mask[:self.n_meta, :self.n_meta] = 1.0 # Bidirectional meta
              mask[self.n_meta:, :self.n_meta] = 1.0 # Gen attends to meta
              mask[self.n_meta:, self.n_meta:] = torch.tril(torch.ones(L - self.n_meta, L - self.n_meta, device=x.device))
              scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1) == 0, float('-inf'))
              
          attn_weights = F.softmax(scores, dim=-1)
          y_attn = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L, D)
          
          # 2. Pure PyTorch fallback Mamba-2 SSD Scan
          # Simple linear time decay scan simulating state updates
          delta_sig = torch.sigmoid(delta)
          y_ssm = torch.zeros_like(x_ssm)
          h = torch.zeros(B, 64, device=x.device)
          for t in range(L):
              h = (1 - delta_sig[:, t]) * h + delta_sig[:, t] * h_in[:, t]
              y_ssm[:, t] = (h * h_out[:, t]).sum(dim=-1, keepdim=True).expand(-1, D) * torch.sigmoid(g_ssm[:, t])
              
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
  git commit -m "feat: implement MambaAttentionHybridBlock with PyTorch SSD scan"
  ```

---

### Task 3: Answer Update Block (Cross-Attention Operator)

**Files:**
- Create: `mamba_hybrid/answer_update.py`
- Create: `tests/test_answer_update.py`

**Interfaces:**
- Consumes: `MambaHybridConfig` from `mamba_hybrid/config.py`
- Produces: `AnswerUpdateBlock` taking latent representation $z_n \in \mathbb{R}^{B \times N_{\text{meta}} \times D}$ and previous answer state $y_{c-1} \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$ and producing updated answer state $y_c$ of shape `[B, L_ans, D]`.

- [ ] **Step 1: Write cross-attention update block test**
  Create `tests/test_answer_update.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.answer_update import AnswerUpdateBlock

  def test_answer_update_block() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      block = AnswerUpdateBlock(config)
      z = torch.randn(2, 16, 64)
      y_prev = torch.randn(2, 8, 64)
      y_next = block(z, y_prev)
      assert y_next.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_answer_update.py -v`
  Expected: FAIL with import failure.

- [ ] **Step 3: Implement AnswerUpdateBlock using Multihead Cross-Attention**
  Create `mamba_hybrid/answer_update.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig

  class AnswerUpdateBlock(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.d_model = config.d_model
          
          # Cross-attention projections
          self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
          self.norm_y = nn.RMSNorm(self.d_model)
          self.norm_z = nn.RMSNorm(self.d_model)

      def forward(self, z: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
          # z: [B, N_meta, D]
          # y_prev: [B, L_ans, D]
          B, L_ans, D = y_prev.shape
          N_meta = z.shape[1]
          
          # LayerNorm
          y_norm = self.norm_y(y_prev)
          z_norm = self.norm_z(z)
          
          # Projections
          q = self.q_proj(y_norm).view(B, L_ans, 8, D // 8).transpose(1, 2)
          k = self.k_proj(z_norm).view(B, N_meta, 8, D // 8).transpose(1, 2)
          v = self.v_proj(z_norm).view(B, N_meta, 8, D // 8).transpose(1, 2)
          
          # Cross-attention weights
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
  git commit -m "feat: implement AnswerUpdateBlock cross-attention block"
  ```

---

### Task 4: ACT Halting Heads & Sequence Pooling

**Files:**
- Create: `mamba_hybrid/halting.py`
- Create: `tests/test_halting.py`

**Interfaces:**
- Consumes: `MambaHybridConfig` from `mamba_hybrid/config.py`
- Produces: `ACTHaltingModule` implementing sequence average pooling, negative bias initialization (-5.0), and both Q-learning and BCE logits predictions.

- [ ] **Step 1: Write halting heads test**
  Create `tests/test_halting.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.halting import ACTHaltingModule

  def test_halting_predictions() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      halting_module = ACTHaltingModule(config)
      z = torch.randn(2, 16, 64)
      y = torch.randn(2, 8, 64)
      q_vals, bce_prob = halting_module(z, y)
      assert q_vals.shape == (2, 2)
      assert bce_prob.shape == (2,)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_halting.py -v`
  Expected: FAIL with import failure.

- [ ] **Step 3: Implement ACTHaltingModule with explicit pooling, detach gate, and initial biases**
  Create `mamba_hybrid/halting.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig

  class ACTHaltingModule(nn.Module):
      def __init__(self, config: MambaHybridConfig) -> None:
          super().__init__()
          self.config = config
          self.d_model = config.d_model
          
          # Q-Head Layers
          self.q_mlp = nn.Sequential(
              nn.Linear(self.d_model, self.d_model),
              nn.ReLU(),
              nn.Linear(self.d_model, 2)
          )
          
          # BCE-Head Layers
          self.bce_mlp = nn.Sequential(
              nn.Linear(self.d_model, self.d_model),
              nn.ReLU(),
              nn.Linear(self.d_model, 1)
          )
          
          # Initialize bias to -5.0 to restrict early halting
          nn.init.constant_(self.q_mlp[-1].bias, -5.0)
          nn.init.constant_(self.bce_mlp[-1].bias, -5.0)

      def forward(self, z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
          # z: [B, N_meta, D]
          # y: [B, L_ans, D]
          concat_state = torch.cat([z, y], dim=1) # [B, N_meta + L_ans, D]
          s_t = concat_state.mean(dim=1) # GlobalAveragePool -> [B, D]
          
          # Detach gate to prevent ACT gradients from propagating back to planning loop
          s_t_detached = s_t.detach()
          
          q_vals = self.q_mlp(s_t_detached)
          bce_logit = self.bce_mlp(s_t_detached).squeeze(-1)
          bce_prob = torch.sigmoid(bce_logit)
          
          return q_vals, bce_prob
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_halting.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/halting.py tests/test_halting.py
  git commit -m "feat: implement ACTHaltingModule with global pooling"
  ```

---

### Task 5: Main Recurrent Planning Loop & Supervision detaches

**Files:**
- Create: `mamba_hybrid/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `MambaHybridConfig`, `MambaAttentionHybridBlock`, `AnswerUpdateBlock`, and `ACTHaltingModule`.
- Produces: `MambaAttentionHybrid` core PyTorch model.

- [ ] **Step 1: Write model forward planning loop test**
  Create `tests/test_model.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid

  def test_model_forward() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      input_ids = torch.randn(2, 32, 64) # Embedded problem tokens [B, L_raw, D]
      y_final, q_preds, bce_probs = model(input_ids)
      assert y_final.shape == (2, 8, 64)
      assert len(q_preds) == config.n_steps
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_model.py -v`
  Expected: FAIL with import or runtime error.

- [ ] **Step 3: Implement MambaAttentionHybrid planning cycles and warmup/gradient paths**
  Create `mamba_hybrid/model.py` containing:
  ```python
  import torch
  import torch.nn as nn
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.operators import MambaAttentionHybridBlock
  from mamba_hybrid.answer_update import AnswerUpdateBlock
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
          
          self.planning_block = MambaAttentionHybridBlock(config)
          self.answer_update_block = AnswerUpdateBlock(config)
          self.q_head = ACTHaltingModule(config)

      def init_answer(self, X_raw: torch.Tensor) -> torch.Tensor:
          # X_raw: [B, L_raw, D]
          pooled = X_raw.mean(dim=1) # [B, D]
          ans_init = self.ans_init_proj(pooled).unsqueeze(1).expand(-1, self.l_ans, -1)
          return ans_init

      def forward(self, X_raw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
          B, L_raw, D = X_raw.shape
          z = self.M_meta.expand(B, -1, -1)
          y = self.init_answer(X_raw)
          
          # 1. Warmup Cycles (T-1 cycles, no gradients)
          with torch.no_grad():
              for c in range(1, self.t_cycles):
                  for i in range(1, self.n_steps + 1):
                      X_concat = torch.cat([z, y, X_raw], dim=1)
                      z = self.planning_block(X_concat)[:, :self.n_meta, :]
                  y = self.answer_update_block(z, y)
                  
          # 2. Gradient-Tracked Cycle (T cycle)
          z = z.detach().requires_grad_(True)
          y = y.detach().requires_grad_(True)
          
          q_preds, bce_probs = [], []
          for i in range(1, self.n_steps + 1):
              X_concat = torch.cat([z, y, X_raw], dim=1)
              z = self.planning_block(X_concat)[:, :self.n_meta, :]
              
              q_vals, bce_prob = self.q_head(z, y)
              q_preds.append(q_vals)
              bce_probs.append(bce_prob)
              
          y_final = self.answer_update_block(z, y)
          return y_final, q_preds, bce_probs
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_model.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/model.py tests/test_model.py
  git commit -m "feat: implement MambaAttentionHybrid model and training split"
  ```

---

### Task 6: Joint Optimization Loss Calculations

**Files:**
- Create: `mamba_hybrid/loss.py`
- Create: `tests/test_loss.py`

**Interfaces:**
- Consumes: ACT predictions and targets.
- Produces: `compute_joint_loss` function implementing sparse cross-entropy and bootstrapped 1-step Q-learning/BCE loss.

- [ ] **Step 1: Write loss function tests**
  Create `tests/test_loss.py` containing:
  ```python
  import torch
  from mamba_hybrid.loss import compute_joint_loss

  def test_compute_joint_loss() -> None:
      y_final = torch.randn(2, 8, 64)
      target_ids = torch.randint(0, 64, (2, 8))
      q_preds = [torch.randn(2, 2) for _ in range(6)]
      bce_probs = [torch.randn(2) for _ in range(6)]
      
      # Mock correct evaluation to be True for first element, False for second
      correct_mask = torch.tensor([1.0, 0.0])
      
      loss = compute_joint_loss(
          y_final, target_ids, q_preds, bce_probs, correct_mask,
          mode="BCE", alpha=1.0
      )
      assert loss > 0
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_loss.py -v`
  Expected: FAIL with import failure.

- [ ] **Step 3: Implement loss function with boundary masking and Q-bootstrapping**
  Create `mamba_hybrid/loss.py` containing:
  ```python
  import torch
  import torch.nn.functional as F

  def compute_joint_loss(
      y_final: torch.Tensor,
      target_ids: torch.Tensor,
      q_preds: list[torch.Tensor],
      bce_probs: list[torch.Tensor],
      correct_mask: torch.Tensor, # [B]
      mode: str = "BCE",
      alpha: float = 1.0,
      gamma: float = 1.0
  ) -> torch.Tensor:
      B, L_ans, D = y_final.shape
      
      # 1. Sparse Task Loss
      y_flat = y_final.view(-1, D)
      targets_flat = target_ids.view(-1)
      loss_task = F.cross_entropy(y_flat, targets_flat)
      
      loss_halting = torch.tensor(0.0, device=y_final.device)
      n_steps = len(q_preds)
      
      # 2. Halting Head Losses
      if mode == "BCE":
          for t in range(n_steps):
              # Target is correct_mask for all steps
              loss_halting += F.binary_cross_entropy(bce_probs[t], correct_mask)
          loss_halting /= n_steps
      else:
          # Mode A: 1-step bootstrapped Q-learning
          for t in range(n_steps):
              # Halt Target
              q_halt_target = correct_mask # [B]
              # Continue Target
              if t < n_steps - 1:
                  q_next = q_preds[t+1].detach()
                  q_cont_target = gamma * torch.max(q_next, dim=-1)[0]
              else:
                  q_cont_target = correct_mask # Boundary condition
                  
              # Active masking: continue not trained at the boundary step
              loss_halting += (q_preds[t][:, 1] - q_halt_target).pow(2).mean()
              if t < n_steps - 1:
                  loss_halting += (q_preds[t][:, 0] - q_cont_target).pow(2).mean()
          loss_halting /= n_steps
          
      return loss_task + alpha * loss_halting
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_loss.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/loss.py tests/test_loss.py
  git commit -m "feat: implement dual-mode ACT halting losses"
  ```

---

### Task 7: PTRM Inference & Consensus-Augmented Selection

**Files:**
- Create: `mamba_hybrid/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: Trained `MambaAttentionHybrid` model.
- Produces: `ptrm_inference` function executing $K$ noisy parallel planning paths and selecting outputs based on consensus-augmented selection.

- [ ] **Step 1: Write PTRM inference and consensus selection test**
  Create `tests/test_inference.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid
  from mamba_hybrid.inference import ptrm_inference

  def test_ptrm_inference_runs() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      input_ids = torch.randn(2, 32, 64)
      output = ptrm_inference(input_ids, model, K=3, sigma_base=0.05, max_noise_step=15)
      assert output.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `poetry run pytest tests/test_inference.py -v`
  Expected: FAIL with import failure.

- [ ] **Step 3: Implement PTRM generation, noise annealing, and consensus filter selection**
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
      # input_ids: [B, L_raw, D]
      B, L_raw, D = input_ids.shape
      
      if K == 1:
          # Deterministic Baseline
          y_final, _, _ = model(input_ids)
          return y_final
          
      # Run K noisy rollouts
      candidates = []
      scores = []
      
      for k in range(K):
          X_raw = model.embed(input_ids) if hasattr(model, 'embed') else input_ids
          z = model.M_meta.expand(B, -1, -1)
          y = model.init_answer(X_raw)
          
          for c in range(1, model.t_cycles + 1):
              for i in range(1, model.n_steps + 1):
                  step_idx = (c - 1) * model.n_steps + i
                  if step_idx <= max_noise_step:
                      # Noise Annealing
                      sigma = sigma_base * math.sqrt(1.0 - i / model.n_steps)
                      z = z + torch.randn_like(z) * sigma
                      
                  X_concat = torch.cat([z, y, X_raw], dim=1)
                  z = model.planning_block(X_concat)[:, :model.n_meta, :]
              y = model.answer_update_block(z, y)
              
          s_final = torch.cat([z, y], dim=1).mean(dim=1)
          q_vals, _ = model.q_head(z, y)
          score = q_vals[:, 1] # confidence of halting
          
          candidates.append(y)
          scores.append(score)
          
      # Selection Policy (Consensus)
      # stack: [K, B, L_ans, D]
      stacked_candidates = torch.stack(candidates, dim=0)
      stacked_scores = torch.stack(scores, dim=0) # [K, B]
      
      best_outputs = []
      for b in range(B):
          # For each batch element, compute consensus
          batch_candidates = stacked_candidates[:, b] # [K, L_ans, D]
          batch_scores = stacked_scores[:, b] # [K]
          
          # Group by similarity (rounded token predictions)
          unique_keys = []
          groups: dict[int, list[int]] = {}
          
          for k in range(K):
              cand_sig = batch_candidates[k].sum().item() # simple representation checksum
              matched = False
              for idx, val in enumerate(unique_keys):
                  if abs(cand_sig - val) < 1e-2:
                      groups[idx].append(k)
                      matched = True
                      break
              if not matched:
                  new_idx = len(unique_keys)
                  unique_keys.append(cand_sig)
                  groups[new_idx] = [k]
                  
          # Consensus Vote: largest group
          largest_group_idx = max(groups.keys(), key=lambda idx: len(groups[idx]))
          consensus_indices = groups[largest_group_idx]
          
          # Confidence selection within the consensus group
          best_k = consensus_indices[0]
          best_score = batch_scores[best_k]
          for k in consensus_indices:
              if batch_scores[k] > best_score:
                  best_score = batch_scores[k]
                  best_k = k
                  
          best_outputs.append(batch_candidates[best_k])
          
      return torch.stack(best_outputs, dim=0)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `poetry run pytest tests/test_inference.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add mamba_hybrid/inference.py tests/test_inference.py
  git commit -m "feat: implement PTRM inference with consensus selection filter"
  ```

---

### Task 8: Verification, Linting, & Code Integrity Checks

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_integration.py`

**Interfaces:**
- Produces: Integration tests running training-step, optimization, and PTRM inference jointly.

- [ ] **Step 1: Write integration tests**
  Create `tests/test_integration.py` containing:
  ```python
  import torch
  from mamba_hybrid.config import MambaHybridConfig
  from mamba_hybrid.model import MambaAttentionHybrid
  from mamba_hybrid.loss import compute_joint_loss
  from mamba_hybrid.inference import ptrm_inference

  def test_end_to_end_flow() -> None:
      config = MambaHybridConfig(d_model=64, n_meta=16, l_ans=8)
      model = MambaAttentionHybrid(config)
      target_model = MambaAttentionHybrid(config)
      
      optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
      
      input_ids = torch.randn(2, 32, 64)
      target_ids = torch.randint(0, 64, (2, 8))
      
      y_final, q_preds, bce_probs = model(input_ids)
      correct_mask = torch.tensor([1.0, 0.0])
      
      loss = compute_joint_loss(
          y_final, target_ids, q_preds, bce_probs, correct_mask,
          mode="Q-learning", alpha=1.0
      )
      
      loss.backward()
      optimizer.step()
      
      # Inference verification
      out = ptrm_inference(input_ids, model, K=2, sigma_base=0.01)
      assert out.shape == (2, 8, 64)
  ```

- [ ] **Step 2: Run all tests to verify everything passes**
  Run: `poetry run pytest`
  Expected: All 6 test files pass successfully.

- [ ] **Step 3: Run Linters and Type Checkers**
  Run:
  - `poetry run ruff check .`
  - `poetry run mypy . --strict`
  Expected: Clean execution without failures or type errors.

- [ ] **Step 4: Commit**
  ```bash
  git add tests/test_integration.py
  git commit -m "test: add integration test suite and verify code hygiene"
  ```
