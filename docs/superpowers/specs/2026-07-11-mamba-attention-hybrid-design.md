# Engineering & Evaluation Spec: Mamba-Attention Recursive Reasoning Hybrid

This document outlines the architecture, training procedure, and evaluation pipeline for the Mamba-Attention Recursive Reasoning Hybrid framework. The core objective is to design a model that separates latent constraint propagation ("thinking slow") from autoregressive sequence printing ("thinking fast"), stabilized by Adaptive Computation Time (ACT) and stochastically scaled at test-time using the Probabilistic Tiny Recursive Model (PTRM) approach.

---

## 1. Core Hybrid Operator & Meta-Token Initialization

The primary sequence-mixing building block merges transformer self-attention and State Space Model (SSM/Mamba-2) pathways in parallel.

### 1.1 Meta-Token Prepended Input
Let $X_{\text{raw}} \in \mathbb{R}^{B \times L_{\text{raw}} \times D}$ be the input problem/context sequence of embedded token IDs. We define a learnable meta-token parameter matrix $M_{\text{meta}} \in \mathbb{R}^{1 \times N_{\text{meta}} \times D}$, where $N_{\text{meta}}$ is the number of meta-tokens (e.g., 128) and $D$ is the model dimension. 

Before passing to the hybrid block, the meta-tokens are prepended to the input sequence:
$$\tilde{X} = \text{Concat}\left(\text{Broadcast}(M_{\text{meta}}, B), X_{\text{raw}}\right) \in \mathbb{R}^{B \times (N_{\text{meta}} + L_{\text{raw}}) \times D}$$

### 1.2 Parallel Input Projection
To minimize layer complexity and memory bandwidth, a single fused projection projects $\tilde{X}$ to obtain parameters for both attention and Mamba-2 paths:
* **Attention Projections:**
  $$[Q, K, V] = \tilde{X} W_{\text{attn\_proj}} \quad \text{where } W_{\text{attn\_proj}} \in \mathbb{R}^{D \times 3D}$$
* **Mamba-2 / SSD Projections:**
  $$[X_{\text{ssm}}, G_{\text{ssm}}, H_{\text{in}}, H_{\text{out}}, \Delta] = \tilde{X} W_{\text{ssm\_proj}}$$
  where:
  * Features: $X_{\text{ssm}} \in \mathbb{R}^{B \times L \times (D \cdot E)}$ (expansion factor $E$, typically 2)
  * Gate: $G_{\text{ssm}} \in \mathbb{R}^{B \times L \times (D \cdot E)}$
  * State Inputs ($H_{\text{in}}$): $\mathbb{R}^{B \times L \times (H \cdot D_{\text{state}})}$ (heads $H$, state size $D_{\text{state}}$, typically 64)
  * State Outputs ($H_{\text{out}}$): $\mathbb{R}^{B \times L \times (H \cdot D_{\text{state}})}$
  * Step Sizes: $\Delta \in \mathbb{R}^{B \times L \times H}$
  *(Note: State matrices are renamed to $H_{\text{in}}, H_{\text{out}}$ to avoid collision with batch size $B$ and key/value projection notation).*

### 1.3 Parallel Execution Paths
1. **Attention Path:**
   $$Y_{\text{attn}} = \text{MultiHeadAttention}(Q, K, V) \in \mathbb{R}^{B \times L \times D}$$
   * **Latent Planning Phase:** Attention is fully bidirectional (non-causal).
   * **Autoregressive Generation Phase:** Attention utilizes a **Prefix-Causal Mask**. Let $i$ and $j$ be indices in the range $[1, N_{\text{meta}} + L_{\text{gen}}]$. The mask matrix $M$ is defined as:
     $$M_{ij} = \begin{cases} 
     1 & \text{if } i \le N_{\text{meta}} \text{ and } j \le N_{\text{meta}} \quad \text{(Bidirectional meta-tokens)} \\
     1 & \text{if } i > N_{\text{meta}} \text{ and } j \le N_{\text{meta}} \quad \text{(Generated tokens attend to meta-tokens)} \\
     1 & \text{if } i > N_{\text{meta}}, j > N_{\text{meta}}, \text{ and } i \ge j \quad \text{(Causal attention for generation)} \\
     0 & \text{otherwise}
     \end{cases}$$
2. **Mamba-2 / SSD Path:**
   $$Y_{\text{ssm}} = \text{Mamba2SSDScan}(X_{\text{ssm}}, G_{\text{ssm}}, H_{\text{in}}, H_{\text{out}}, \Delta) \in \mathbb{R}^{B \times L \times D}$$
   *Uses pure PyTorch tensor operations as a default fallback for CPU/GPU portability, routing to official CUDA kernels if toggled.*

### 1.4 Scale-Normalized Fusion
To resolve the magnitude gap between parallel branches, outputs are normalized, scaled, and fused:
$$\begin{aligned}
\hat{Y}_{\text{attn}} &= \text{RMSNorm}(Y_{\text{attn}}) \\
\hat{Y}_{\text{ssm}} &= \text{RMSNorm}(Y_{\text{ssm}}) \\
Y_{\text{fused}} &= \frac{(\hat{Y}_{\text{attn}} \odot \beta_1) + (\hat{Y}_{\text{ssm}} \odot \beta_2)}{2} \in \mathbb{R}^{B \times L \times D}
\end{aligned}$$
The final output is projected back to the residual stream:
$$\text{Output} = \tilde{X} + Y_{\text{fused}} W_{\text{out\_proj}} \quad \text{where } W_{\text{out\_proj}} \in \mathbb{R}^{D \times D}$$

---

## 2. Latent Planning Loop & Recurrent State Propagation

The planning architecture maintains a strict division of labor between reasoning and answer representation.

### 2.1 Dual-State Representation & Initialization
We maintain two distinct recurrent variables:
* **$z_t \in \mathbb{R}^{B \times N_{\text{meta}} \times D}$ (Latent Reasoning State):** Refines abstract constraints. Initialized as: $z_0 = M_{\text{meta}}$.
* **$y_t \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$ (Answer State):** Represents the embedding space representation of the current prediction. Initialized via $\text{InitAnswer}(X_{\text{raw}})$:
  $$y_0 = \text{Broadcast}\left(\text{Linear}_{\text{ans}}\left(\text{GlobalAveragePool}(X_{\text{raw}})\right), L_{\text{ans}}\right) \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$$
  where $L_{\text{ans}}$ is the target answer token length, and $\text{Linear}_{\text{ans}} \in \mathbb{R}^{D \times D}$ is a trainable projection.

### 2.2 The Cycle Architecture
The planning process consists of $T$ cycles. Each cycle $c \in [1, T]$ contains:
1. **$n$ Latent Reasoning Steps:** For $i = 1, \dots, n$:
   $$\tilde{X}_i = \text{Concat}(z_{i-1}, y_{c-1}, X_{\text{raw}}) \in \mathbb{R}^{B \times (N_{\text{meta}} + L_{\text{ans}} + L_{\text{raw}}) \times D}$$
   $$\tilde{H}_i = \text{PlanningBlock}(\tilde{X}_i)$$
   $$z_i = \tilde{H}_i[:, :N_{\text{meta}}, :] \in \mathbb{R}^{B \times N_{\text{meta}} \times D}$$
   *Attention is non-causal here to propagate constraints globally.*
2. **1 Answer Update Step (Cross-Attention Block):** The answer state is updated at the end of cycle $c$ using cross-attention, preventing direct access to $X_{\text{raw}}$:
   $$y_c = \text{AnswerUpdateBlock}(z_n, y_{c-1}) \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$$
   where:
   * Queries ($Q_y$): Projected from $y_{c-1} \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$
   * Keys ($K_z$) & Values ($V_z$): Projected from $z_n \in \mathbb{R}^{B \times N_{\text{meta}} \times D}$
   * Output: $y_c = \text{Softmax}\left(\frac{Q_y K_z^T}{\sqrt{D}}\right) V_z \quad$ (followed by residual connection and projection)

### 2.3 Warmup vs. Gradient Phase
* **Warmup Cycles ($c \in [1, T-1]$):** Evaluated under `torch.no_grad()` to propagate states without storing activation graphs.
* **Supervision Cycle ($c = T$):** Both $z$ and $y$ are detached at the boundary (`z.detach().requires_grad_(True)`). We run the final cycle of $n+1$ steps with gradient tracking enabled, constraining the backward pass to $n+1$ steps to prevent OOM errors.

---

## 3. Dual ACT Halting Policies & Training Losses

Adaptive Computation Time (ACT) decides when to stop planning. We support two configurable heads.

### 3.1 Sequence Representation Pooling & Detach Logic
To produce sequence-level halting decisions from token-level representations, both ACT heads apply global average pooling across the sequence dimension of the combined state:
$$s_t = \text{GlobalAveragePool}\left(\text{Concat}(z_t, y_t)\right) \in \mathbb{R}^{B \times D}$$

> [!IMPORTANT]
> **Halting Head Detach Gate:** To prevent halting head optimization (Q-loss or BCE halting loss) from corrupting the core planning representation, $s_t$ must be detached from the computational graph of the planning backbone before feeding into the halting networks:
> $$\text{planning backbone} \rightarrow s_t \rightarrow s_t\text{.detach()} \rightarrow \text{halting head}$$

---

### 3.2 Mode A: Q-Learning Head (Temporal Credit Assignment)
We define a Q-network $f_Q(s_t) \rightarrow \mathbb{R}^2$ outputting values for $a \in \{\text{continue}, \text{halt}\}$:
$$[Q(s_t, \text{continue}), Q(s_t, \text{halt})] = \text{Linear}(\text{ReLU}(\text{Linear}(s_t\text{.detach()}))))$$

* **1-Step Bootstrapped Q-Targets:**
  * **Halt Target:** $Q^*(s_t, \text{halt}) = R_t$.
  * **Continue Target ($t < m$):**
    $$Q^*(s_t, \text{continue}) = \gamma \max \left( Q_{\text{target}}(s_{t+1}, \text{continue}), Q_{\text{target}}(s_{t+1}, \text{halt}) \right)$$
    *At the absolute step boundary $M_{\text{max}}$, the continue target is undefined and the loss is masked ($\lambda_m^{\text{cont}} = 0$).*
* **Discount Factor ($\gamma$):** By default, $\gamma = 1.0$ (undiscounted) because the objective is finite-horizon reasoning rather than infinite-horizon reinforcement learning.
* **Reward Definition ($R_t$):**
  $$R_t = \mathbb{I}(\hat{y}_t == Y_{\text{target}})$$
  *While this binary reward is the default, future iterations can swap $R_t$ with token accuracy, log-likelihood, or verifier scores without framework modification.*
* **Target Network ($Q_{\text{target}}$):** Updated via Polyak EMA tracking:
  $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{online}} + (1 - \tau) \theta_{\text{target}} \quad (\text{default } \tau = 0.005)$$
* **Loss Function:**
  $$\mathcal{L}_{\text{Q}} = \frac{1}{m} \sum_{t=1}^{m} \Big[ \lambda_t^{\text{cont}} \left( Q(s_t, \text{continue}) - Q^*(s_t, \text{continue}) \right)^2 + \lambda_t^{\text{halt}} \left( Q(s_t, \text{halt}) - Q^*(s_t, \text{halt}) \right)^2 \Big]$$

---

### 3.3 Mode B: BCE Halting Classifier (Static Correctness Classifier)
Predicts halting probability $p_t = \sigma(\text{Linear}(s_t\text{.detach()}))$ with targets $p_t^* = 1.0$ if the prediction $\hat{y}_t$ is correct, else $0.0$.
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{t=1}^{m} \left( p_t^* \log(p_t) + (1 - p_t^*) \log(1 - p_t) \right)$$

---

### 3.4 Training Safeguards & Noise Regularization
* **Negative Bias Initialization:** Halting logits are initialized with bias **$-5.0$** (halting probability $\approx 0.007$) to force exploration during early training.
* **Randomized $M_{\text{min}}$:** During training, $M_{\text{min}} \sim \text{Uniform}(1, M_{\text{max}})$ is sampled per batch. The model is forced to continue until $t \ge M_{\text{min}}$.
* **Training-Time Noise Regularization:** To prevent distribution shift and calibrate the halting heads on perturbed states, we introduce training-time noise injection. With probability $p_{\text{noise\_train}} = 0.15$ during the supervision cycle, noise is injected into the latent state $z_i$:
  $$z_{i,\text{noisy}}^{\text{train}} = z_i + \epsilon_i \quad \text{where } \epsilon_i \sim \mathcal{N}(0, \eta \cdot \sigma_{\text{base}}^2 I) \text{ and } \eta \sim \text{Uniform}(0, 0.5)$$

---

### 3.5 Loss Combination
We train the model using a joint loss:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{halting}}$$
where:
* $\mathcal{L}_{\text{task}} = \text{CrossEntropy}(\text{Decode}(y_T), Y_{\text{target}})$ applied sparsely at cycle $T$.
* $\mathcal{L}_{\text{halting}} = \mathcal{L}_{\text{Q}}$ (Mode A) or $\mathcal{L}_{\text{BCE}}$ (Mode B).
* $\alpha \in \mathbb{R}^+$ is a scaling hyperparameter (default $\alpha = 1.0$). 

> [!TIP]
> **Default Implementation Path:** Build and validate the BCE halting classifier (Mode B) first, then implement the Q-learning head (Mode A) behind a configuration flag to minimize risk.

---

## 4. PTRM Test-Time Scaling (Gaussian Noise & Selection)

PTRM injects noise into the continuous latent space at inference to explore alternative solution paths.

### 4.1 Stochastic Trajectory Generation
Given $K$ parallel candidate rollouts:
* **K=1 (Deterministic Baseline):** Noise is bypassed ($\sigma = 0$) for exact reproducibility.
* **K > 1:** For each rollout $k$, we inject Gaussian noise at each step $i$ of cycle $c$:
  $$z_{i}^{\text{noisy}} = z_{i} + \epsilon_{i} \quad \text{where } \epsilon_{i} \sim \mathcal{N}(0, \sigma_i^2 I)$$
  $$\tilde{X}_i = \text{Concat}(z_i^{\text{noisy}}, y_{c-1}, X_{\text{raw}})$$
  $$z_{i+1} = \text{PlanningBlock}(\tilde{X}_i)[:, :N_{\text{meta}}, :]$$
  $$y_c = \text{AnswerUpdateBlock}(z_n^{\text{noisy}}, y_{c-1})$$

### 4.2 Noise Variance Mitigation
We support two compatible or independent mitigation settings:
1. **Noise Annealing (Default):** Noise is scaled down over the $n$ latent steps (total latent reasoning steps per cycle) of each cycle:
   $$\sigma_i = \sigma_{\text{base}} \cdot \sqrt{1 - \frac{i}{n}}$$
2. **`max_noise_step` Limit:** Noise injection is zeroed out for all steps $i > \text{max\_noise\_step}$ to allow final-phase trajectory convergence. Both settings can be combined (annealing step-wise until the limit is hit, then hard-zeroing).

---

### 4.3 Candidate Selection & Consensus Filter

> [!WARNING]
> **Halting Head Distribution Shift:** The noise injection is strictly **inference-only** (training is done on clean trajectories, regularized with minor noise). Evaluating noisy states introduces a distribution shift. The halting heads may produce uncalibrated or over-confident scores.

To mitigate this, the framework defaults to **Policy B** for all $K > 1$.

* **Policy A: Confidence-Based Selection (Argmax):**
  $$k^* = \arg\max_{k \in [1, K]} \text{Score}_k \quad \text{where } \text{Score}_k = Q(s_{m_k}, \text{halt}) \text{ or } p_{m_k}$$
* **Policy B: Consensus-Augmented Selection (Default):**
  1. Group the decoded answers of all $K$ candidates into unique prediction classes $P_1, \dots, P_G$.
  2. Identify the majority prediction class $P_{\text{maj}}$ (the most frequent answer).
  3. Select the candidate rollout $k^*$ within the majority group $P_{\text{maj}}$ that has the highest confidence score:
     $$k^* = \arg\max_{k \in P_{\text{maj}}} \text{Score}_k$$
  *If all candidates predict unique answers ($G=K$), the selection falls back to Policy A.*

---

#### 4.4 Computational Complexity Callout
Running $K$ rollouts increases inference compute cost by $K\times$. Because rollouts are independent, they are batched together along the batch dimension (increasing the effective batch size to $B \cdot K$). This is highly parallelizable on GPUs but increases peak activation memory, representing a direct trade-off between inference-time memory footprint and model accuracy.

---

## 5. Hyperparameter Settings, Training Recipe, and Evaluation

This section details the concrete settings and setup required to reproduce training and evaluation pipelines.

### 5.1 Recommended Starting Defaults

| Hyperparameter | Symbol | Recommended Default | Description |
| :--- | :--- | :--- | :--- |
| Model Dimension | $D$ | $512$ | Model embedding and latent dimension |
| Meta-Tokens Count | $N_{\text{meta}}$ | $128$ | Length of planning prefix |
| Answer Length | $L_{\text{ans}}$ | $64$ | Output sequence length (depends on task) |
| Warmup Cycles | $T - 1$ | $4$ | Cycle count evaluated without gradients |
| Latent Steps / Cycle | $n$ | $6$ | Recurrent steps per cycle |
| Max Halting Steps | $M_{\text{max}}$ | $35$ | Absolute steps cap: $T \cdot (n+1)$ |
| Noise Scale base | $\sigma_{\text{base}}$ | $0.05$ | Base standard deviation for PTRM noise |

### 5.2 Training Recipe
* **Optimizer:** AdamW with $\beta_1 = 0.9, \beta_2 = 0.98$, weight decay $0.01$.
* **Learning Rate:** Peak learning rate of $1\text{e-}4$, using a linear warmup over the first $10\%$ of steps followed by cosine decay to $1\text{e-}6$.
* **Batch Size:** $32$ samples.
* **Gradient Clipping:** Max norm $1.0$ applied to all parameters.
* **Steps:** $100,000$ training steps *(a recommended baseline for convergence monitoring, not a fixed requirement)*.

### 5.3 Evaluation Metrics
The framework logs the following metrics during training and evaluation:
1. **Task Accuracy:** Token-level and sequence-level matching accuracy compared to ground truth targets.
2. **Average Reasoning Steps:** Mean step count $\bar{m}$ taken by the ACT head before halting.
3. **Halting Precision / Recall:** Calibration metrics comparing predicted halting positions to the step where accuracy first stabilizes.

---

## 6. Algorithmic Outlines (Pseudocode)

### 6.1 Training Step Pseudocode (Mode A: Q-learning)
```python
def train_step(input_ids, target_ids, model, optimizer, target_model, M_max, n, T):
    optimizer.zero_grad()
    
    # 1. Initialization
    B = input_ids.size(0)
    X_raw = model.embed(input_ids) # [B, L_raw, D]
    z = model.M_meta.expand(B, -1, -1) # [B, N_meta, D]
    y = model.init_answer(X_raw) # [B, Lans, D]
    
    # 2. Warmup Cycles (torch.no_grad)
    with torch.no_grad():
        for c in range(1, T):
            for i in range(1, n + 1):
                X_concat = torch.cat([z, y, X_raw], dim=1)
                z = model.planning_block(X_concat)[:, :N_meta, :]
            y = model.answer_update_block(z, y)
            
    # 3. Supervision Cycle (Gradients Enabled)
    z = z.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    
    states, q_preds = [], []
    M_min = random.randint(1, M_max)
    
    # Run final n planning steps with grad
    for i in range(1, n + 1):
        X_concat = torch.cat([z, y, X_raw], dim=1)
        z = model.planning_block(X_concat)[:, :N_meta, :]
        
        # Regularization noise
        if random.random() < 0.15:
            z = z + torch.randn_like(z) * random.uniform(0, 0.25) * 0.05
            
        s_t = pool(z, y)
        q_pred = model.q_head(s_t.detach()) # Detached gate to isolate loss
        states.append(s_t)
        q_preds.append(q_pred)
        
    y_final = model.answer_update_block(z, y)
    
    # 4. Loss Computation
    loss_task = cross_entropy(y_final, target_ids)
    
    # Compute bootstrapped Q-targets
    loss_q = 0.0
    for t in range(n):
        # Halt reward
        R_t = 1.0 if evaluate_correct(y_final, target_ids) else 0.0
        
        # Halt Target
        q_halt_target = R_t
        # Continue Target
        if t < n - 1:
            with torch.no_grad():
                q_next = target_model.q_head(states[t+1])
            q_cont_target = torch.max(q_next, dim=-1)[0]
        else:
            q_cont_target = R_t # boundary condition
            
        loss_q += (q_preds[t][:, 0] - q_cont_target).pow(2).mean()
        loss_q += (q_preds[t][:, 1] - q_halt_target).pow(2).mean()
        
    total_loss = loss_task + loss_q
    total_loss.backward()
    optimizer.step()
    
    # Update target model via Polyak EMA
    update_target_network(model, target_model, tau=0.005)
```

### 6.2 Inference Rollouts Pseudocode (PTRM K-selection)
```python
def ptrm_inference(input_ids, model, K, sigma_base, n, T, max_noise_step):
    B = input_ids.size(0)
    candidates = []
    scores = []
    
    for k in range(K):
        # 1. Initialization
        X_raw = model.embed(input_ids)
        z = model.M_meta.expand(B, -1, -1)
        y = model.init_answer(X_raw)
        
        # 2. Run planning loop
        for c in range(1, T + 1):
            for i in range(1, n + 1):
                # Apply noise if K > 1
                if K > 1 and (c * n + i) <= max_noise_step:
                    # Annealing
                    sigma = sigma_base * math.sqrt(1 - i/n)
                    z = z + torch.randn_like(z) * sigma
                    
                X_concat = torch.cat([z, y, X_raw], dim=1)
                z = model.planning_block(X_concat)[:, :N_meta, :]
            y = model.answer_update_block(z, y)
            
        s_final = pool(z, y)
        score = model.q_head(s_final)[:, 1] # Halt Q-value or BCE prob
        
        candidates.append(y)
        scores.append(score)
        
    # Selection Policy (Consensus Filter)
    best_candidate = consensus_selection(candidates, scores)
    return model.decode(best_candidate)
```

---

## 7. Hardware & Dependencies

* **Software:** Requires PyTorch $\ge 1.12$ and the `mamba-ssm` package.
* **Hardware:** A single GPU with $\ge 16$ GB VRAM (e.g., NVIDIA RTX 4090 / A100) is recommended for standard training. PTRM inference is highly parallelizable; batching $K$ candidates scales memory linearly.
