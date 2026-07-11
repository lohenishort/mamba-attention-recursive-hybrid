# Engineering & Evaluation Spec: Mamba-Attention Recursive Reasoning Hybrid

This document outlines the architecture, training procedure, and evaluation pipeline for the Mamba-Attention Recursive Reasoning Hybrid framework. The core objective is to design a model that separates latent constraint propagation ("thinking slow") from autoregressive sequence printing ("thinking fast"), stabilized by Adaptive Computation Time (ACT) and stochastically scaled at test-time using the Probabilistic Tiny Recursive Model (PTRM) approach.

---

## 1. Core Hybrid Operator & Meta-Token Initialization

The primary sequence-mixing building block merges transformer self-attention and State Space Model (SSM/Mamba-2) pathways in parallel.

### 1.1 Meta-Token Prepended Input
Let $X_{\text{raw}} \in \mathbb{R}^{B \times L_{\text{raw}} \times D}$ be the input problem/context sequence. We define a learnable meta-token parameter matrix $M_{\text{meta}} \in \mathbb{R}^{1 \times N_{\text{meta}} \times D}$, where $N_{\text{meta}}$ is the number of meta-tokens (e.g., 128) and $D$ is the model dimension. 

Before passing to the hybrid block, the meta-tokens are prepended to the input sequence:
$$\tilde{X} = \text{Concat}\left(\text{Broadcast}(M_{\text{meta}}, B), X_{\text{raw}}\right) \in \mathbb{R}^{B \times (N_{\text{meta}} + L_{\text{raw}}) \times D}$$

### 1.2 Parallel Input Projection
To minimize layer complexity and memory bandwidth, a single fused projection projects $\tilde{X}$ to obtain parameters for both attention and Mamba-2 paths:
* **Attention Projections:**
  $$[Q, K, V] = \tilde{X} W_{\text{attn\_proj}} \quad \text{where } W_{\text{attn\_proj}} \in \mathbb{R}^{D \times 3D}$$
* **Mamba-2 / SSD Projections:**
  $$[X_{\text{ssm}}, G_{\text{ssm}}, B, C, \Delta] = \tilde{X} W_{\text{ssm\_proj}}$$
  where:
  * Features: $X_{\text{ssm}} \in \mathbb{R}^{B \times L \times (D \cdot E)}$ (expansion factor $E$, typically 2)
  * Gate: $G_{\text{ssm}} \in \mathbb{R}^{B \times L \times (D \cdot E)}$
  * State Inputs: $B \in \mathbb{R}^{B \times L \times (H \cdot D_{\text{state}})}$ (heads $H$, state size $D_{\text{state}}$, typically 64)
  * State Outputs: $C \in \mathbb{R}^{B \times L \times (H \cdot D_{\text{state}})}$
  * Step Sizes: $\Delta \in \mathbb{R}^{B \times L \times H}$

### 1.3 Parallel Execution Paths
1. **Attention Path:**
   $$Y_{\text{attn}} = \text{MultiHeadAttention}(Q, K, V) \in \mathbb{R}^{B \times L \times D}$$
   *Attention is non-causal (bidirectional) during the latent planning phase, and causal during the autoregressive generation phase.*
2. **Mamba-2 / SSD Path:**
   $$Y_{\text{ssm}} = \text{Mamba2SSDScan}(X_{\text{ssm}}, G_{\text{ssm}}, B, C, \Delta) \in \mathbb{R}^{B \times L \times D}$$
   *Uses pure PyTorch tensor operations as a default fallback for CPU/GPU portability, routing to official CUDA kernels if toggled.*

### 1.4 Scale-Normalized Fusion
To resolve the magnitude gap between parallel branches, outputs are normalized and scaled using learnable parameters $\beta_1, \beta_2 \in \mathbb{R}^D$:
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

### 2.1 Dual-State Representation
We maintain two distinct recurrent variables:
* **$z_t \in \mathbb{R}^{B \times N_{\text{meta}} \times D}$ (Latent Reasoning State):** Refines abstract constraints. Initialized as: $z_0 = M_{\text{meta}}$.
* **$y_t \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$ (Answer State):** Represents the embedding space representation of the current prediction. Initialized as: $y_0 = \text{InitAnswer}(X_{\text{raw}})$.

### 2.2 The Cycle Architecture
The planning process consists of $T$ cycles. Each cycle $c \in [1, T]$ contains:
1. **$n$ Latent Reasoning Steps:** For $i = 1, \dots, n$:
   $$\tilde{X}_i = \text{Concat}(z_{i-1}, y_{c-1}, X_{\text{raw}}) \in \mathbb{R}^{B \times (N_{\text{meta}} + L_{\text{ans}} + L_{\text{raw}}) \times D}$$
   $$\tilde{H}_i = \text{PlanningBlock}(\tilde{X}_i)$$
   $$z_i = \tilde{H}_i[:, :N_{\text{meta}}, :] \in \mathbb{R}^{B \times N_{\text{meta}} \times D}$$
   *Attention is non-causal here to propagate constraints globally.*
2. **1 Answer Update Step:** The answer state is updated at the end of cycle $c$ **without** direct access to $X_{\text{raw}}$:
   $$y_c = \text{AnswerUpdateBlock}(z_n, y_{c-1}) \in \mathbb{R}^{B \times L_{\text{ans}} \times D}$$

### 2.3 Warmup vs. Gradient Phase
* **Warmup Cycles ($c \in [1, T-1]$):** Evaluated under `torch.no_grad()` to propagate states without storing activation graphs.
* **Supervision Cycle ($c = T$):** Both $z$ and $y$ are detached at the boundary (`z.detach().requires_grad_(True)`). We run the final cycle of $n+1$ steps with gradient tracking enabled, constraining the backward pass to $n+1$ steps to prevent OOM errors.

---

## 3. Dual ACT Halting Policies & Training Losses

Adaptive Computation Time (ACT) decides when to stop planning. We support two configurable heads.

### 3.1 Mode A: Q-Learning Head (Temporal Credit Assignment)
We define a Q-network $f_Q(z_t, y_t) \rightarrow \mathbb{R}^2$ outputting values for $a \in \{\text{continue}, \text{halt}\}$:
$$[Q(s_t, \text{continue}), Q(s_t, \text{halt})] = \text{Linear}(\text{ReLU}(\text{Linear}(\text{Concat}(z_t, y_t))))$$

* **1-Step Bootstrapped Q-Targets:**
  * **Halt Target:** $Q^*(s_t, \text{halt}) = R_t = \mathbb{I}(\hat{y}_t == Y_{\text{target}})$
  * **Continue Target ($t < m$):**
    $$Q^*(s_t, \text{continue}) = \max \left( Q_{\text{target}}(s_{t+1}, \text{continue}), Q_{\text{target}}(s_{t+1}, \text{halt}) \right)$$
    *At the absolute step boundary $M_{\text{max}}$, the continue target is undefined and the loss is masked ($\lambda_m^{\text{cont}} = 0$).*
* **Target Network ($Q_{\text{target}}$):** Updated via Polyak EMA tracking:
  $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{online}} + (1 - \tau) \theta_{\text{target}} \quad (\text{default } \tau = 0.005)$$
* **Loss Function:**
  $$\mathcal{L}_{\text{Q}} = \frac{1}{m} \sum_{t=1}^{m} \Big[ \lambda_t^{\text{cont}} \left( Q(s_t, \text{continue}) - Q^*(s_t, \text{continue}) \right)^2 + \lambda_t^{\text{halt}} \left( Q(s_t, \text{halt}) - Q^*(s_t, \text{halt}) \right)^2 \Big]$$

### 3.2 Mode B: BCE Halting Classifier (Static Correctness Classifier)
Predicts halting probability $p_t = \sigma(\text{Linear}(\text{Concat}(z_t, y_t)))$ with targets $p_t^* = 1.0$ if the prediction $\hat{y}_t$ is correct, else $0.0$.
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{t=1}^{m} \left( p_t^* \log(p_t) + (1 - p_t^*) \log(1 - p_t) \right)$$

### 3.3 Training Safeguards
* **Negative Bias Initialization:** Halting logits are initialized with bias **$-5.0$** (halting probability $\approx 0.007$) to force exploration during early training.
* **Randomized $M_{\text{min}}$:** During training, $M_{\text{min}} \sim \text{Uniform}(1, M_{\text{max}})$ is sampled per batch. The model is forced to continue until $t \ge M_{\text{min}}$.

### 3.4 Sparse Supervision Loss
Task cross-entropy loss is applied sparsely, only at the final step of cycle $T$:
$$\mathcal{L}_{\text{task}} = \text{CrossEntropy}(\text{Decode}(y_T), Y_{\text{target}})$$
Intermediate states $y_c$ ($c < T$) and $z$ receive no direct task loss.

---

## 4. PTRM Test-Time Scaling (Gaussian Noise & Selection)

PTRM injects noise into the continuous latent space at inference to explore alternative solution paths.

### 4.1 Stochastic Trajectory Generation
Given $K$ parallel candidate rollouts:
* **K=1 (Deterministic Baseline):** Noise is bypassed ($\sigma = 0$) for exact reproducibility.
* **K > 1:** For each rollout $k$, we inject Gaussian noise at each step $i$:
  $$z_{i}^{\text{noisy}} = z_{i} + \epsilon_{i} \quad \text{where } \epsilon_{i} \sim \mathcal{N}(0, \sigma_i^2 I)$$
  $$\tilde{X}_i = \text{Concat}(z_i^{\text{noisy}}, y_{c-1}, X_{\text{raw}})$$
  $$z_{i+1} = \text{PlanningBlock}(\tilde{X}_i)[:, :N_{\text{meta}}, :]$$
  $$y_c = \text{AnswerUpdateBlock}(z_n^{\text{noisy}}, y_{c-1})$$

### 4.2 Noise Variance Mitigation
We support two compatible or independent mitigation settings:
1. **Noise Annealing (Default):** Noise is scaled down over the $n$ latent steps of each cycle:
   $$\sigma_i = \sigma_{\text{base}} \cdot \sqrt{1 - \frac{i}{n}}$$
2. **`max_noise_step` Limit:** Noise injection is zeroed out for all steps $i > \text{max\_noise\_step}$ to allow final-phase trajectory convergence.

### 4.3 Candidate Selection & Consensus Filter

> [!WARNING]
> **Halting Head Distribution Shift:** The noise injection is strictly **inference-only** (training is done on clean trajectories). Evaluating noisy states introduces a distribution shift. The halting heads may produce uncalibrated or over-confident scores.

To mitigate this, the framework defaults to **Policy B** for all $K > 1$.

* **Policy A: Confidence-Based Selection (Argmax):**
  $$k^* = \arg\max_{k \in [1, K]} \text{Score}_k \quad \text{where } \text{Score}_k = Q(s_{m_k}, \text{halt}) \text{ or } p_{m_k}$$
* **Policy B: Consensus-Augmented Selection (Default):**
  1. Group the decoded answers of all $K$ candidates into unique prediction classes $P_1, \dots, P_G$.
  2. Identify the majority prediction class $P_{\text{maj}}$ (the most frequent answer).
  3. Select the candidate rollout $k^*$ within the majority group $P_{\text{maj}}$ that has the highest confidence score:
     $$k^* = \arg\max_{k \in P_{\text{maj}}} \text{Score}_k$$
  *If all candidates predict unique answers ($G=K$), the selection falls back to Policy A.*
