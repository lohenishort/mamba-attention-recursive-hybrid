# Algorithmic Complexity and Efficiency Report

## Scope

This report covers every production function, method, generated dataclass constructor, training/evaluation entrypoint, and Rust extension function in:

- `mamba_hybrid/**/*.py`
- `scripts/*.py`, excluding `scripts/test_*.py`
- `native/src/lib.rs`

Python tests, notebooks, generated binaries, build artifacts, and module files without callables are excluded. The two Rust unit-test helpers are listed separately for complete Rust `fn` coverage. Big-O describes scaling, not absolute runtime. GPU kernel launches, synchronization, host transfers, autograd retention, and native marshalling are called out where they materially change real efficiency without changing the asymptotic class.

## Notation

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `D` | Model width, `d_model` |
| `H` | Attention/SSM heads, currently 8 |
| `S` | SSM state size per head, currently 16 |
| `E_s` | SSM expansion, currently 2 |
| `N` | Number of meta tokens, `n_meta` |
| `A` | Planner answer-state slots, `l_ans` |
| `L` | Raw input sequence length |
| `P=N+A+L` | Planner concatenated sequence length |
| `R` | Latent steps per cycle, `n_steps` |
| `C` | Executed reasoning cycles, at most `M_max` |
| `K` | PTRM stochastic rollouts |
| `V` | Output vocabulary size |
| `J` | Printer hybrid layers |
| `F` | Task experts, currently 4 |
| `Q` | Generic sequence length for one hybrid block |
| `T` | Task-printer teacher-forced or generated decoder length; this can differ from `A` |
| `M` | Number of tensor elements for an elementwise operation |
| `n` | Dataset sample count |
| `b` | Number of training/evaluation batches |
| `e` | Training epochs |
| `g` | Maze grid cells; for side length `s`, `g=s^2` |
| `p` | Maze path length |
| `v` | Graph vertices |
| `m` | Graph edges |
| `rho` | Expected number of retry attempts before a generated maze is accepted |
| `q` | Sudoku cells, normally 81 |
| `k` | Sudoku blanks |
| `d` | Sudoku symbol choices, normally 9 |
| `X` | Bytes read, written, downloaded, or serialized |
| `Pi` | Model parameter count |

Core composite costs used by script tables:

```text
HB(B,Q) = O(B*Q^2*D + B*Q*(1+E_s)*D^2 + B*Q*D*H*S + B*Q*E_s*D*S)
AU(B,N,A) = O(B*(N+A)*D^2 + B*A*N*D)
CYCLE(B,P) = O(R*HB(B,P) + AU(B,N,A))
PLAN(B,P) = O(C*CYCLE(B,P))
PRINT(B,prefix,T,V) = O(J*HB(B,prefix+T) + B*T*D*V)
GENERATE(B,prefix,T,V) = sum(t=1..T) PRINT(B,prefix,t,V)
```

The dense attention term in `HB`, `O(B*Q^2*D)`, dominates when sequence length is sufficiently large relative to model width. Projection terms `O(B*Q*D^2)` can dominate when `D >> Q`. The SSM term is linear in `Q`, but the pure-PyTorch fallback has a Python loop and many kernel launches.

## Executive Findings

| Priority | Area | Complexity | Efficiency assessment |
|---|---|---:|---|
| Critical | Uncached autoregressive printer | `O(B*J*D*T^3)` generated-length attention component | Each token reruns the full growing decoder sequence. KV/SSM caching is the largest inference optimization opportunity. |
| Critical | Recursive planning attention | `O(C*R*B*P^2*D)` | Every latent step recomputes dense attention over `[meta, answer, raw]`, although only meta outputs are retained. |
| Critical | Full-recursion planner memory | `O(C*R*B*H*P^2)` attention activations plus scan states | The current manual `matmul`/softmax implementation materializes scores and weights. Correct full BPTT is expensive by design; activation checkpointing is preferable to detaching. |
| High | Maze generator BFS | `O(g^2)` time and space per attempt | `queue.pop(0)` and copied full paths turn otherwise linear graph search quadratic. Use `deque` plus parent pointers. |
| High | Sudoku uniqueness generation | Exponential, up to `O(q^2*d^k)` per puzzle | Inherent backtracking is amplified by repeated uniqueness checks after clue removal. |
| High | Fallback SSD scan | `O(B*Q*E_s*D*S)` arithmetic | Asymptotically linear but sequential Python iteration causes poor GPU utilization. Official Triton kernels should be used on supported GPUs. |
| High | PTRM | Approximately `K` times planner work | Rollouts are batched efficiently, but vocabulary projection adds `O(K*B*A*D*V)`, argmax adds `O(K*B*A*V)`, and consensus requires host transfer. |
| Medium | Maze correctness during training | Native ready-array path `O(B*A)`; GPU transfer `O(B*(g+A))` | Rust removes Python/grid traversal, but GPU training still copies predictions and grids to CPU. A tensor-native validator would avoid synchronization. |
| Medium | Dijkstra masks | `O(B*v^2)` time and space | Appropriate for dense adjacency. Sparse edges can only reduce this toward `O(B*(v+m))` if the dense mask/logit representation is also redesigned. |
| Medium | Mixed-task MoE routing | `O(B*L*D^2)` | Homogeneous batches use one expert call; heterogeneous batches retain a per-sample Python loop and poor GPU utilization. |
| Low | Rust array consensus | Expected `O(K*B*A)` | Same asymptotic class as Python hashing, but the contiguous NumPy API is zero-copy and reduces object overhead. The sequence API still copies all tokens. |

These planner rankings describe the planner core. Complete ACT training also decodes `C` printer outputs and retains their graphs; for large vocabularies or long task outputs, `O(C*B*T*D*V)` projection work and printer activations may dominate the planner.

## Implementation Boundary Guidance

| Work type | Correct implementation domain | Reason |
|---|---|---|
| Branch-heavy CPU parsing, exact-sequence hashing, offline CPU Sudoku/path validation | Rust/PyO3 | Compiled scalar loops reduce Python-object overhead. NumPy array APIs should be preferred to nested-sequence conversion. Accelerator-resident batch validation should remain tensor-native to avoid host transfer. |
| Dense projections, attention, loss computation, MoE experts, answer updates | PyTorch tensor operations | These operations require accelerator residency and autograd. Moving them through ordinary PyO3 would force host copies or Python callbacks. |
| Sequential SSD scan on supported GPUs | Official Triton/CUDA kernel | The scan is linear but needs device-side fusion and a registered backward implementation. Plain Rust is not a replacement for a GPU kernel. |
| Recursive cycle and autoregressive control flow | Python/PyTorch, optionally compiled/cached | The loop invokes neural modules and dynamic stopping logic. Rust calling back into Python would not remove model kernel costs. |
| Data-generator graph search and backtracking | Python with better algorithms, or Rust after profiling | Fix algorithmic complexity first. Rewriting an `O(g^2)` BFS in Rust is less valuable than making it `O(g)`. |

## Core Model Package

### `mamba_hybrid/config.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| Generated `MambaHybridConfig.__init__` (`config.py:4`) | `O(1)` | `O(1)` | Fixed scalar fields. |
| `MambaHybridConfig.__post_init__` (`config.py:22`) | `O(1)` | `O(1)` | Iterates over ten fixed validation fields. |

### `mamba_hybrid/attention.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `PrefixCausalAttention.__init__` (`attention.py:17`) | `O(D^2)` | Persistent `O(D^2)` | Output projection initialization. |
| `PrefixCausalAttention.forward` (`attention.py:28`) | `O(B*L^2*D + B*L*D^2)` | `O(B*H*L^2 + B*L*D)` | Dense quadratic attention. Prefix causality changes legality, not complexity. |

### `mamba_hybrid/ssm.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `Mamba2SSDScan.__init__` (`ssm.py:20`) | `O(E_s*D^2)` | Persistent `O(E_s*D^2)` | Output projection. |
| `Mamba2SSDScan.forward` (`ssm.py:39`) | Fallback `O(B*L*E_s*D*S + B*L*E_s*D^2)` | Inference `O(B*L*E_s*D + B*E_s*D*S)`; training `O(B*L*E_s*D*S)` | Linear scan, but fallback loops over `L` in Python. The additional `O(B*L*D*H*S)` term in `HB` belongs to the parent block's `h_in/h_out` input projection, not this scan after projections are supplied. Triton preserves Big-O and improves constants. |

### `mamba_hybrid/operators.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `RMSNorm.__init__` (`operators.py:14`) | `O(D)` | Persistent `O(D)` | Efficient. |
| `RMSNorm.forward` (`operators.py:19`) | `O(M)` | `O(M)` | Elementwise; variance is computed in float32. |
| `TaskPrefixedMoeLayer.__init__` (`operators.py:31`) | `O(F*D^2)` | Persistent `O(F*D^2)` | Creates four `D -> 4D -> D` experts. |
| `TaskPrefixedMoeLayer.forward` (`operators.py:61`) | `O(B*L*D^2 + B)` | `O(B*L*D)` plus MLP activations | Homogeneous batches execute once. Mixed batches loop over samples, preserving Big-O but hurting GPU utilization. |
| `MambaAttentionHybridBlock.__init__` (`operators.py:95`) | `O(parameter count)` | Persistent `O(D^2 + F*D^2)` | Parameter initialization only. |
| `MambaAttentionHybridBlock.forward` (`operators.py:125`) | Causal `O(HB(B,L))`; non-causal adds a second linear SSM scan | `O(B*H*L^2 + B*L*D)` plus scan state | Dense attention dominates. Non-causal mode flips and scans the sequence twice. |

### `mamba_hybrid/answer_update.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `AnswerUpdateBlock.__init__` (`answer_update.py:14`) | `O(D^2)` | Persistent `O(D^2)` | Four projections and two norms. |
| `AnswerUpdateBlock.forward` (`answer_update.py:28`) | `O(B*(N+A)*D^2 + B*A*N*D)` | `O(B*H*A*N + B*(N+A)*D)` | Cross-attention is quadratic only in the two different state lengths, `A*N`. |

### `mamba_hybrid/planning.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `PlanningLoop.__init__` (`planning.py:15`) | `O(parameter count)` | One hybrid block and one or four answer blocks | MoE multiplies answer-update parameters by four. |
| `PlanningLoop.update_answer` (`planning.py:38`) | `O(AU(B,N,A)+B)` | `O(B*A*D + B*H*A*N)` | Homogeneous fast path is batched; mixed tasks loop per sample. |
| `PlanningLoop.forward` (`planning.py:70`) | `O(R*HB(B,P)+AU(B,N,A))` | Inference one-step peak; training `O(R*B*H*P^2)` plus scan activations | Full graph is retained. `torch.rand(...).item()` can synchronize once per eligible noise step. |

### `mamba_hybrid/halting.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `ACTHaltingModule.__init__` (`halting.py:7`) | `O(D^2)` | Persistent `O(D^2)` | Two small MLP heads. |
| `ACTHaltingModule.get_q_values` (`halting.py:31`) | `O(B*(N+A)*D + B*D^2)` | `O(B*D)` | Mean pooling plus MLP. Planner state is detached. |
| `ACTHaltingModule.forward` (`halting.py:52`) | `O(B*(N+A)*D + B*D^2)` | `O(B*D)` | Same as Q head, with scalar sigmoid output. |
| `polyak_update` (`halting.py:78`) | `O(Pi)` | `O(number of parameter tensors)` references | One in-place operation per parameter/buffer; many small GPU kernels are possible. |

### `mamba_hybrid/model.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `MambaAttentionHybrid.__init__` (`model.py:17`) | `O(Pi)` | Persistent `O(Pi)` | Standard model initialization. |
| `_validate_tasks` (`model.py:34`) | `O(B)` | `O(B)` | Builds a temporary task-name set. |
| `init_answer` (`model.py:43`) | Aligned `O(B*A*D^2)`; pooled `O(B*L*D+B*D^2+B*A*D)` | `O(B*L*D+B*A*D)` | Masked pooling materializes an input-sized product. |
| `decode_answer` (`model.py:59`) | `O(B*A*D*V)` | `O(B*A*V)` | Vocabulary projection; large `V` can dominate. |
| `build_memory_prefix` (`model.py:63`) | `O(B*P*D)` | `O(B*P*D+B*P)` | Concatenates and copies meta, answer, and raw states. |
| `_initialize` (`model.py:85`) | `O(B)+init_answer` | Answer storage; meta expansion is a view | Efficient initialization. |
| `forward_state_trajectory` (`model.py:99`) | `O(C*(CYCLE(B,P)+B*(N+A)*D+B*D^2))` | Returned states `O(C*B*(N+A)*D)` plus full training graph | Evaluation computes full cycles for all samples, then freezes halted samples. `bool(active.any())` synchronizes. |
| `forward_states` (`model.py:139`) | Same as trajectory | Same peak as trajectory | Wrapper discards explicit states only after they have been built. |
| `forward` (`model.py:149`) | Trajectory plus `O(B*A*D*V)` | Planning activations plus `O(B*A*V)` logits | Compatibility decoder path. |
| `forward_q` (`model.py:159`) | `O(C*(CYCLE(B,P)+B*(N+A)*D+B*D^2)+B*A*D*V)` | `O(C*B*(N+A)*D)` states and full graph | Random cycle count uses `.item()`, causing a device synchronization. |

### `mamba_hybrid/loss.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `_QHead.get_q_values` protocol (`loss.py:9`) | N/A | N/A | Type declaration only. |
| `_TargetModel.q_head` protocol (`loss.py:14`) | N/A | N/A | Type declaration only. |
| `compute_bce_joint_loss` (`loss.py:17`) | `O(B*A*V + C*B)` | `O(B*A*V + C*B)` | Cross-entropy dominates; BCE uses one Python/device operation per cycle. |
| `compute_q_joint_loss` (`loss.py:77`) | `O(B*A*V + C*(B*(N+A)*D+B*D^2))` | `O(B*A*V+B*D)` excluding retained planner states | Recomputes target Q-values for every nonterminal cycle. |

### `mamba_hybrid/inference.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `select_consensus` (`inference.py:12`) | `O(K*B*A*V + K*B*A)` expected | Host/device `O(K*B*A + K*B)` plus selected logits | `argmax` dominates. CPU transfer synchronizes GPU work; Rust improves constants and temporary memory. |
| `ptrm_inference` (`inference.py:33`) | `K=1`: model forward. `K>1`: rollouts plus `O(K*B*A*D*V + K*B*A*V)` | `O(K*B*A*D + K*B*A*V)` | Decodes every rollout before consensus; vocabulary logits can be very large. |
| `ptrm_state_rollouts` (`inference.py:72`) | `O(C*(R*HB(K*B,P)+AU(K*B,N,A)))` | `O(K*B*H*P^2 + K*B*P*D)` | Correctly batches rollouts, but always runs `M_max`; no ACT early stop. |

### `mamba_hybrid/printer.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `AutoregressivePrinter.__init__` (`printer.py:15`) | `O(Pi)` | Persistent `O(Pi)` | Embeddings, `J` hybrid blocks, norm, vocabulary head. |
| `AutoregressivePrinter.forward` (`printer.py:42`) | `O(J*HB(B,prefix+T)+B*T*D*V)` | For `Q=prefix+T`: `O(J*B*H*Q^2 + J*B*Q*E_s*D*S + J*B*Q*D + B*T*V)` in training | Prefix-causal mask is dense. Fallback SSM activations matter when sequences are shorter; no cache is maintained. |
| `AutoregressivePrinter.generate` (`printer.py:84`) | Summed over growing prefixes; attention component `O(B*J*D*(prefix^2*T + prefix*T^2 + T^3))` | Under `no_grad`, final-step peak `O(B*H*Q^2 + B*Q*E_s*D + B*E_s*D*S + B*Q*D + B*T*V)` for `Q=prefix+T` | Layers execute sequentially, so inference scan buffers do not multiply by `J`. Repeated concatenation is `O(B*T^2)`, and `bool(finished.all())` synchronizes every token. |

## Evaluation and Task Utilities

### `mamba_hybrid/evaluation.py`

Hash-table bounds are expected/amortized. Pathological collisions can degrade consensus toward quadratic candidate comparisons.

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `_py_select_consensus` (`evaluation.py:63`) | Expected `O(total candidate tokens)` | Same order due tuple copies and `Counter` | CPU fallback. |
| `_py_select_consensus_batch` (`evaluation.py:85`) | Expected `O(K*B*A)` | Peak `O(K*A)` plus result | Processes one batch item at a time. |
| `_py_validate_maze_path` (`evaluation.py:109`) | `O(grid rows + p)` | `O(1)` | Does not scan the entire grid. |
| `_py_validate_maze_moves` (`evaluation.py:132`) | `O(grid rows + examined tokens)` | `O(1)` | Early exits on EOS/error. |
| `_py_validate_maze_moves_batch` (`evaluation.py:163`) | `O(B*grid rows + total examined tokens)` | `O(B)` result | Python loop fallback. |
| `_py_validate_sudoku_board` (`evaluation.py:193`) | `O(q)`; fixed-size `O(1)` for 81 cells | `O(q)` | Materializes rows, columns, boxes, and sets. |
| `select_consensus` (`evaluation.py:219`) | Expected `O(total candidate tokens)` | `O(total tokens)` due PyO3 sequence conversion | Rust reduces interpreter overhead but list marshalling is not zero-copy. |
| `select_consensus_array` (`evaluation.py:228`) | Expected `O(K*B*A)` | Native `O(B+K)` working space; fallback `O(K*B*A)` | NumPy-to-Rust input is zero-copy only for matching contiguous arrays. |
| `validate_maze_path` (`evaluation.py:243`) | Python `O(rows+p)`; native end-to-end `O(g+p)` due marshalling | Native `O(g+p)` | Native may have worse asymptotics for wide grids because PyO3 copies every cell. |
| `validate_maze_moves_array` (`evaluation.py:256`) | Ready native arrays `O(B*A)`; conversion/fallback `O(B*(g+A))` | Native `O(B)`; fallback `O(B*(g+A))` | Major native win when arrays are already contiguous CPU `int64`. |
| `validate_sudoku_board` (`evaluation.py:281`) | Fixed-size `O(1)` | Fixed-size `O(1)` | Native lowers allocations/constants only. |

### `mamba_hybrid/tasks/common.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `shift_targets_right` (`tasks/common.py:6`) | `O(B*A)` | Returned `O(B*A)` | Vectorized and device-preserving. |

### `mamba_hybrid/tasks/gsm8k.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `normalize_answer` (`tasks/gsm8k.py:14`) | `O(string length)` | `O(string length)` | Multiple linear string passes. |
| `extract_answer` (`tasks/gsm8k.py:24`) | `O(string length)` | `O(string length)` | Partition plus normalization. |
| `encode_bytes` (`tasks/gsm8k.py:32`) | `O(characters+UTF-8 bytes)` | `O(UTF-8 bytes)` | Builds bytes and boxed token list. |
| `decode_bytes` (`tasks/gsm8k.py:38`) | `O(tokens through EOS)` | `O(tokens through EOS)` | Scalar CPU loop. |
| `encode_answer` (`tasks/gsm8k.py:53`) | `O(characters+bytes)` | `O(characters+bytes)` | Creates shifted copies. |
| `allowed_answer_tokens` (`tasks/gsm8k.py:59`) | `O(1)` | `O(1)` | Rebuilds a fixed 12-token frozenset per call. |

### `mamba_hybrid/tasks/maze.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| Generated `DecodedMazePath.__init__` (`tasks/maze.py:32`) | `O(1)` | `O(1)` excluding referenced path | Stores references without copying. |
| `path_to_moves` (`tasks/maze.py:40`) | `O(p)` | `O(p)` | Linear and appropriate. |
| `pad_moves` (`tasks/maze.py:53`) | `O(A)` | `O(A)` | Builds Python lists before tensor conversion. |
| `decode_moves` (`tasks/maze.py:60`) | Python path `O(p+rows)`; native validation can marshal `O(g+p)` | Returned path `O(p)` | Native path validator copies the full grid. |
| `maze_correct_mask` (`tasks/maze.py:99`) | Ready CPU/native `O(B*A)`; GPU/fallback `O(B*(g+A))` | Same conversion order plus `O(B)` output | CPU validation forces accelerator synchronization and transfer. |

### `mamba_hybrid/tasks/dijkstra.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| Generated `DijkstraMetrics.__init__` (`tasks/dijkstra.py:9`) | `O(1)` | `O(1)` | Four floats. |
| `encode_parent_targets` (`tasks/dijkstra.py:17`) | `O(v)` | `O(v)` | Python list then tensor copy. |
| `valid_parent_mask` (`tasks/dijkstra.py:27`) | `O(B*v^2)` | `O(B*v^2)` | Dense tensor path. Source validation uses `bool(any())`, synchronizing accelerators. |
| `constrain_parent_logits` (`tasks/dijkstra.py:52`) | `O(B*v^2)` | `O(B*v^2)` | Dense mask and output copy. |
| `optimal_parent_mask` (`tasks/dijkstra.py:64`) | `O(B*v^2)` | `O(B*v^2)` | Vectorized broadcasting. |
| `dijkstra_correct_mask` (`tasks/dijkstra.py:99`) | `O(B*v^2)` | `O(B*v^2)` | Full optimal mask dominates gather/reduction. |
| `compute_dijkstra_metrics` (`tasks/dijkstra.py:111`) | `O(B*v^2)` | `O(B*v^2)` | Four `.item()` calls synchronize device results to CPU. |
| `distances_from_sample` (`tasks/dijkstra.py:138`) | `O(v)` | `O(v)` | Python list then tensor copy. |

## Operational Scripts

### Shared helpers: `scripts/utils.py`

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `seed_everything` (`utils.py:12`) | `O(1)`; `O(device count)` for CUDA seeding | `O(1)` | Efficient. |
| `require_file` (`utils.py:22`) | `O(1)` metadata lookup | `O(1)` | I/O latency dominated. |
| `exact_match` (`utils.py:27`) | `O(B*A)` | `O(B*A)` | Vectorized comparisons. |
| `config_from_dict` (`utils.py:36`) | `O(number of input/config keys)` | Same order | Reflection and dictionary filtering. |
| `deterministic_split_indices` (`utils.py:43`) | `O(n)` | `O(n)` | Full random permutation. |
| `save_split` (`utils.py:56`) | `O(n)+IO(X)` | `O(n)` | Serialization I/O. |
| `load_validation_indices` (`utils.py:69`) | `O(validation count)` | `O(1)` new storage | Validates existing list. |

### Data generation and downloads

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `generate_maze` (`generate_data.py:7`) | Expected `O(rho*g^2)`; unbounded absolute worst case | `O(g^2)` | Quadratic copied-path BFS and `pop(0)` hotspot. |
| `generate_data.main` (`generate_data.py:63`) | `O(n*rho*g^2)+IO(n*(g+p))` | `O(n*(g+p)+g^2)` | Retains all samples while the current generator also needs quadratic scratch. |
| `generate_single_maze` (`generate_massive_maze.py:8`) | Expected `O(rho*g^2)` | `O(g^2)` | Delegates same generator. |
| `generate_massive_maze.main` (`generate_massive_maze.py:13`) | Work `O(n*rho*g^2)`; ideal wall time divided by workers | Parent `O(n*(g+p))`, workers `O(worker_count*g^2)` | Multiprocessing improves wall time, not total work. |
| `pattern` (`generate_massive_sudoku.py:7`) | `O(1)` | `O(1)` | Arithmetic index mapping. |
| `shuffle` (`generate_massive_sudoku.py:11`) | `O(n)` | `O(n)` | Random sample copy. |
| `count_solutions` (`generate_massive_sudoku.py:15`) | Worst `O(q*d^k)` | `O(k*d)` recursion/set state | Exponential backtracking. |
| `generate_sudoku_board` (`generate_massive_sudoku.py:41`) | Worst `O(q^2*d^k)` | `O(q+k*d)` | Up to `q` uniqueness checks. |
| `generate_massive_sudoku.main` (`generate_massive_sudoku.py:65`) | `O(n*q^2*d^k)+IO(n*q)` | `O(q+k*d)` streaming records | CPU-bound generation. |
| `download_file` (`download_all_datasets.py:14`) | `IO(X)` | Streaming `O(1)` buffers | Network/disk bound. |
| `build_maze_dataset` (`download_all_datasets.py:27`) | `O(n*rho*g^2)+IO(X)` | `O(n*(g+p)+g^2)` | Retained samples plus current-generator scratch. |
| Download wrapper `generate_sudoku_board` (`download_all_datasets.py:38`) | `O(q^2*d^k)` | `O(q+k*d)` | Delegation. |
| `build_sudoku_dataset` (`download_all_datasets.py:43`) | `O(n*q^2*d^k)+IO(X)` | `O(n*q)` | Retains all samples. |
| `generate_dijkstra_graph` (`download_all_datasets.py:57`) | `O(v^2+m*log m)` | `O(v^2+m)` | Dense graph creation plus heap shortest paths. |
| `build_dijkstra_dataset` (`download_all_datasets.py:97`) | `O(n*(v^2+m*log m))+IO(n*v^2)` | `O(n*v^2)` | Dense samples dominate. |
| `download_all_datasets.main` (`download_all_datasets.py:105`) | Sum of builders/downloads/clone | Peak of retained dataset sizes | Mixed CPU and I/O. |

### Maze training: `scripts/train_maze.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `MazeDataset.__init__` (`train_maze.py:31`) | `IO(n*(g+A))` | Persistent `O(n*(g+A))` | Loads full dataset. |
| `MazeDataset.__len__` (`train_maze.py:45`) | `O(1)` | `O(1)` | Efficient. |
| `MazeDataset.__getitem__` (`train_maze.py:48`) | `O(g+p+A)` | `O(g+p+A)` | Re-encodes path on every access. |
| `MazeReasoningModel.__init__` (`train_maze.py:58`) | `O(Pi)` | Persistent `O(Pi)` | Standard initialization. |
| `encode_inputs` (`train_maze.py:84`) | `O(B*g*D)` | `O(B*g*D)` | Vectorized embeddings and 2D positions. |
| `forward` (`train_maze.py:104`) | `PLAN(B,N+A+g)+PRINT(B,P,A,V)` | Corresponding model activations | Here `P=N+A+g`; final-cycle printer only. |
| `forward_cycle_logits` (`train_maze.py:116`) | `PLAN(B,N+A+g)+C*PRINT(B,P,A,V)` | Training retains `C` complete printer graphs plus `O(C*B*A*V)` logits | Required for per-cycle ACT targets; memory is not just the logits. |
| `generate` (`train_maze.py:133`) | `PLAN(B,N+A+g)+GENERATE(B,P,A,V)` | Generation peak | Uncached autoregressive path. |
| `train_maze.main` (`train_maze.py:157`) | `O(e*b_train*(cycle forward+backward+Pi)+e*b_val*forward)+IO` | Dataset + optimizer `O(Pi)` + model activations | Training compute dominates; maze correctness adds host transfers each cycle. |
| `train_maze_laptop.main` (`train_maze_laptop.py:6`) | Delegated maze training complexity | Delegated space | Constant-overhead preset wrapper. |

### Dijkstra training: `scripts/train_dijkstra.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `augment_dijkstra_example` (`train_dijkstra.py:29`) | `O(v^2)` | `O(v^2)` | Dense permutation/copy. |
| `DijkstraDataset.__init__` (`train_dijkstra.py:65`) | `O(1)` | `O(1)` beyond referenced samples | Efficient. |
| `DijkstraDataset.__len__` (`train_dijkstra.py:76`) | `O(1)` | `O(1)` | Efficient. |
| `DijkstraDataset.__getitem__` (`train_dijkstra.py:79`) | `O(v^2)` | `O(v^2)` | Dense adjacency dominates. |
| `DijkstraReasoningModel.__init__` (`train_dijkstra.py:95`) | `O(Pi)` | Persistent `O(Pi)` | Standard initialization. |
| `encode_inputs` (`train_dijkstra.py:121`) | `O(B*v^2*D)` | `O(B*(v^2+v*D))` | Dense adjacency feature projection. |
| `forward` (`train_dijkstra.py:150`) | `PLAN(B,N+A+v)+PRINT(B,P,v,V)+O(B*v^2)` | Model activations plus dense legality mask | Here `P=N+A+v`; dense graph operations. |
| `forward_cycle_logits` (`train_dijkstra.py:166`) | Planner plus `C` printers and `O(C*B*v^2)` masks | Training retains `C` complete printer graphs plus `O(C*B*v*V)` logits | ACT supervision multiplier. |
| `generate` (`train_dijkstra.py:187`) | Planner plus `GENERATE(B,P,v,V)+O(B*v^2)` | Generation peak plus legality mask | Uncached token-by-token parents. |
| `train_dijkstra.main` (`train_dijkstra.py:215`) | `O(e*b_train*(cycle forward+backward+Pi)+e*b_val*forward)+IO` | Loaded data + optimizer + activations | Compute-bound; validation accumulates graph tensors. |

### GSM8K training: `scripts/train_gsm8k.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `GSM8KDataset.__init__` (`train_gsm8k.py:35`) | `O(total text bytes)+IO(X)` | Persistent `O(total text bytes)` | Parses and tokenizes all records eagerly. |
| `GSM8KDataset.__len__` (`train_gsm8k.py:55`) | `O(1)` | `O(1)` | Efficient. |
| `GSM8KDataset.__getitem__` (`train_gsm8k.py:58`) | `O(question+answer bytes)` | Same order | Creates tensors per access. |
| `collate_gsm8k` (`train_gsm8k.py:67`) | `O(B*(max question+max answer))` | Same order | Dynamic batch padding. |
| `GSM8KReasoningModel.__init__` (`train_gsm8k.py:88`) | `O(Pi)` | Persistent `O(Pi)` | Includes maximum question-position parameters. |
| `encode_questions` (`train_gsm8k.py:118`) | `O(B*L*D)` | `O(B*L*D)` | Vectorized embedding. |
| `forward_cycle_logits` (`train_gsm8k.py:132`) | Planner plus `C` printer calls | Full recursive/printer activations | Expected for ACT targets. |
| `forward` (`train_gsm8k.py:152`) | Same as cycle-logit forward | Same as cycle-logit forward | Inefficient for final-only callers because it computes and discards intermediate cycle logits. |
| `generate` (`train_gsm8k.py:164`) | Planner plus `GENERATE(B,P,T,V)` | Generation peak | GSM8K uses `A=1` planner slot but an independent decoder limit `T`, normally 16. |
| `_sequence_correct` (`train_gsm8k.py:186`) | `O(B*T*V)` | `O(B*T)` | Vocabulary argmax dominates. |
| `train_gsm8k.main` (`train_gsm8k.py:191`) | `O(e*b_train*(cycle forward+backward+Pi)+e*b_val*cycle forward)+IO` | Dataset + optimizer + activations | Validation unnecessarily computes all cycle logits. |

### Sudoku training: `scripts/train_sudoku.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `augment_sudoku` (`train_sudoku.py:20`) | `O(q)` | `O(q)` | Efficient randomized relabeling/permutation. |
| `SudokuDataset.__init__` (`train_sudoku.py:52`) | `O(1)` | `O(1)` beyond referenced samples | Efficient. |
| `SudokuDataset.__len__` (`train_sudoku.py:56`) | `O(1)` | `O(1)` | Efficient. |
| `SudokuDataset.__getitem__` (`train_sudoku.py:59`) | `O(q)` | `O(q)` | Appropriate for fixed board size. |
| `sudoku_completion_targets` (`train_sudoku.py:75`) | `O(B*q)` | `O(B*q)` | Vectorized clue masking. |
| `SudokuReasoningModel.__init__` (`train_sudoku.py:86`) | `O(Pi)` | Persistent `O(Pi)` | Standard initialization. |
| `_apply_clues` (`train_sudoku.py:114`) | `O(B*q*V)` | `O(B*q*V)` | Copies/forces full logits. |
| `_encode_inputs` (`train_sudoku.py:119`) | `O(B*q*D)` | `O(B*q*D)` | Vectorized. |
| `_generate_from_prefix` (`train_sudoku.py:131`) | `GENERATE(B,P,q,V)+O(B*q^2)` | Under `no_grad`, printer peak plus `O(B*q*V)`; with gradients, sum of all `q` printer graphs | This method is not itself decorated with `no_grad`; callers determine whether cubic-order activation retention occurs. |
| `forward` (`train_sudoku.py:155`) | Teacher forcing: planner + one printer; otherwise generation | Corresponding activations | Behavior depends on decoder inputs. |
| `forward_cycle_logits` (`train_sudoku.py:176`) | Planner + `C` printers + clue forcing | Training retains `C` complete printer graphs plus `O(C*B*q*V)` logits | ACT supervision. |
| `train_sudoku.main` (`train_sudoku.py:198`) | `O(e*b_train*(cycle forward+backward+Pi)+e*b_val*forward)+e*IO(Pi)` | Dataset + optimizer + activations | Saves a complete checkpoint every epoch. |

### Multitask training: `scripts/train_multitask.py`

| Function | Time | Auxiliary/activation space | Assessment |
|---|---:|---:|---|
| `NativeMultiTaskModel.__init__` (`train_multitask.py:42`) | `O(Pi)` | Persistent `O(Pi)` | Planner shared, task heads separate. |
| `forward_task` (`train_multitask.py:71`) | Selected task forward plus `O(1)` dispatch | Selected task activations | Efficient dispatch. |
| `round_robin_batches` (`train_multitask.py:104`) | `O(total yielded batches)` | `O(F)` iterator state plus prefetched batches | Homogeneous batches support efficient expert routing. |
| `_sequence_correct` (`train_multitask.py:116`) | `O(B*T*V)` | `O(B*T)` | Argmax dominated for the selected task output length. |
| `native_task_loss` (`train_multitask.py:121`) | Selected task cycle-forward plus task metric | Selected task activations | Maze adds CPU validation; Dijkstra adds quadratic masks. |
| `train_multitask.main` (`train_multitask.py:207`) | Sum over tasks of `O(e*b_task*(forward+backward+Pi))+IO` | All datasets + unified optimizer + max task activations | AdamW scans the unified model after every task batch. |

### Diagnosis and evaluation scripts

| Function | Time | Auxiliary space | Assessment |
|---|---:|---:|---|
| `diagnose_sudoku.main` (`diagnose_sudoku.py:10`) | `IO(X)+sample_count*GENERATE(1,P,q,V)` | Model + data + generation peak | Serial diagnostic generation. |
| `print_sudoku_board` (`evaluate_sudoku.py:9`) | `O(q)` | `O(sqrt(q))` | Stdout-I/O dominated. |
| `evaluate_sudoku.main` (`evaluate_sudoku.py:23`) | `IO(X)+(validation count+1)*GENERATE(1,P,q,V)` | Full dataset + model + generation peak | No batching; one model call per board. |
| `evaluate_maze.main` (`evaluate_maze.py:11`) | `IO(X)+b_val*GENERATE(B,P,A,V)+O(n_val*(g+A))` | Full data + model + batch peak | Dataset effectively loaded twice. |
| `evaluate_dijkstra.main` (`evaluate_dijkstra.py:11`) | `IO(X)+b_val*GENERATE(B,P,v,V)+O(n_val*v^2)` | Accumulates `O(n_val*v^2)` graph data | Dense accumulation can dominate memory. |
| `evaluate_gsm8k.main` (`evaluate_gsm8k.py:15`) | `IO(X)+b_test*GENERATE(B,P,T,V)+O(n_test*T)` | Dataset + model + batch peak | Batched generation with decoder length independent of planner `A`. |
| `evaluate_multitask.main` (`evaluate_multitask.py:18`) | `IO(X)` plus one generation per task | Full datasets plus largest generation graph; Sudoku is called without an outer `no_grad` context | Loads complete datasets to evaluate one sample from each task and may unnecessarily retain Sudoku generation graphs. |

## Rust Extension

### `native/src/lib.rs`

Rust `HashMap` bounds are expected. Hashing a sequence is linear in sequence length even when table lookup is expected constant time. Native functions currently hold the Python GIL and run serially.

| Function | Time | Auxiliary space | Boundary behavior |
|---|---:|---:|---|
| `consensus_index` (`lib.rs:9`) | Expected `O(total candidate tokens)`; collision worst up to `O(candidate_count^2*max length)` | `O(unique candidates)` | Borrows Rust vectors; no token copies inside helper. |
| `select_consensus` (`lib.rs:38`) | Expected `O(total candidate tokens)` | PyO3 copies `O(total tokens)` plus hash map | Python sequences are not zero-copy. |
| `select_consensus_array` (`lib.rs:43`) | Expected `O(K*B*A)` | `O(max unique rollouts + B)` | Borrows C-contiguous NumPy buffers zero-copy. |
| `validate_maze_path` (`lib.rs:100`) | Rust body `O(rows+p)`; Python end-to-end `O(g+p)` | PyO3 copies `O(g+p)` | Entire nested grid is marshalled. |
| `validate_maze_moves_array` (`lib.rs:131`) | `O(B*A)` worst case | `O(B)` output | Zero-copy contiguous arrays; only visited cells are read. |
| `validate_maze_moves_flat` (`lib.rs:190`) | `O(A)` with early exit | `O(1)` | Private scalar loop; requires EOS at goal. |
| `validate_sudoku_board` (`lib.rs:236`) | Fixed-size `O(1)` | Fixed-size `O(1)` | Sequence copies are bounded to 81 cells. |
| `_native` (`lib.rs:282`) | `O(1)` | `O(1)` | Registers five fixed functions. |
| `consensus_prefers_majority_then_confidence` test (`lib.rs:296`) | Fixed-size `O(1)` | `O(1)` | Test-only, included for complete Rust function coverage. |
| `maze_moves_require_legal_goal_reaching_eos` test (`lib.rs:302`) | Fixed-size `O(1)` | `O(1)` | Test-only, included for complete Rust function coverage. |

## Scaling Examples

### Planner

For `B=16`, `N=128`, `A=64`, raw length `L=900`, `P=1092`, `R=6`, and `C=5`, dense planner attention scales with:

```text
C * R * B * P^2 * D
```

The `P^2` term means increasing raw sequence length eventually dominates the linear SSM benefit. Doubling `P` approximately quadruples attention compute and attention-map memory once attention is dominant; at smaller `P` or much larger `D`, projection work can remain dominant. The current implementation explicitly materializes `scores` and `attn_weights`, so the quadratic memory bound applies. A future fused memory-efficient attention backend could lower saved-score memory without changing quadratic arithmetic.

### Printer generation

Without caching, generating `T` tokens reruns the printer for lengths `1..T`. The sum of squared sequence lengths introduces a cubic generated-length component:

```text
sum(t=1..T) (prefix+t)^2
= O(prefix^2*T + prefix*T^2 + T^3)
```

The `T^3` term is the asymptotic generated-length component for fixed prefix and no early EOS. When `prefix >> T`, the observed attention cost can instead be dominated by `prefix^2*T`; repeated full-sequence vocabulary projection also contributes a potentially dominant quadratic `T` term. Caching would reduce repeated historical attention/projection work substantially, although prefix attention and output-head costs would remain.

### Full-recursion training

Full BPTT retains activations for all `C*R` planner blocks:

```text
attention activation memory = O(C*R*B*H*P^2)
scan activation memory      = O(C*R*B*P*E_s*D*S)
```

Activation checkpointing can trade additional recomputation for lower retained memory while preserving correct gradients.

## Workload-Specific Optimization Order

Apply these orders after profiling the target workload; they are not a universal ranking.

### Inference latency

1. Add incremental state/KV caching to `AutoregressivePrinter.generate` and task-specific generation paths.
2. Remove per-token host synchronizations such as `bool(finished.all())` where practical.
3. Avoid loading complete datasets for one-sample smoke tests and batch Sudoku evaluation.

### Training throughput

1. Enable official Triton SSD kernels on compatible GPUs and remove avoidable mask/noise synchronization.
2. Avoid cycle-logit decoding in final-only GSM8K forward/validation calls.
3. Keep maze correctness on-device or implement a tensor-native validator to remove GPU-to-CPU grid transfer.
4. Group heterogeneous MoE samples by task instead of invoking experts one sample at a time.

### Training memory

1. Add activation checkpointing around recursive planning and cycle-specific printers rather than detaching states.
2. Reduce planner attention scope or evaluate sparse/chunked attention with task-quality regression tests.
3. Reduce retained cycle logits/graphs when the halting objective does not require all decoded cycles.

### Data preparation

1. Rewrite maze generation with `collections.deque` and parent pointers to reduce BFS from `O(g^2)` to `O(g)` time and space per attempt.
2. Profile and parallelize Sudoku uniqueness generation, while recognizing its exponential worst case.
3. Stream generated datasets instead of retaining all samples before serialization.

### Large graph tasks

1. Adopt sparse adjacency, mask, and output representations together. Changing only adjacency storage cannot remove the current dense `O(B*v^2)` output lower bound.

## Complexity Caveats

- Big-O suppresses constants. Rust and Triton can be much faster while retaining the same asymptotic class.
- GPU operations are asynchronous until `.item()`, `bool(tensor)`, `.cpu()`, or similar host-visible operations synchronize them.
- PyTorch autograd can retain tensors that appear temporary in forward code. Training memory therefore differs from inference auxiliary memory.
- `expand` is usually a view, while concatenation, arithmetic, `reshape` after incompatible strides, and many flips materialize storage.
- Hash-map complexity is expected/amortized; adversarial collisions can worsen it.
- Sudoku is fixed at 81 cells operationally, making many board checks formally constant, but generalized solver behavior remains exponential.
- Network and disk operations are represented as `IO(X)` because latency and bandwidth, rather than CPU operations, dominate them.
