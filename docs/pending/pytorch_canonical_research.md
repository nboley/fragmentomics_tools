# Canonical PyTorch & Lightning Code: Agent Ruleset Research

*Compiled 2026-09-24. Environment: torch 2.5.1, CUDA 12.4, PyTorch Lightning 2.x. Context: research codebase (training, simulation, analysis).*

---

## 1. Prioritised Ruleset

Rules ordered by impact. Each has a rationale, source, and where possible a bad/good contrast.

### Tier 1: Silent-Correctness Rules (get these wrong and results are quietly wrong)

#### R1. Always gate eval/inference with `model.eval()` and `model.train()`

**Rationale:** Forgetting `model.eval()` before validation means Dropout stays active and BatchNorm uses batch statistics instead of running statistics. Results look plausible but are wrong. Forgetting `model.train()` after validation disables Dropout and freezes BN during subsequent training.

```python
# BAD
for epoch in range(epochs):
    train_one_epoch(model, ...)
    val_loss = validate(model, ...)  # dropout still active, BN uses batch stats

# GOOD
for epoch in range(epochs):
    model.train()
    train_one_epoch(model, ...)
    model.eval()
    with torch.inference_mode():
        val_loss = validate(model, ...)
```

**Source:** [PyTorch official tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html); [Common training mode mistakes (Medium)](https://medium.com/@niruthiha2000/the-silent-killer-of-your-pytorch-models-why-90-of-ml-engineers-get-training-modes-wrong-d04becfacd8b)

---

#### R2. Use `torch.inference_mode()` for inference, not `torch.no_grad()`

**Rationale:** `inference_mode` disables both gradient computation AND autograd view-tracking/version-counter overhead. Faster than `no_grad` with no downside when you truly don't need the output tensors to participate in autograd later. Available since PyTorch 1.9.

```python
# ACCEPTABLE (but slower)
with torch.no_grad():
    preds = model(x)

# PREFERRED
with torch.inference_mode():
    preds = model(x)
```

**Note:** `inference_mode` does NOT call `model.eval()` -- you still need both.

**Source:** [PyTorch docs](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html); [PyTorch Forums discussion](https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099)

---

#### R3. Never use `.item()`, `.cpu()`, `.numpy()` inside training loops (GPU sync points)

**Rationale:** Each of these forces a CUDA synchronization, stalling the GPU pipeline. The GPU sits idle waiting for the CPU to catch up. Use them only for periodic logging, never per-batch.

```python
# BAD -- sync every step
for batch in loader:
    loss = train_step(batch)
    running_loss += loss.item()  # forces GPU sync

# GOOD -- log periodically, or let Lightning handle it
for i, batch in enumerate(loader):
    loss = train_step(batch)
    running_loss += loss.detach()  # stays on GPU
    if i % log_interval == 0:
        print(f"loss: {running_loss.item() / log_interval:.4f}")
        running_loss = 0.0
```

**Source:** [PyTorch Performance Tuning Guide (official)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html); [Lightning Speed Guide](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)

---

#### R4. Avoid in-place operations on tensors that require grad

**Rationale:** PyTorch discourages in-place ops in autograd. They can overwrite values needed for backward and cause version-counter errors. The memory savings are almost never significant enough to justify the risk.

```python
# BAD -- may cause "modified by an inplace operation" error
x += bias           # in-place
x.add_(other)       # in-place

# GOOD
x = x + bias        # out-of-place, autograd-safe
x = x.add(other)
```

**Exception:** In-place ops are safe on leaf tensors that don't require grad (e.g., updating parameters manually with `torch.no_grad()`), and on tensors in `inference_mode`.

**Source:** [PyTorch Autograd Mechanics (official)](https://docs.pytorch.org/docs/main/notes/autograd.html); [PyTorch inplace advice (lernapparat.de)](https://lernapparat.de/pytorch-inplace)

---

#### R5. `.detach()` before storing tensors from the computation graph

**Rationale:** Storing loss tensors, hidden states, or intermediate activations without `.detach()` retains the entire computation graph in memory, causing OOM. This is the #1 cause of "memory keeps growing" in training loops.

```python
# BAD -- retains entire graph
losses.append(loss)               # graph lives until losses is GC'd
hidden = model.get_hidden(x)      # graph attached

# GOOD
losses.append(loss.detach())      # or loss.item() if you want a float
hidden = model.get_hidden(x).detach()
```

**Source:** [PyTorch memory leak debugging](https://adhdecode.com/articles/pytorch/pytorch-memory-leak-debugging-gpu/); [Memory leak detection (Neural Base)](https://theneuralbase.com/pytorch/learn/intermediate/memory-leak-detection/)

---

#### R6. `retain_graph=True` is almost always wrong

**Rationale:** Prevents the computation graph from being freed after backward. Needed only for multi-backward calls (meta-learning, some RL). If you find yourself needing it, the architecture likely needs rethinking.

**Source:** [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html); [Mindful Chase troubleshooting](https://www.mindfulchase.com/explore/troubleshooting-tips/fixing-gpu-memory-leaks,-gradient-accumulation-issues,-and-training-performance-bottlenecks-in-pytorch-lightning.html)

---

#### R7. Create tensors directly on the target device

```python
# BAD -- allocates on CPU then copies
t = torch.zeros(shape).cuda()
t = torch.rand(shape).to(device)

# GOOD -- allocates directly on device
t = torch.zeros(shape, device=device)
t = torch.rand(shape, device=device)
```

**Rationale:** Avoids a pointless CPU allocation and H2D copy. Trivial to fix, easy to grep for.

**Source:** [PyTorch Performance Tuning Guide (official)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

### Tier 2: Performance Rules (correct results but slower without these)

#### R8. Use `optimizer.zero_grad(set_to_none=True)`

```python
# DEFAULT (slower)
optimizer.zero_grad()  # fills grads with zeros

# PREFERRED
optimizer.zero_grad(set_to_none=True)  # sets grads to None, avoids memset
```

**Rationale:** Setting gradients to `None` instead of zero avoids a memset and can be faster. This is the default in Lightning. Minor caveat: some optimizers may not handle `None` grads, but all standard ones do.

**Source:** [PyTorch Performance Tuning Guide (official)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

#### R9. Set `torch.backends.cudnn.benchmark = True` for fixed-size inputs

**Rationale:** Lets cuDNN benchmark multiple convolution algorithms on the first forward pass and cache the fastest. Significant speedup for conv-heavy models. But: produces non-deterministic results across runs, and wastes time if input sizes vary (re-benchmarks on every new size).

```python
# GOOD for fixed-size inputs (most research training)
torch.backends.cudnn.benchmark = True

# REQUIRED for reproducible debugging (much slower)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # older API
torch.use_deterministic_algorithms(True)    # comprehensive, torch >= 1.8
```

**Source:** [PyTorch Performance Tuning Guide (official)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html); [stas00/ml-engineering reproducibility guide](https://github.com/stas00/ml-engineering/blob/master/training/reproducibility/README.md)

---

#### R10. DataLoader: `num_workers > 0`, `pin_memory=True`, `persistent_workers=True`

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,            # tune upward until throughput plateaus
    pin_memory=True,          # enables async H2D transfer
    persistent_workers=True,  # avoids worker restart overhead between epochs
)
```

**Rationale:** `num_workers=0` means data loading blocks the training loop. `pin_memory=True` enables faster GPU transfers via page-locked memory. `persistent_workers=True` avoids the overhead of spawning new worker processes each epoch.

**Caveats:**
- Too many workers wastes CPU RAM and can exhaust `/dev/shm` in containers.
- Set `worker_init_fn` for reproducible per-worker seeding.
- Use `prefetch_factor` (default 2) to control memory vs throughput tradeoff.

**Source:** [PyTorch DataLoader docs](https://docs.pytorch.org/docs/2.14/data.html); [Lightning Speed Guide](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)

---

#### R11. Remove conv bias when followed by BatchNorm

```python
# BAD -- bias is absorbed by BN and wasted
nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),  # default
nn.BatchNorm2d(out_ch),

# GOOD
nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
nn.BatchNorm2d(out_ch),
```

**Rationale:** BN's learned bias (`beta`) subsumes the conv bias. The extra parameter wastes memory and is mathematically redundant.

**Source:** [PyTorch Performance Tuning Guide (official)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

#### R12. Mixed precision: use `torch.amp.autocast` (not the deprecated `torch.cuda.amp`)

```python
# DEPRECATED API (still works but will be removed)
with torch.cuda.amp.autocast():
    output = model(input)

# CURRENT API (torch >= 2.0)
with torch.amp.autocast("cuda"):
    output = model(input)
```

**Rationale:** The device-generic `torch.amp.autocast` API replaced the CUDA-specific one. Using it avoids deprecation warnings and works across backends.

**Version note (important for our env):** torch 2.5.1 supports both APIs but the old one emits deprecation warnings. `GradScaler` has similarly moved from `torch.cuda.amp.GradScaler` to `torch.amp.GradScaler("cuda")`.

**Source:** [PyTorch AMP docs (official)](https://docs.pytorch.org/docs/main/amp.html); [AMP recipe tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

---

#### R13. `torch.compile`: useful but opt-in; maintain an eager baseline

**Rationale:** `torch.compile` can yield significant speedups through kernel fusion, but: compilation time is non-trivial, it can introduce numerical differences, graph breaks reduce benefits, and dynamic shapes cause recompilation. For research code, the overhead often exceeds the benefit unless training runs are long.

**Best practice:** Keep code that runs correctly in eager mode. Apply `torch.compile` selectively to compute-heavy modules, not the entire model. Pin input shapes. Maintain an eager baseline for debugging.

```python
# SELECTIVE compilation (preferred for research)
model.encoder = torch.compile(model.encoder)

# FULL model compilation (for long production runs)
model = torch.compile(model, mode="default")
```

**Source:** [Edward Yang's blog (PyTorch core dev)](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/); [State of torch.compile Aug 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/); [PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

---

### Tier 3: Lightning-Specific Rules

#### R14. Separate model from system in LightningModule

**Rationale (official Lightning style guide):** A *model* is a component (ResNet, RNN). A *system* is a LightningModule that orchestrates models, losses, and training logic. Keep nn.Module models as standalone, reusable components passed into the LightningModule.

```python
# BAD -- model architecture entangled with training logic
class MySystem(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(...)  # architecture mixed in
        self.conv2 = nn.Conv2d(...)
    def forward(self, x): ...
    def training_step(self, batch, batch_idx): ...

# GOOD -- clean separation
class MyModel(nn.Module):
    def __init__(self): ...
    def forward(self, x): ...

class MySystem(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def training_step(self, batch, batch_idx):
        return self.model(batch)
```

**Source:** [Lightning Style Guide (official)](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)

---

#### R15. `configure_optimizers` return contract

The method can return 6 different types. The safest and most explicit pattern:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",   # or "step"
            "frequency": 1,
            "monitor": "val_loss", # required for ReduceLROnPlateau
        },
    }
```

**Traps:**
- If you return a bare optimizer, Lightning defaults scheduler `interval` to `"epoch"` -- if you meant `"step"`, your LR schedule is silently wrong.
- `ReduceLROnPlateau` requires `"monitor"` key. Omitting it raises a cryptic error.
- In manual optimization, `interval`/`frequency` keys are ignored even if provided.

**Source:** [Lightning Optimization docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html); [Lightning Issue #20937](https://github.com/Lightning-AI/pytorch-lightning/issues/20937)

---

#### R16. Know what Lightning already does -- don't hand-roll it

Lightning automatically handles these. Reimplementing them is wasted effort and a source of bugs:

| Feature | Lightning provides | Don't hand-roll |
|---|---|---|
| `optimizer.zero_grad()` | Automatic (with `set_to_none=True` default) | Calling it in `training_step` |
| `loss.backward()` | Automatic | Calling it in `training_step` |
| `optimizer.step()` | Automatic | Calling it in `training_step` |
| Gradient clipping | `Trainer(gradient_clip_val=...)` | Manual `clip_grad_norm_` |
| Mixed precision | `Trainer(precision="16-mixed")` | Manual `autocast`/`GradScaler` |
| Multi-GPU | `Trainer(devices=N, strategy="ddp")` | Manual DDP setup |
| Checkpointing | `ModelCheckpoint` callback | Manual `torch.save` |
| Early stopping | `EarlyStopping` callback | Manual epoch-loop logic |
| Logging | `self.log("name", value)` | Manual TensorBoard writer |
| LR scheduling | `configure_optimizers` return dict | Manual `scheduler.step()` |
| Gradient accumulation | `Trainer(accumulate_grad_batches=N)` | Manual accumulation logic |

**Source:** [Lightning docs](https://lightning.ai/docs/pytorch/stable/); [PyTorch Lightning for Dummies (AssemblyAI)](https://www.assemblyai.com/blog/pytorch-lightning-for-dummies)

---

#### R17. Separation of concerns: LightningModule vs DataModule vs Callbacks

| Component | Belongs here | Does NOT belong here |
|---|---|---|
| **LightningModule** | Model architecture, forward pass, training/validation/test steps, loss computation, `configure_optimizers` | Data loading, data transforms, logging infrastructure |
| **LightningDataModule** | Dataset construction, transforms, DataLoader configuration, data splitting | Model logic, training logic |
| **Callbacks** | Logging, visualization, early stopping, LR monitoring, checkpointing, non-essential side effects | Core model logic, essential training math |
| **Trainer** | Hardware config, precision, distributed strategy, epoch management | Anything already in the module/datamodule/callbacks |

**Source:** [Lightning Style Guide](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html); [DataModules docs](https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/datamodules.html)

---

#### R18. Callbacks should be independent and order-agnostic

**Rationale:** Lightning executes callbacks in registration order, but your callbacks should not depend on execution order. A callback that requires another callback to have run first is a design smell -- factor the shared state into the LightningModule or a shared object.

**Source:** [Lightning Callback docs](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/callback.html); [Callback order (Neural Base)](https://theneuralbase.com/pytorch-lightning/learn/beginner/callback-order-of-execution/)

---

### Tier 4: Checkpoint & Serialization Rules

#### R19. Save `state_dict`, not the full model

```python
# BAD -- pickles the class definition, breaks on refactoring
torch.save(model, "model.pt")

# GOOD
torch.save(model.state_dict(), "model.pt")
# To load:
model = MyModel(...)
model.load_state_dict(torch.load("model.pt", weights_only=True))
```

**Rationale:** Pickling the full model creates brittle coupling to the class definition. Renaming a method or moving the class breaks loading. `state_dict` is a plain dict of parameter tensors.

**Security note:** Since PyTorch 2.6+ (our env is 2.5.1), `weights_only=True` is the default for `torch.load()`. In 2.5.1, you must pass it explicitly. Never load untrusted checkpoints without `weights_only=True` -- pickle can execute arbitrary code.

**Source:** [PyTorch Saving/Loading tutorial (official)](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

---

#### R20. Checkpoint for resume must include optimizer + scheduler state

```python
# COMPLETE checkpoint for resume
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss,
}
torch.save(checkpoint, path)
```

**Rationale:** Without optimizer state (Adam's momentum buffers, etc.), resumed training starts with fresh momentum and converges differently. Without scheduler state, the LR schedule restarts from epoch 0. Lightning handles this automatically via its checkpoint system.

**Source:** [PyTorch checkpointing tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html); [Lightning checkpointing docs](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)

---

### Tier 5: Reproducibility Rules (debug-time, not production)

#### R21. Full reproducibility requires 4 seeds + 2 flags + 1 env var

```python
import random, numpy as np, torch, os

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Required for deterministic GPU ops
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Must be set BEFORE CUDA context creation
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

**Caveats:**
- `torch.use_deterministic_algorithms(True)` raises errors for ops without deterministic implementations (e.g., some scatter ops). Pass `warn_only=True` to downgrade to warnings.
- Deterministic mode has a real performance penalty. Use for debugging only, not production training.
- DataLoader workers need per-worker seeding via `worker_init_fn`.
- Seeds must be set BEFORE model construction for full bitwise reproducibility.

**Source:** [stas00/ml-engineering reproducibility guide](https://github.com/stas00/ml-engineering/blob/master/training/reproducibility/README.md); [PyTorch Reproducibility guide (official)](https://docs.pytorch.org/docs/stable/notes/randomness.html)

---

## 2. Existing Published Agent Skills / Rulesets for PyTorch

### 2.1. PyTorch's own CLAUDE.md / AGENTS.md

**URL:** https://github.com/pytorch/pytorch/blob/main/CLAUDE.md (AGENTS.md is a symlink to it)

**Content:** Focused on contributing to the PyTorch framework itself, not on using PyTorch. Covers: build system (`pip install -e . -v --no-build-isolation`), test framework (`torch.testing._internal`), git/ghstack workflow, linting (`lintrunner -a`), CUDA conventions. The code style advice is minimal and PyTorch-internal: "minimize comments," "avoid trivial helper functions," "match existing patterns."

**Assessment:** Not useful as a template for user-facing PyTorch code rules. It's a contributing guide, not a style guide for ML code.

---

### 2.2. awesome-cursorrules: PyTorch + scikit-learn

**URL:** https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules/pytorch-scikit-learn-cursorrules-prompt-file/.cursorrules

**Content:** Targeted at chemistry/ML applications. Rules include: use autograd for custom losses, implement LR scheduling and early stopping, proper data splitting, clear project structure. Very generic -- reads like a checklist of ML basics rather than a set of checkable rules.

**Assessment:** Low quality for our purposes. Too generic, no code examples, no version-specific advice. The chemistry-domain framing makes it niche.

---

### 2.3. claude-ml-skills (tungcorn/K-Dense Inc.)

**URL:** https://github.com/tungcorn/claude-ml-skills

**Content:** 15 ML/DL skills for AI agents in the Anthropic Agent Skills format. The `pytorch-lightning` skill covers: LightningModule structure, Trainer configuration, multi-GPU/TPU, callbacks, logging (W&B, TensorBoard), distributed training (DDP, FSDP, DeepSpeed). Also has a general `pytorch-deep-learning` skill covering tensor ops through torch.compile and FSDP.

**Assessment:** Most promising existing resource. Well-structured with progressive disclosure (~500 lines per skill). However, I could not fetch the actual skill content (only the README), so the depth and accuracy of the rules is unverified. Worth fetching and evaluating the individual skill files.

---

### 2.4. mcpmarket.com PyTorch Deep Learning Skill

**URL:** https://mcpmarket.com/tools/skills/pytorch-deep-learning

**Content:** Covers tensor management, nn.Module patterns, training loops, AMP, torch.compile, DDP/FSDP, checkpointing, TorchScript/ONNX export. Emphasizes `inference_mode()` over `no_grad()`, deterministic settings, and "profile before optimize."

**Assessment:** Reasonable coverage of topics but positioned as a general-purpose skill. Missing Lightning-specific rules. The emphasis on TorchScript/ONNX export is production-oriented and less relevant for research.

---

### 2.5. IgorSusmelj/pytorch-styleguide

**URL:** https://github.com/IgorSusmelj/pytorch-styleguide

**Content:** Community guide (~2K GitHub stars). Covers: naming conventions (Google Python style), model architecture patterns (sequential, skip connections, multi-output), training loop template, project file organization, debugging with tqdm timing. Recommends `BackgroundGenerator` for async data loading.

**Assessment:** Good practical advice from a research/startup perspective, but dated in several ways: uses `super(ClassName, self).__init__()` (Python 2 style), `DataParallel` instead of `DistributedDataParallel`, manual device handling instead of Lightning. The project structure advice (separate `networks.py`, `layers.py`, `losses.py`, `ops.py`) is reasonable. Worth cherry-picking from but not adopting wholesale.

---

### 2.6. stas00/ml-engineering

**URL:** https://github.com/stas00/ml-engineering

**Content:** Comprehensive ML engineering open book by Stas Bekman (ex-HuggingFace). Covers reproducibility, debugging PyTorch, distributed training, performance. The reproducibility section is authoritative and well-maintained.

**Assessment:** High quality, but it's a reference book, not an agent ruleset. The reproducibility chapter is the most directly actionable piece -- the seed/determinism guidance is the best I found. The debugging section is also valuable but too long for a skill file.

---

## 3. Anti-Patterns with Canonical Fixes

### AP1. Accumulating loss without `.detach()` or `.item()`

**Symptom:** GPU memory grows linearly during training.
**Root cause:** Appending `loss` (a graph-attached tensor) to a list retains the entire computation graph.
**Fix:** `losses.append(loss.detach())` or `losses.append(loss.item())`.

---

### AP2. Forgetting `model.eval()` / `model.train()` toggle

**Symptom:** Validation metrics are noisy or don't match inference results.
**Root cause:** Dropout active during eval; BatchNorm using batch stats instead of running stats.
**Fix:** Always bracket validation with `model.eval()` before and `model.train()` after. In Lightning, this is handled automatically.

---

### AP3. Device mismatch between model and data

**Symptom:** `RuntimeError: Expected all tensors to be on the same device`.
**Root cause:** Model on GPU, labels or auxiliary tensors on CPU. Common with custom loss functions or metrics that create new tensors.
**Fix:** In Lightning, use `self.device` to create tensors. In vanilla PyTorch, establish a `device` variable early and use it everywhere.

```python
# BAD -- hard-codes device
target = torch.zeros(10).cuda()

# GOOD -- uses model's device
target = torch.zeros(10, device=self.device)
```

---

### AP4. `optimizer.zero_grad()` in the wrong place

**Symptom:** Gradients accumulate across batches (intentional for gradient accumulation, but a bug if unintended).
**Root cause:** `zero_grad()` called after `optimizer.step()` instead of before `loss.backward()`, or simply missing.
**Fix:** Standard order is: `zero_grad()` -> `forward` -> `loss.backward()` -> `optimizer.step()`. In Lightning, this is automatic.

---

### AP5. NumPy operations inside `nn.Module.forward()`

**Symptom:** Slow forward pass, broken autograd, or tensors silently detached.
**Root cause:** NumPy ops don't participate in PyTorch's computation graph. The tensor->numpy->tensor round-trip detaches gradients.
**Fix:** Use PyTorch tensor operations. If you must use NumPy (e.g., for a library that requires it), do so outside the forward pass and `.detach()` explicitly.

**Source:** [IgorSusmelj styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)

---

### AP6. Calling `.forward()` directly instead of `module(input)`

**Symptom:** Hooks don't fire, some module bookkeeping is skipped.
**Root cause:** `module(input)` calls `__call__`, which runs hooks and then `forward`. Calling `.forward()` directly bypasses this.
**Fix:** Always call `output = module(input)`, never `output = module.forward(input)`.

**Source:** [IgorSusmelj styleguide](https://github.com/IgorSusmelj/pytorch-styleguide); [PyTorch nn.Module docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

---

### AP7. Using `DataParallel` instead of `DistributedDataParallel`

**Symptom:** Poor multi-GPU scaling, GIL contention.
**Root cause:** `DataParallel` uses threading (GIL-bound), `DistributedDataParallel` uses multiprocessing with NCCL. DDP is faster even on a single machine.
**Fix:** Use DDP (or let Lightning handle it with `strategy="ddp"`).

**Note:** In a research codebase, if you're only ever using 1 GPU, this doesn't matter. **Production-context rule.**

**Source:** [PyTorch Performance Tuning Guide (official)](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

### AP8. Excessive logging per iteration

**Symptom:** Training is slow despite fast model/data.
**Root cause:** Logging to disk/network every step, or worse, creating matplotlib figures every step.
**Fix:** Log every N steps. In Lightning, `self.log()` respects `log_every_n_steps` (default 50).

**Source:** [IgorSusmelj styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)

---

### AP9. Hardcoded `.cuda()` calls

**Symptom:** Code breaks on CPU-only machines, multi-GPU setups, or Apple Silicon.
**Root cause:** `.cuda()` hard-codes device 0. Doesn't compose with device placement strategies.
**Fix:** Use `.to(device)` with a configurable device, or in Lightning, rely on automatic device placement.

```python
# BAD
model = Model().cuda()
x = x.cuda()

# GOOD (vanilla PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
x = x.to(device)

# GOOD (Lightning) -- nothing to do, Trainer handles it
```

---

### AP10. Not unscaling before gradient clipping with AMP

**Symptom:** Gradient clipping thresholds are applied to scaled gradients, making the clip value effectively wrong.
**Root cause:** `GradScaler` scales the loss (and therefore gradients). Clipping must happen after unscaling.
**Fix:**

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # unscale FIRST
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

In Lightning, this is handled automatically when you set `gradient_clip_val` on the Trainer.

**Source:** [PyTorch AMP docs](https://docs.pytorch.org/docs/main/amp.html)

---

## 4. Testing ML Code: What to Assert

### Discriminating tests (worth writing)

| Test | What it catches | Example |
|---|---|---|
| **Output shape** | Architecture bugs, off-by-one in dimensions | `assert model(x).shape == (B, C, H, W)` |
| **Gradient flow** | Dead layers, detached tensors, missing connections | After `loss.backward()`, assert `all(p.grad is not None for p in model.parameters() if p.requires_grad)` |
| **No NaN in gradients** | Numerical instability, log(0), div/0 | `assert all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)` |
| **Loss decreases on overfit** | Broken loss, learning rate, optimizer setup | Train for 500 steps on 1 batch, assert final_loss < initial_loss * 0.01 |
| **Batch independence** | Accidental cross-batch information leakage | Zero one sample in a batch, verify only that sample's gradient is zero |
| **Train/eval mode effects** | Dropout/BN working correctly | Assert `model.eval()` output is deterministic across calls (Dropout disabled) |
| **Device movement** | Tensors left on wrong device | `model.to("cuda"); assert model(x.cuda()).device.type == "cuda"` |
| **Parameter count** | Accidental architecture changes | `assert sum(p.numel() for p in model.parameters()) == expected_count` |

### Tests NOT worth writing

- Tests that assert exact tensor values (fragile, restate the implementation)
- Tests of built-in PyTorch/Lightning functionality (e.g., "does DataLoader batch correctly")
- Tests that require exact floating-point equality without tolerances
- Tests of private implementation details that will change

### Testing practices

- **Seed everything** in test fixtures (`@pytest.fixture(autouse=True)` that calls `seed_everything`)
- **Make tests fast** -- use tiny models (2 layers, 8 channels), tiny datasets (2-10 samples), few steps (5-50)
- **Use `torch.testing.assert_close`** instead of manual tolerance checks
- **Test statistical properties** for stochastic components (e.g., dropout output mean should be ~input mean)

**Sources:** [How to Trust Your DL Code (krokotsch.eu)](https://krokotsch.eu/posts/deep-learning-unit-tests/); [torchtest library](https://github.com/suriyadeepan/torchtest); [Don't Mock ML Models (Eugene Yan)](https://eugeneyan.com/writing/unit-testing-ml/)

---

## 5. What I Could NOT Verify / Sources Conflict / Looks Outdated

### 5.1. Lightning Style Guide content unverifiable

The Lightning style guide at `lightning.ai/docs/pytorch/stable/starter/style_guide.html` renders via JavaScript and could not be fetched for detailed content extraction. The search-result summaries confirm the model-vs-system separation principle, but I could not verify the full list of recommendations or check for version-specific changes in Lightning 2.6.x. **Recommendation:** Read the page manually in a browser.

### 5.2. `torch.compile` advice is rapidly evolving

Edward Yang's blog posts (PyTorch core dev) from Nov 2024 and Aug 2025 are the best available guidance, but torch.compile behavior changes significantly between minor versions. Our environment (torch 2.5.1) may not support all the features described in the Aug 2025 post (which targets torch 2.6+). **Version-sensitive: verify before adopting.**

### 5.3. `weights_only=True` default in `torch.load`

Multiple sources state this became the default in "PyTorch 2.6" or "2.13.0" -- the version numbers are inconsistent across sources (the "2.13.0" claim is likely a torch version numbering confusion). In our torch 2.5.1 environment, `weights_only` defaults to `False` with a deprecation warning. **Must pass explicitly in our env.**

### 5.4. `BackgroundGenerator` recommendation (IgorSusmelj guide)

The styleguide recommends `prefetch_generator.BackgroundGenerator` for async data loading. This was useful before PyTorch's DataLoader gained `num_workers > 0` and `persistent_workers`. With modern DataLoader settings, `BackgroundGenerator` is redundant. The guide hasn't been updated to reflect this. **Outdated advice.**

### 5.5. `super(ClassName, self).__init__()` in community guides

Multiple community guides use the Python 2 form. In Python 3, `super().__init__()` is preferred and standard. **Cosmetic but signals guide age.**

### 5.6. Disagreement: should you use `torch.compile` in research?

- **Yang (PyTorch core):** Use it selectively, maintain eager baselines, expect numerical differences. Worth it for runs > few hours.
- **Community practice:** Most research code does not use `torch.compile` yet (based on search results and repo surveys).
- **Our assessment:** For our use case (research, not production), torch.compile is opt-in. Don't require it, don't prevent it, but always keep eager-compatible code.

### 5.7. `GradScaler` necessity with bfloat16

With bfloat16 precision (available on Ampere+ GPUs), `GradScaler` is generally not needed because bfloat16 has the same exponent range as float32 and doesn't suffer from the underflow that float16 does. Lightning handles this automatically (`precision="bf16-mixed"` skips GradScaler). Some sources don't distinguish between float16 and bfloat16 when discussing AMP. **Matters for our CUDA 12.4 / Ampere+ env.**

### 5.8. claude-ml-skills quality unverified

The `tungcorn/claude-ml-skills` repository looks well-structured from the README, but I could not fetch the actual skill file contents (only the README listing the 15 skills). The skill files may be high quality or may be superficial -- **needs manual review before adoption.**

### 5.9. Lightning `manual_optimization` ignores scheduler config

The Lightning docs state that in manual optimization mode, `interval`, `frequency`, and other lr_scheduler_config keys from `configure_optimizers` are ignored silently. This is documented but easy to miss -- if you switch from automatic to manual optimization, your LR schedule silently stops working unless you call `self.lr_schedulers().step()` yourself.

---

## Appendix A: Source Index

### Official Documentation
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- **primary source for performance rules**
- [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html) -- **primary source for in-place/gradient rules**
- [PyTorch AMP docs](https://docs.pytorch.org/docs/main/amp.html) -- mixed precision API
- [PyTorch Saving/Loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) -- checkpointing
- [PyTorch DataLoader docs](https://docs.pytorch.org/docs/2.14/data.html) -- DataLoader parameters
- [Lightning Style Guide](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) -- LightningModule structure
- [Lightning Speed Guide](https://lightning.ai/docs/pytorch/stable/advanced/speed.html) -- training speed tips
- [Lightning Checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) -- checkpoint resume
- [Lightning LightningModule docs](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) -- configure_optimizers contract

### Authoritative Community Sources
- [stas00/ml-engineering](https://github.com/stas00/ml-engineering/blob/master/training/reproducibility/README.md) -- reproducibility (Stas Bekman, ex-HuggingFace)
- [Edward Yang's blog](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/) -- torch.compile (PyTorch core dev)
- [IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide) -- community style guide (~2K stars, research perspective)
- [How to Trust Your DL Code](https://krokotsch.eu/posts/deep-learning-unit-tests/) -- unit testing ML code
- [Don't Mock ML Models](https://eugeneyan.com/writing/unit-testing-ml/) -- ML testing philosophy

### Agent Skills / Rulesets
- [PyTorch CLAUDE.md](https://github.com/pytorch/pytorch/blob/main/CLAUDE.md) -- framework contributing guide (not user-facing style)
- [claude-ml-skills](https://github.com/tungcorn/claude-ml-skills) -- 15 ML skills for Claude/Cursor/Cline agents
- [awesome-cursorrules PyTorch](https://github.com/PatrickJS/awesome-cursorrules) -- .cursorrules collection
- [mcpmarket PyTorch skill](https://mcpmarket.com/tools/skills/pytorch-deep-learning) -- MCP marketplace skill
