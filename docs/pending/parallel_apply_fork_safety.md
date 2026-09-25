# `parallel_apply` fork safety

**Status:** design, not implemented. Owner-approved direction; open questions at the end.
**Date:** 2026-09-25
**Applies to:** `DataFrameBase.parallel_apply` / `_parallel_apply` in `fragmentomics_tools/dataframe.py`

---

## 1. Evidence first

Every claim below was measured on this platform (Python 3.12.14, glibc 2.41,
numpy 2.5.3 against MKL, 16 logical / 8 physical cores). This section leads
because several confident-sounding explanations of this bug turned out to be
wrong, and the ones that survived contact with a measurement are a much
smaller set than the ones that did not.

### 1.1 The actual root cause, from a live capture of the real failure

The suite hung a third time on 2026-09-25 while this document was being
written. `py-spy` on the live processes gave the mechanism directly, so the
following is not inferred from a synthetic reproduction — it is the failure
itself.

**Child** (`pid 573753`, 14 min elapsed, ~0 s CPU), innermost frame first:

```
wakeup       (concurrent/futures/process.py:87)    <- blocked on a futex
_python_exit (concurrent/futures/process.py:104)
_shutdown    (threading.py:1594)
_bootstrap   (multiprocessing/process.py:332)
_launch      (multiprocessing/popen_fork.py:71)    <- the fork
... inherited parent stack ...
_spawn_process / _start_executor_manager_thread / submit / map
_parallel_apply (dataframe.py:283)
run             (test_parallel_apply.py:295)       <- a caller THREAD
```

**Parent** (`pid 573088`): MainThread in `t.join()` at the test; `Thread-20`
in `ProcessPoolExecutor.shutdown -> join`; `Thread-24` in
`Process.join -> popen_fork.poll`; `tqdm_monitor` idle in `wait`.

The relevant CPython source, which flags its own hazard in a comment:

```python
def _python_exit():
    items = list(_threads_wakeups.items())
    for _, thread_wakeup in items:
        # call not protected by ProcessPoolExecutor._shutdown_lock
        thread_wakeup.wakeup()        # -> `with self._lock:`
    for t, _ in items:
        t.join()
```

**Mechanism.** `_threads_wakeups` is *module-level* state in
`concurrent.futures.process`, mapping executor-manager threads to
`_ThreadWakeup` objects that each carry a lock. Three caller threads each
build their own `ProcessPoolExecutor` concurrently. One thread forks from
inside `submit -> _start_executor_manager_thread -> _spawn_process` while a
sibling thread holds one of those locks. The child inherits the entire dict —
describing threads that do not exist in it — and at interpreter shutdown
`threading._shutdown()` invokes the registered `_python_exit()`, which walks
the dict and calls `wakeup()`. That blocks on the orphaned lock forever. The
parent then waits on a child that can never exit.

**The trigger is concurrent `ProcessPoolExecutor` creation from multiple
threads.** Not the progress bar, not the allocator, not BLAS.

### 1.1b Synthetic vectors, and how badly they mislead

Each of these was constructed *before* the live capture, while reasoning about
what "could" hang a child. Only one reproduced, and it turned out not to be
what actually happens.

| Vector | Trials | Hangs | Bearing on the real failure |
|---|---:|---:|---|
| tqdm lock held at fork | 1 | 1 | Real deadlock, **but not this one.** `tqdm_monitor` is idle in the live dump. |
| malloc contention (6 threads) | 60 | 0 | Not a factor. |
| MKL pool alive, idle | 60 | 0 | Not a factor. |
| MKL pool actively computing | 60 | 0 | Not a factor. |
| single-threaded control | 60 | 0 | — |

The tqdm result is the instructive one: a genuine, reproducible deadlock that
is **not the bug**. Constructing a plausible failure and reproducing it is not
the same as diagnosing the failure in front of you.

### 1.2 Claims that were tested and found FALSE

These are recorded because each was asserted during analysis, and each is the
kind of thing that sounds authoritative enough to survive a code review.

| Claim | Verdict |
|---|---|
| "The reproduced tqdm-lock deadlock is what hangs the suite." | **False.** The live capture shows `tqdm_monitor` idle and the block in `concurrent.futures.process`. A reproducible deadlock that is not the one occurring. |
| "This is a structural `fork`-versus-native-threads problem; BLAS is the real enemy." | **False.** It is a specific CPython module-state-across-fork bug, triggered by concurrent executor creation from threads. |
| "The child hangs on its first `malloc` because a thread held the allocator lock." | **False here.** glibc 2.41 locks all malloc arenas before `fork` and reinitialises them in the child. True on old glibc; it is the version that circulates. 0 hangs in 60 trials under deliberate contention. |
| "The BLAS is OpenBLAS." | **False.** It is MKL (`libmkl_*.so`; the loaded OpenMP runtime is LLVM `libomp.so`). Consequence: `OPENBLAS_NUM_THREADS=1` silently does nothing. `OMP_NUM_THREADS=1` works. |
| "CPython's `os.fork()` DeprecationWarning only counts Python threads." | **False.** It fired with `threading.active_count() == 1` and 8 OS threads. It reads the OS count. |
| "The first `parallel_apply` call in a process forks cleanly." | **Misleading.** True only if the process has done no BLAS work. After one matmul the first call already forks an 8-thread process. |
| "`threadpoolctl.threadpool_limits(1)` removes the BLAS threads." | **False.** Thread count stays at 8. It caps how many MKL *uses*, not how many *exist*. |

### 1.3 Claims that were tested and held

| Claim | Evidence |
|---|---|
| `fork` keeps only the calling thread; the child cannot see what the parent had. | Parent 8 OS threads at fork, child reports 1. |
| A `disable=True` tqdm bar still takes tqdm's lock. | Instrumented the lock: **2 acquisitions** for a disabled bar. `__new__` takes it before `__init__` reads `disable`. |
| `verbose=False` does not prevent the tqdm monitor thread. | `tqdm.__new__` starts `TMonitor` before `disable` is consulted. |
| The monitor can be prevented and stopped. | `tqdm.monitor_interval = 0` prevents creation; `tqdm.monitor.exit()` stops a running one, costing **0.08 ms**. |
| `monitor_interval = 0` does **not** stop an already-running monitor. | Measured; only prevents new creation. This is the common notebook case. |
| The MKL/OpenMP pool can be destroyed and rebuilt at runtime. | `omp_pause_resource_all(omp_pause_hard)` on the mapped `libomp.so`: 8 → 1 threads, 3/3. Rebuilds on next BLAS call at no measurable cost (52.2 ms steady vs 45.9 ms after pause, best-of-5). |
| A fork taken while the pool is paused is clean. | Parent 1 thread, child 1 thread. |
| Detecting the risk condition is cheap. | `os.listdir('/proc/self/task')` = 16.7 µs; `/proc/self/status` = 56.2 µs. Against a fork costing milliseconds. |

### 1.4 Two traps in `omp_pause_resource_all`

Both reproduced 3/3:

- **Calling `omp_pause_soft` first makes the subsequent `omp_pause_hard` fail** (`rc=1`, threads stay at 8).
- **The return code is not a success signal.** `rc=1` means *either* "failed, threads still alive" (after a soft pause) *or* "already paused, benign no-op" (double hard pause). Same code, opposite meanings.

The second trap is why the design below asserts on a thread count rather than
branching on `rc`.

---

## 2. The problem

`_parallel_apply` creates workers with `multiprocessing.get_context("fork")`.
`fork` duplicates the memory image but keeps only the calling thread. Any lock
another thread held at that instant is copied in its locked state, owned by a
thread that no longer exists in the child. Nothing will release it, and the
next code in the child to want that lock blocks forever.

It is a race: the lock must be held at the microsecond of the fork. That is
why the same commit hung for 12 hours once and passed in 82 seconds on re-run.

The specific lock is CPython's own. See §1.1 — `concurrent.futures.process`
keeps a module-level `_threads_wakeups` dict whose entries carry locks, and
its `_python_exit` handler walks that dict in the child, where the threads it
describes do not exist.

**Observed failures.** Three occurrences, all in
`test_parallel_apply.py::TestConcurrentCallers::test_threads_do_not_corrupt_each_other`,
which runs three caller threads that each build their own executor:

| # | Duration | CPU | Outcome |
|---|---|---|---|
| 1 | 12 h 03 m | 53 s | Killed. Two children at 0 s CPU. |
| 2 | — | — | Passed in ~1 s on re-run at the identical commit. |
| 3 | 15 m+ | 38 s | Killed. One child at 0 s CPU. **Live `py-spy` capture — §1.1.** |

Occurrence 2 is why this must be treated as a race rather than a
deterministic failure: the same commit both hung and passed.

**This is inherited, not introduced.** The pre-rewrite implementation
(`6e6583d^`) also used a fork context and also built its bar after starting
workers. One thing did regress: it guarded bar construction with
`if verbose:`, so `verbose=False` created no bar and no monitor. The current
code constructs unconditionally and passes `disable=(not verbose)`, which does
not prevent the monitor. Two call sites that deliberately pass `verbose=False`
(`btd_experiments.py:148`, `btd13.py:221`) silently lost that protection.

### 2.1 Blast radius

- **Zero threaded callers in production.** Across `fragmentomics_tools`,
  `biomarker`, `biomarker-pipeline` and `biomarker-projects`, the only caller
  of `parallel_apply` from a non-main thread is the guard test itself. One
  near-miss — `cdx2_filtered_vs_random_control.py` runs four library jobs in a
  `ThreadPoolExecutor` — reaches neither `parallel_apply` nor either
  transitive entry point.
- **Transitive entry points widen it:** `load_fragment_arrays` and
  `set_fragment_array_weights` both call `parallel_apply` internally — 16 `.py`
  and 22 notebook-source call sites are indirect callers.
- **Notebook call sites:** 291 source-cell hits across 154 files (a further 54
  hits in 17 files are *output* cells, not calls, and are excluded).

---

## 3. Design

### 3.1 Refuse to fork from a non-main thread — this is the fix

The root cause needs **concurrent executor creation from multiple threads**.
Remove that and the bug cannot occur. Before creating the pool, if the process
has Python-level threads beyond the caller, raise with a clear message rather
than forking.

This directly targets the measured mechanism, and nothing else in this
document is load-bearing for it.

Blast radius is essentially nil: across four repos there are **zero**
production callers of `parallel_apply` from a non-main thread (§2.1). The only
one is the guard test.

### 3.2 Stop our own progress bar leaving a thread behind

Independent of the root cause, `parallel_apply` leaves a `tqdm_monitor` thread
alive after its first call, which makes every later call in that process fork
multi-threaded. That is a real self-inflicted hazard (an orphaned tqdm lock
*is* a reproducible deadlock — §1.1b — just not the one we hit), and it also
interacts with §3.1: without this, the check in §3.1 would fire inside workers
during nested calls.

Set `tqdm.monitor_interval = 0` before constructing our bar and stop any
already-running monitor, restoring the previous value in a `finally`. Note
that setting the interval alone is **not** sufficient — it prevents creation
but does not stop a monitor that already exists, which is the common notebook
case. `TMonitor.exit()` costs 0.08 ms.

### 3.3 Do not apply the check on the `n_workers == 1` path

That path runs in-process and never forks.

### 3.4 What about the BLAS threads?

**Dropped from this design.** They were the subject of considerable
investigation (§1.3, §1.4) before the live capture showed them to be
irrelevant: 120 trials produced no hang, and the real failure has nothing to
do with them.

The `omp_pause_resource_all` findings are retained in §1.3/§1.4 because they
are correct and were expensive to establish, and because a future OS-level
thread check would need them. They should not be implemented now. Adding an
MKL pause to fix a CPython module-state bug would be unexplainable to the next
reader, and §1.4's traps make it a non-trivial amount of fragile code.

### 3.5 What this does to the two pass-or-hang tests

`test_threads_do_not_corrupt_each_other` and `test_nested_calls_work`
currently have exactly two outcomes: pass, or hang forever. Under this design
the threaded one becomes a `pytest.raises` assertion — deterministic and
instant. This is strictly better than adding `join(timeout=...)`, which would
convert a hang into a *flake*, and flaky tests get ignored and then deleted.

The data-isolation property that test guards is real and worth keeping; it
caught a silent-wrong-answer regression during the original rewrite. It needs
re-expressing in a form that does not require forking from threads.

---

## 4. What this design does NOT do

- It does not make `parallel_apply` safe to call from multiple threads. It
  makes that case fail loudly instead of hanging. Given zero production
  threaded callers, that is a deliberate narrowing of the contract.
- It does not touch BLAS threads. They are present but were shown not to cause
  this failure — see §3.4.
- It does not move off `fork`. `fork` is what allows `fn` to be a lambda and
  keeps the frame from being serialized per call. `spawn` and `forkserver`
  both require pickling `fn` and the frame.

---

## 5. Open questions

1. **Has this ever affected real use?** Exactly one occurrence is known: an
   overnight test-suite run, no user waiting. If notebook users have been
   restarting hung kernels without reporting it, the priority is higher than
   the evidence currently supports.
2. **Portability of the OpenMP pause.** Measured only against LLVM `libomp`
   via conda MKL. Intel `libiomp5` is present in the environment but is not
   the runtime that loads. Untested.
3. **Pausing while a native caller is inside a BLAS call.** The thread check
   catches our own threads; it does not establish that pausing mid-computation
   from another thread is safe.
4. **Should the suite gain a `timeout` wrapper regardless?** Orthogonal to
   this design, and it is what turned a silent 12-hour wedge into a reportable
   event.

---

## 6. Provenance

The analysis behind this document went wrong repeatedly before it went right,
and the corrections are listed in §1.2 rather than quietly omitted.

The decisive lesson is §1.1b. Four mechanisms were proposed — the allocator,
BLAS, tqdm, and finally CPython's executor state. One of them (tqdm) was even
*reproduced deterministically in a purpose-built harness*, which felt like
confirmation and was not: the live capture shows `tqdm_monitor` sitting idle
while the process deadlocked somewhere else entirely.

What finally worked was not better reasoning. It was catching the failure
alive and running `py-spy` on it. Every hypothesis before that was constructed
from plausibility; the answer came from the stack of the thing that was
actually stuck. Where a bug is reproducible-but-rare, the effort is better
spent on instrumenting a real occurrence than on building a harness for a
guess.

The running record, including each retraction and what disproved it, is in
`COORDINATION.fragmentomics_tools.md` under the F10 gate rows.
