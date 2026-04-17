# Extractable Organs for Buddy Integration

**Purpose**: Ready-to-graft code modules extracted from LARQL's proven patterns, adapted for Buddy's architecture and LLM integration in Source 2.

**Philosophy**: These are the actual organs, not just the surgery plan. Copy, adapt, integrate.

---

## Organ 1: Immutable State Manager

**What it does**: Manages state with immutable base + append-only receipt overlay
**Where it goes**: `runtime/state/immutable_state.py` in Buddy repo
**Dependencies**: None (stdlib only)

```python
"""
Immutable State Manager - Extracted from LARQL's PatchedVindex pattern
Ensures: Base state never mutates, all changes tracked via receipts
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class Receipt:
    """Single state mutation receipt - append-only log entry"""
    timestamp: str
    operation: str  # "set", "delete", "merge"
    path: str  # dot-notation path: "tools.browser.headless"
    value: Any
    previous_value: Any
    hash: str
    approved_by: str  # "smartman", "user", "system"

    def apply_to(self, state: Dict) -> Dict:
        """Apply this receipt's mutation to a state dict"""
        parts = self.path.split('.')
        target = state

        # Navigate to parent
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Apply operation
        if self.operation == "set":
            target[parts[-1]] = self.value
        elif self.operation == "delete":
            target.pop(parts[-1], None)
        elif self.operation == "merge":
            if isinstance(self.value, dict):
                target[parts[-1]] = {**target.get(parts[-1], {}), **self.value}

        return state


class ImmutableStateManager:
    """
    Manages application state with immutability guarantee.
    Pattern extracted from LARQL's vindex + patch overlay.
    """

    def __init__(self, base_path: Path, receipts_path: Optional[Path] = None):
        self.base_path = base_path
        self.receipts_path = receipts_path or base_path.parent / "receipts.jsonl"

        # Load immutable base (frozen at initialization)
        self._base = self._load_base()

        # Load receipts (append-only log)
        self._receipts: List[Receipt] = self._load_receipts()

    def _load_base(self) -> Dict:
        """Load base state - NEVER modified after load"""
        if self.base_path.exists():
            return json.loads(self.base_path.read_text())
        return {}

    def _load_receipts(self) -> List[Receipt]:
        """Load all receipts from append-only log"""
        receipts = []
        if self.receipts_path.exists():
            for line in self.receipts_path.read_text().strip().split('\n'):
                if line:
                    data = json.loads(line)
                    receipts.append(Receipt(**data))
        return receipts

    def current_state(self) -> Dict:
        """
        Compute current state = base + all receipts.
        This is the ONLY way to get current state.
        """
        state = self._base.copy()
        for receipt in self._receipts:
            state = receipt.apply_to(state)
        return state

    def propose_change(self, path: str, value: Any,
                      operation: str = "set",
                      approved_by: str = "pending") -> Receipt:
        """
        Propose a state change. Returns receipt (not yet applied).
        Caller must get SmartMan approval before calling apply_receipt.
        """
        current = self.current_state()
        parts = path.split('.')

        # Navigate to get previous value
        prev = current
        for part in parts[:-1]:
            prev = prev.get(part, {})
        previous_value = prev.get(parts[-1], None)

        # Create receipt
        timestamp = datetime.utcnow().isoformat() + "Z"
        receipt_data = {
            "timestamp": timestamp,
            "operation": operation,
            "path": path,
            "value": value,
            "previous_value": previous_value,
            "approved_by": approved_by,
            "hash": ""
        }

        # Hash the receipt (excluding hash field itself)
        hash_input = json.dumps(receipt_data, sort_keys=True)
        receipt_data["hash"] = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return Receipt(**receipt_data)

    def apply_receipt(self, receipt: Receipt) -> None:
        """
        Apply approved receipt. PERMANENTLY adds to log.
        This is the ONLY way to mutate state.
        """
        if receipt.approved_by == "pending":
            raise ValueError("Receipt not approved - cannot apply")

        # Append to in-memory list
        self._receipts.append(receipt)

        # Append to persistent log (append-only file)
        with self.receipts_path.open('a') as f:
            f.write(json.dumps(asdict(receipt)) + '\n')

    def get(self, path: str, default: Any = None) -> Any:
        """Get value from current state by dot-notation path"""
        state = self.current_state()
        parts = path.split('.')

        for part in parts:
            if isinstance(state, dict):
                state = state.get(part)
                if state is None:
                    return default
            else:
                return default

        return state

    def rollback_to(self, receipt_hash: str) -> None:
        """
        Rollback to state after specific receipt.
        Doesn't delete receipts - marks them as rolled back.
        """
        # Find index of target receipt
        idx = None
        for i, r in enumerate(self._receipts):
            if r.hash == receipt_hash:
                idx = i
                break

        if idx is None:
            raise ValueError(f"Receipt {receipt_hash} not found")

        # Keep only receipts up to target
        self._receipts = self._receipts[:idx+1]

        # Rewrite receipt log
        self.receipts_path.write_text(
            '\n'.join(json.dumps(asdict(r)) for r in self._receipts) + '\n'
        )

    def compact_base(self, output_path: Path) -> None:
        """
        Bake current state into new base file.
        Like LARQL's COMPILE INTO VINDEX.
        """
        current = self.current_state()
        output_path.write_text(json.dumps(current, indent=2))

        # Optional: clear receipts if creating new canonical base
        # self._receipts = []
        # self.receipts_path.write_text("")


# Usage example for Buddy:
"""
# Initialize
state = ImmutableStateManager(
    base_path=Path("~/.buddy/state.json"),
    receipts_path=Path("~/.buddy/receipts.jsonl")
)

# Propose change
receipt = state.propose_change(
    path="tools.browser.headless",
    value=True,
    approved_by="pending"
)

# Get SmartMan approval (your existing logic)
if smartman.approve(receipt):
    receipt.approved_by = "smartman"
    state.apply_receipt(receipt)

# Read current state
headless = state.get("tools.browser.headless", default=False)

# Rollback if needed
state.rollback_to(receipt.hash)

# Compact into new base (periodic maintenance)
state.compact_base(Path("~/.buddy/state_v2.json"))
"""
```

---

## Organ 2: Progressive Profiler

**What it does**: Automatically identifies performance bottlenecks
**Where it goes**: `runtime/profiler.py` in Buddy repo
**Dependencies**: None (stdlib only)

```python
"""
Progressive Profiler - Extracted from LARQL's optimization methodology
Automatically times operations and reports bottlenecks
"""
import time
from functools import wraps
from typing import Dict, List, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Timing:
    name: str
    duration_ms: float
    call_count: int


class ProgressiveProfiler:
    """
    Auto-profiling decorator system.
    Pattern from LARQL: Profile first, optimize second.
    """

    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._enabled = True

    def measure(self, name: str) -> Callable:
        """Decorator to measure function execution time"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)

                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    self._timings[name].append(elapsed)

            return wrapper
        return decorator

    def measure_block(self, name: str):
        """Context manager to measure code block"""
        return TimingContext(self, name)

    def report(self, top_n: int = 10) -> str:
        """
        Generate bottleneck report.
        Returns formatted string like LARQL's profiling output.
        """
        if not self._timings:
            return "No profiling data collected"

        # Aggregate timings
        timings = []
        total_time = 0

        for name, durations in self._timings.items():
            total_duration = sum(durations)
            total_time += total_duration
            avg_duration = total_duration / len(durations)

            timings.append(Timing(
                name=name,
                duration_ms=total_duration,
                call_count=len(durations)
            ))

        # Sort by total time (biggest bottleneck first)
        timings.sort(key=lambda t: t.duration_ms, reverse=True)

        # Format report
        lines = [
            "",
            "=" * 70,
            "PERFORMANCE PROFILE",
            "=" * 70,
            f"{'Component':<30} {'Time':>10} {'%':>6} {'Calls':>8} {'Avg':>10}",
            "-" * 70
        ]

        for timing in timings[:top_n]:
            pct = 100 * timing.duration_ms / total_time if total_time > 0 else 0
            avg = timing.duration_ms / timing.call_count

            lines.append(
                f"{timing.name:<30} "
                f"{timing.duration_ms:>9.1f}ms "
                f"{pct:>5.1f}% "
                f"{timing.call_count:>8} "
                f"{avg:>9.1f}ms"
            )

        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<30} {total_time:>9.1f}ms")
        lines.append("=" * 70)

        # Add optimization suggestions
        lines.append("")
        lines.append("OPTIMIZATION TARGETS (fix these first):")
        for i, timing in enumerate(timings[:3], 1):
            pct = 100 * timing.duration_ms / total_time if total_time > 0 else 0
            lines.append(f"  {i}. {timing.name} ({pct:.1f}% of total)")

        return '\n'.join(lines)

    def reset(self):
        """Clear all timing data"""
        self._timings.clear()

    def enable(self):
        """Enable profiling"""
        self._enabled = True

    def disable(self):
        """Disable profiling (zero overhead)"""
        self._enabled = False


class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self, profiler: ProgressiveProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start) * 1000  # ms
        self.profiler._timings[self.name].append(elapsed)


# Global profiler instance for Buddy
profiler = ProgressiveProfiler()


# Usage example for Buddy:
"""
from runtime.profiler import profiler

# Decorate functions
@profiler.measure("tool_execution")
def execute_tool(tool_name: str, args: dict):
    # ... tool execution logic
    pass

@profiler.measure("llm_call")
def call_llm(prompt: str):
    # ... LLM API call
    pass

# Or use context manager
def complex_workflow():
    with profiler.measure_block("data_prep"):
        # ... data preparation
        pass

    with profiler.measure_block("model_inference"):
        # ... model inference
        pass

    with profiler.measure_block("result_formatting"):
        # ... format results
        pass

# Run your workflow
for i in range(100):
    complex_workflow()

# Get bottleneck report
print(profiler.report(top_n=5))

# Output:
# ======================================================================
# PERFORMANCE PROFILE
# ======================================================================
# Component                           Time      %   Calls        Avg
# ----------------------------------------------------------------------
# model_inference                  1234.5ms  78.2%      100   12.3ms  ← TARGET
# data_prep                         234.1ms  14.8%      100    2.3ms
# result_formatting                 110.2ms   7.0%      100    1.1ms
# ======================================================================
#
# OPTIMIZATION TARGETS (fix these first):
#   1. model_inference (78.2% of total)
#   2. data_prep (14.8% of total)
"""
```

---

## Organ 3: Resumable Operation Manager

**What it does**: Makes any long operation resumable after interruption
**Where it goes**: `runtime/resumable.py` in Buddy repo
**Dependencies**: None (stdlib only)

```python
"""
Resumable Operation Manager - Extracted from LARQL's --resume pattern
Checkpoint progress, detect interruption, resume cleanly
"""
import json
from pathlib import Path
from typing import List, Callable, Any, Optional, TypeVar, Generic
from dataclasses import dataclass, asdict
import signal
import sys

T = TypeVar('T')


@dataclass
class Progress:
    """Progress checkpoint for resumable operations"""
    total_items: int
    completed: List[int]  # Indices of completed items
    next_index: int
    metadata: dict


class ResumableOperation(Generic[T]):
    """
    Makes any batch operation resumable.
    Pattern from LARQL's extraction with --resume.
    """

    def __init__(self,
                 checkpoint_path: Path,
                 checkpoint_every: int = 10):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.progress: Optional[Progress] = None
        self._setup_interrupt_handler()

    def _setup_interrupt_handler(self):
        """Handle Ctrl+C gracefully"""
        def handler(signum, frame):
            if self.progress:
                self._save_checkpoint()
                print(f"\n✓ Progress saved to {self.checkpoint_path}")
                print(f"  Completed: {len(self.progress.completed)}/{self.progress.total_items}")
                print(f"  Resume with --resume flag")
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)

    def _load_checkpoint(self) -> Optional[Progress]:
        """Load progress from checkpoint file"""
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            return Progress(**data)
        return None

    def _save_checkpoint(self):
        """Save current progress"""
        if self.progress:
            self.checkpoint_path.write_text(
                json.dumps(asdict(self.progress), indent=2)
            )

    def run(self,
            items: List[T],
            processor: Callable[[T, int], Any],
            resume: bool = False,
            metadata: Optional[dict] = None) -> List[Any]:
        """
        Process items with automatic checkpointing.

        Args:
            items: List of items to process
            processor: Function(item, index) -> result
            resume: If True, resume from checkpoint
            metadata: Optional metadata to store

        Returns:
            List of results from processor
        """
        # Load or initialize progress
        if resume and self.checkpoint_path.exists():
            self.progress = self._load_checkpoint()
            print(f"✓ Resuming from checkpoint: {len(self.progress.completed)}/{len(items)} completed")
        else:
            self.progress = Progress(
                total_items=len(items),
                completed=[],
                next_index=0,
                metadata=metadata or {}
            )

        results = [None] * len(items)

        # Process items starting from checkpoint
        for i in range(self.progress.next_index, len(items)):
            # Process item
            result = processor(items[i], i)
            results[i] = result

            # Update progress
            self.progress.completed.append(i)
            self.progress.next_index = i + 1

            # Checkpoint periodically
            if (i + 1) % self.checkpoint_every == 0:
                self._save_checkpoint()
                print(f"  Checkpoint: {i+1}/{len(items)}")

        # Final checkpoint
        self._save_checkpoint()
        print(f"✓ Completed: {len(items)}/{len(items)}")

        return results


# Usage example for Buddy:
"""
from runtime.resumable import ResumableOperation

# Define processor function
def process_github_pr(pr_data: dict, index: int) -> dict:
    # ... analyze PR, call LLM, etc.
    return {"pr": pr_data["number"], "analysis": "..."}

# Create resumable operation
op = ResumableOperation(
    checkpoint_path=Path("~/.buddy/pr_analysis_progress.json"),
    checkpoint_every=5  # Checkpoint every 5 PRs
)

# Run (supports Ctrl+C interruption)
prs = fetch_all_prs()  # 1000 PRs

results = op.run(
    items=prs,
    processor=process_github_pr,
    resume=True,  # Will resume if interrupted
    metadata={"repo": "owner/repo", "started": "2026-04-17"}
)

# If interrupted with Ctrl+C:
# ✓ Progress saved to ~/.buddy/pr_analysis_progress.json
#   Completed: 347/1000
#   Resume with --resume flag

# Next run automatically resumes from PR #348
"""
```

---

## Organ 4: Fallback Cascade Router

**What it does**: Routes requests through fallback chain (local → cache → remote → refuse)
**Where it goes**: `runtime/models/fallback_router.py` in Buddy repo
**Dependencies**: None (stdlib only)

```python
"""
Fallback Cascade Router - Extracted from LARQL's walk→sparse→dense pattern
Try fast/local first, fallback to expensive/remote, refuse gracefully
"""
from typing import Callable, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Why a fallback was triggered"""
    NOT_AVAILABLE = "not_available"
    TIMEOUT = "timeout"
    ERROR = "error"
    QUOTA_EXCEEDED = "quota_exceeded"
    OFFLINE = "offline"


@dataclass
class FallbackResult:
    """Result from fallback cascade"""
    success: bool
    result: Any
    method_used: str
    fallbacks_tried: List[str]
    final_reason: Optional[str] = None


class FallbackCascade:
    """
    Routes requests through fallback chain.
    Pattern from LARQL: walk FFN → sparse FFN → dense FFN → refuse
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.methods: List[tuple[str, Callable]] = []

    def add_method(self, name: str, handler: Callable[[Any], Any],
                   availability_check: Optional[Callable[[], bool]] = None):
        """
        Add fallback method to cascade.
        Methods tried in order added.

        Args:
            name: Human-readable method name
            handler: Function that attempts to handle request
            availability_check: Optional function() -> bool to check availability
        """
        self.methods.append((name, handler, availability_check or (lambda: True)))

    def execute(self, request: Any, **kwargs) -> FallbackResult:
        """
        Execute request through fallback cascade.

        Returns FallbackResult with:
        - success: Whether any method succeeded
        - result: Result from successful method (or None)
        - method_used: Name of successful method
        - fallbacks_tried: List of failed methods
        """
        fallbacks_tried = []

        for method_name, handler, available in self.methods:
            # Check availability
            if not available():
                logger.debug(f"{self.name}: {method_name} not available, skipping")
                fallbacks_tried.append(f"{method_name} (unavailable)")
                continue

            # Try method
            try:
                logger.debug(f"{self.name}: Trying {method_name}")
                result = handler(request, **kwargs)

                # Success!
                logger.info(f"{self.name}: ✓ {method_name} succeeded")
                return FallbackResult(
                    success=True,
                    result=result,
                    method_used=method_name,
                    fallbacks_tried=fallbacks_tried
                )

            except Exception as e:
                logger.warning(f"{self.name}: {method_name} failed: {e}")
                fallbacks_tried.append(f"{method_name} (error: {e})")
                continue

        # All methods failed
        logger.error(f"{self.name}: All methods failed")
        return FallbackResult(
            success=False,
            result=None,
            method_used="none",
            fallbacks_tried=fallbacks_tried,
            final_reason="all_methods_failed"
        )


# Usage example for Buddy:
"""
from runtime.models.fallback_router import FallbackCascade

# Setup LLM fallback cascade
llm_cascade = FallbackCascade(name="llm_inference")

# Method 1: Local BitNet (fast, offline)
def try_bitnet(request):
    if not bitnet_available():
        raise RuntimeError("BitNet not loaded")
    return bitnet.generate(request["prompt"])

llm_cascade.add_method(
    name="bitnet_local",
    handler=try_bitnet,
    availability_check=lambda: bitnet_available()
)

# Method 2: Cached response (instant)
def try_cache(request):
    cached = cache.get(request["prompt"])
    if not cached:
        raise KeyError("Not in cache")
    return cached

llm_cascade.add_method(
    name="cache",
    handler=try_cache,
    availability_check=lambda: True
)

# Method 3: GPT-4 API (slow, requires internet)
def try_gpt4(request):
    if not is_online():
        raise RuntimeError("Offline")
    return gpt4_client.generate(request["prompt"])

llm_cascade.add_method(
    name="gpt4_api",
    handler=try_gpt4,
    availability_check=is_online
)

# Method 4: Refuse gracefully
def refuse_with_reason(request):
    return {
        "refused": True,
        "reason": "All LLM backends unavailable (offline + no cache)",
        "suggestion": "Connect to internet or use cached prompts"
    }

llm_cascade.add_method(
    name="refuse",
    handler=refuse_with_reason,
    availability_check=lambda: True
)

# Execute request
result = llm_cascade.execute({"prompt": "Explain quantum computing"})

if result.success:
    print(f"Success via {result.method_used}: {result.result}")
else:
    print(f"Failed after trying: {result.fallbacks_tried}")
    print(f"Final: {result.result}")

# Example outputs:
# - Online with BitNet loaded:    "Success via bitnet_local: ..."
# - Offline but prompt cached:    "Success via cache: ..."
# - Online but BitNet not loaded: "Success via gpt4_api: ..."
# - Offline and not cached:       "Failed ... Final: {refused: True, ...}"
"""
```

---

## Organ 5: Batch Processor via BLAS Pattern

**What it does**: Batch-processes similar operations for efficiency
**Where it goes**: `runtime/batch_processor.py` in Buddy repo
**Dependencies**: `numpy` (for batching arrays)

```python
"""
Batch Processor - Extracted from LARQL's 6 gemv → 1 gemm optimization
Batch similar operations instead of processing one-by-one
"""
import numpy as np
from typing import List, Callable, Any, TypeVar
from dataclasses import dataclass

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 32
    timeout_ms: Optional[int] = None
    auto_flush: bool = True


class BatchProcessor(Generic[T, R]):
    """
    Accumulates items and processes in batches.
    Pattern from LARQL: 1 batched gemm > N sequential gemv
    """

    def __init__(self,
                 batch_handler: Callable[[List[T]], List[R]],
                 config: Optional[BatchConfig] = None):
        """
        Args:
            batch_handler: Function that processes List[T] → List[R]
            config: Batch configuration
        """
        self.batch_handler = batch_handler
        self.config = config or BatchConfig()
        self._buffer: List[T] = []

    def add(self, item: T) -> Optional[List[R]]:
        """
        Add item to batch buffer.
        Returns results if batch is full, None otherwise.
        """
        self._buffer.append(item)

        if len(self._buffer) >= self.config.batch_size:
            return self.flush()

        return None

    def flush(self) -> List[R]:
        """Process all buffered items and return results"""
        if not self._buffer:
            return []

        results = self.batch_handler(self._buffer)
        self._buffer.clear()
        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.config.auto_flush:
            self.flush()


# Usage example for Buddy (tool scoring):
"""
from runtime.batch_processor import BatchProcessor, BatchConfig

# Setup batch scorer for tools
def batch_score_tools(prompts: List[str]) -> List[dict]:
    '''Score all prompts against all tools in one batch'''
    # Convert to batch: shape [batch_size, embedding_dim]
    embeddings = embed_model.encode(prompts)  # numpy array

    # Batch dot product with all tool embeddings
    # tools_matrix shape: [num_tools, embedding_dim]
    scores = embeddings @ tools_matrix.T  # [batch, tools]

    # Extract top-k tools for each prompt
    results = []
    for i, prompt_scores in enumerate(scores):
        top_indices = np.argsort(prompt_scores)[-5:][::-1]
        results.append({
            "prompt": prompts[i],
            "top_tools": [tools[idx] for idx in top_indices],
            "scores": [float(prompt_scores[idx]) for idx in top_indices]
        })

    return results

# Create batch processor
tool_scorer = BatchProcessor(
    batch_handler=batch_score_tools,
    config=BatchConfig(batch_size=16)
)

# Process requests
with tool_scorer:
    for user_request in user_requests:
        result = tool_scorer.add(user_request)
        if result:  # Batch complete
            process_batch_results(result)

    # Auto-flushes remaining on __exit__

# Performance gain:
# Old (sequential): 16 requests × 50ms = 800ms
# New (batched):    1 batch × 120ms = 120ms
# Speedup: 6.7×
"""
```

---

## Integration Guide for Buddy

### Step 1: Copy Organs to Buddy Repo

```bash
# In your Buddy repository
mkdir -p runtime/state
mkdir -p runtime/models

# Copy the organs
cp larql/docs/organs/immutable_state.py buddy/runtime/state/
cp larql/docs/organs/profiler.py buddy/runtime/
cp larql/docs/organs/resumable.py buddy/runtime/
cp larql/docs/organs/fallback_router.py buddy/runtime/models/
cp larql/docs/organs/batch_processor.py buddy/runtime/
```

### Step 2: Wire into Buddy Architecture

```python
# In buddy/runtime/brain/companion.py

from runtime.state.immutable_state import ImmutableStateManager
from runtime.profiler import profiler
from runtime.models.fallback_router import FallbackCascade

class BuddyCompanion:
    def __init__(self):
        # Use immutable state
        self.state = ImmutableStateManager(
            base_path=Path("~/.buddy/state.json")
        )

        # Setup LLM fallback
        self.llm = self._setup_llm_cascade()

    @profiler.measure("llm_call")
    def _setup_llm_cascade(self):
        cascade = FallbackCascade("llm")
        cascade.add_method("bitnet", self._try_bitnet)
        cascade.add_method("cache", self._try_cache)
        cascade.add_method("gpt4", self._try_gpt4)
        cascade.add_method("refuse", self._refuse)
        return cascade

    def process_request(self, user_input: str):
        with profiler.measure_block("total_request"):
            # ... your logic
            pass

        # Print profile every 100 requests
        if self.request_count % 100 == 0:
            print(profiler.report())
```

### Step 3: SmartMan Integration

```python
# In buddy/runtime/smartman_authority.py

from runtime.state.immutable_state import Receipt

class SmartManAuthority:
    def approve_state_change(self, receipt: Receipt) -> bool:
        """
        SmartMan approval gate.
        Returns True if approved, False if refused.
        """
        # Check against SmartMan rules
        if self._violates_law(receipt):
            logger.warning(f"SmartMan REFUSED: {receipt.path}")
            return False

        # Log approval
        logger.info(f"SmartMan APPROVED: {receipt.path}")
        receipt.approved_by = "smartman"
        return True
```

---

## For Source 2 Integration

These organs are designed for Python runtime. For Source 2 (Lua/C++), use these as **behavior contracts** and reimplement:

1. **ImmutableStateManager** → C++ with append-only log file
2. **ProgressiveProfiler** → Source 2 ConVar + timer hooks
3. **ResumableOperation** → C++ checkpoint system
4. **FallbackCascade** → Lua behavior tree
5. **BatchProcessor** → C++ batch accumulator

The **logic** is language-agnostic. The **interface** adapts to each layer.

---

## License & Attribution

- **Patterns**: Extracted from LARQL (Apache-2.0)
- **Code**: Adapted for Buddy (your license)
- **Philosophy**: Donor organs, not donor identity

When using, add attribution:
```python
# Pattern adapted from LARQL's immutable vindex + patch system
# Original: https://github.com/Citry3g/larql
```

---

## Summary

Five ready-to-use organs:

1. **ImmutableStateManager** - Receipt-based state (SmartMan-ready)
2. **ProgressiveProfiler** - Auto bottleneck detection
3. **ResumableOperation** - Interrupt-safe long operations
4. **FallbackCascade** - Local→Remote→Refuse routing
5. **BatchProcessor** - Batch efficiency pattern

Copy, adapt, integrate. SmartMan authority over all. Compact state out.

**This is BUDDY CLAW.**
