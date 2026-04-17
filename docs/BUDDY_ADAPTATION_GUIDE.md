# LARQL → Buddy: Extractable Design Patterns

**Purpose**: Document LARQL's architectural patterns and optimization strategies that can be adapted into Buddy's system design.

**Philosophy**: Extract the organs (logic, patterns, control flow) — not the identity. Buddy stays Buddy; LARQL's principles inform it.

---

## Core Architectural Patterns Worth Extracting

### 1. **Three-Tier Permission Model** (SmartMan-Compatible)

**LARQL Pattern**:
```
browse   → Read-only queries (DESCRIBE, WALK, SELECT)
inference → Requires model weights (INFER, TRACE)
all       → Full mutation authority (INSERT, DELETE, COMPILE)
```

**Buddy Adaptation**:
```
witness    → Observer mode (read state, no proposals)
operator   → Buddy layer (propose actions, route tools)
authority  → SmartMan layer (approve/refuse mutations)
```

**Key Insight**: Gate capabilities by extraction level, not by feature flags. This prevents privilege escalation and makes the permission model auditable.

**Apply To**:
- `runtime/permission_roundtrip.py` - Permission verification before action
- `runtime/brain/sentinel.py` - Mode detection and enforcement
- `runtime/state/posture_summary.py` - Current capability level

---

### 2. **Immutable Base + Patch Overlay** (Receipt-Compatible)

**LARQL Pattern**:
```
Base vindex:     Read-only, never modified, mmap'd
Patch (.vlp):    JSON overlay, stackable, reversible
PatchedVindex:   Runtime composition, no base mutation
```

**Buddy Adaptation**:
```
Base state:      SmartMan-approved canonical truth
Receipt (.rcpt): JSON record of proposed change
Active state:    Runtime view with pending receipts applied
```

**Key Insight**: Never mutate the canonical source. All changes flow through an append-only log that can be audited, reversed, or reapplied.

**Apply To**:
- `runtime/state/compact_state.py` - Immutable base state
- `runtime/receipts/patch_log.py` - Append-only receipt chain
- `runtime/session_registry.py` - Session-local overlays

**Code Pattern**:
```python
class CompactState:
    def __init__(self, base_path: Path):
        self.base = self._load_immutable_base(base_path)  # mmap or frozen dict
        self.receipts: List[Receipt] = []

    def apply_receipt(self, receipt: Receipt) -> None:
        """Add receipt to overlay - never modifies base"""
        self.receipts.append(receipt)

    def current_view(self) -> Dict:
        """Compute current state = base + all receipts"""
        state = self.base.copy()
        for r in self.receipts:
            state = r.apply_to(state)
        return state
```

---

### 3. **Zero-Copy Mmap-First** (Efficiency Pattern)

**LARQL Pattern**:
```
Gate vectors:  3.3GB mmap'd f32 file, OS-managed pages
Down weights:  Feature-major layout, zero-copy BLAS
Walk FFN:      Direct pointer + read, no allocation
```

**Buddy Adaptation**:
```
Knowledge base:  Large reference data mmap'd (Wikipedia, docs)
BitNet weights:  Model weights mmap'd for CPU inference
Session logs:    Append-only mmap file, no rewrites
```

**Key Insight**: For data >100MB, mmap beats load-into-RAM. The OS manages paging better than you can. Use feature-major layout for sequential access patterns.

**Apply To**:
- `runtime/models/bitnet_adapter.py` - BitNet weight loading
- `runtime/brain/knowledge_cache.py` - Reference data access
- `runtime/session_log_writer.py` - Log appending

**Code Pattern**:
```python
import mmap
import numpy as np

class MmapKnowledgeBase:
    def __init__(self, path: Path):
        self.file = open(path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        # Zero-copy numpy view
        self.vectors = np.frombuffer(self.mmap, dtype=np.float32)

    def query(self, offset: int, count: int) -> np.ndarray:
        """Zero-copy slice - no allocation"""
        return self.vectors[offset:offset+count]
```

---

### 4. **Batch KNN via BLAS** (Performance Pattern)

**LARQL Pattern**:
```
Old:  6 sequential gemv calls per layer (slow)
New:  1 batched gemm per layer (5× faster)
```

**Buddy Adaptation**:
```
Tool routing:    Batch-score all tools once, select top-K
Memory search:   Single FAISS query for all candidates
Model dispatch:  Batch prompts to BitNet, not one-by-one
```

**Key Insight**: Batching through BLAS/SIMD beats sequential loops. One matrix operation is faster than N vector operations.

**Apply To**:
- `runtime/models/model_router.py` - Batch tool scoring
- `runtime/brain/planner_light.py` - Batch candidate evaluation
- `runtime/bridge_packets.py` - Batch packet processing

---

### 5. **Progressive Profiling** (Bottleneck Identification)

**LARQL Pattern**:
```
Component         Time    % of total    Action
──────────────────────────────────────────────
Logits            221ms   41%           ← Target first
FFN               206ms   38%           ✓ Solved (walk)
Attention         84ms    16%           ← Target second
Overhead          7ms     1%            ✓ Clean
```

**Buddy Adaptation**:
```
Component         Time    % of total    Action
──────────────────────────────────────────────
Tool exec         ???ms   ??%           Profile first
LLM call          ???ms   ??%           Batch/cache?
State sync        ???ms   ??%           Reduce writes?
```

**Key Insight**: Profile before optimizing. The bottleneck is rarely where you think. Fix the top 2 items, re-profile, repeat.

**Apply To**:
- `runtime/profiler.py` - New module for timing
- `runtime/brain/companion.py` - Instrument critical path
- All major operations - Add timing decorators

**Code Pattern**:
```python
import time
from functools import wraps

class Profiler:
    def __init__(self):
        self.timings = {}

    def measure(self, name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                self.timings.setdefault(name, []).append(elapsed)
                return result
            return wrapper
        return decorator

    def report(self):
        total = sum(sum(times) for times in self.timings.values())
        for name, times in sorted(self.timings.items(),
                                   key=lambda x: sum(x[1]), reverse=True):
            component_total = sum(times)
            pct = 100 * component_total / total
            print(f"{name:30} {component_total*1000:6.1f}ms  {pct:5.1f}%")
```

---

### 6. **Walk-Only Mode** (Resource Optimization)

**LARQL Pattern**:
```
Full mode:   16.6GB (all weights loaded)
Walk mode:   3.5GB (only attention + embeddings)
Savings:     13GB (78% reduction)
```

**Buddy Adaptation**:
```
Full mode:   All models loaded (BitNet + GPT-4 + embeddings)
Light mode:  Only BitNet local (offline capability)
Witness:     No models (read-only state observer)
```

**Key Insight**: Load only what you need for the current posture. Lazy-load the rest on demand.

**Apply To**:
- `runtime/models/model_router.py` - Load based on mode
- `runtime/brain/companion.py` - Posture-aware initialization
- `runtime/models/fallback_matrix.py` - Fallback when resources limited

---

### 7. **Template Cache** (Context Reuse)

**LARQL Pattern**:
```
Observation: 99% of attention heads are fixed for templates
Method:      Cache attention pattern for "The capital of {X} is"
Result:      84ms → 5ms (only recompute entity-specific heads)
```

**Buddy Adaptation**:
```
Observation: Most tool calls follow patterns
Method:      Cache routing decision for common workflows
Result:      "Search web for {X}" → cached route to browser tool
```

**Key Insight**: Don't recompute what doesn't change. Template patterns repeat; cache the fixed parts.

**Apply To**:
- `runtime/brain/planner_light.py` - Cache common plans
- `runtime/models/model_router.py` - Cache routing decisions
- `runtime/bridge_packets.py` - Template-based packet generation

---

### 8. **Training-Free Mutation** (Knowledge Injection)

**LARQL Pattern**:
```
Method:  Multi-layer constellation (8 layers × small alpha)
Result:  94.6% new fact, 60% preservation (no training!)
Time:    30 seconds (1 forward pass + 8 feature writes)
```

**Buddy Adaptation**:
```
Method:  Session-local memory injection (no base retrain)
Result:  New facts available in current session
Persist: Via receipts, not model weights
```

**Key Insight**: Don't retrain for new facts. Inject them at runtime via overlays. Training is for broad capabilities; runtime injection is for specific knowledge.

**Apply To**:
- `runtime/brain/memory_inject.py` - New module
- `runtime/receipts/knowledge_receipt.py` - Receipt type for facts
- `runtime/state/workspace_state.py` - Session knowledge overlay

---

### 9. **Session Continuation** (Resume Logic)

**LARQL Pattern**:
```
Extract:     --resume flag, checkpoint every N items
Interrupt:   Ctrl+C saves progress.json
Resume:      Reads progress.json, skips completed work
```

**Buddy Adaptation**:
```
Session:     Auto-checkpoint every N actions
Interrupt:   Ctrl+C saves session_state.json
Resume:      "Continue where I left off" loads state
```

**Key Insight**: Every long operation needs resume capability. Checkpoint frequently, detect interruption, resume cleanly.

**Apply To**:
- `runtime/session_registry.py` - Auto-checkpoint logic
- `runtime/brain/companion.py` - Interrupt handlers
- All long operations - Add progress tracking

**Code Pattern**:
```python
class ResumableOperation:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint = checkpoint_path
        self.progress = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict:
        if self.checkpoint.exists():
            return json.loads(self.checkpoint.read_text())
        return {"completed": [], "next_index": 0}

    def run(self, items: List):
        start_idx = self.progress["next_index"]
        try:
            for i in range(start_idx, len(items)):
                self.process(items[i])
                self.progress["completed"].append(i)
                self.progress["next_index"] = i + 1
                if i % 100 == 0:  # Checkpoint every 100
                    self._save_checkpoint()
        except KeyboardInterrupt:
            self._save_checkpoint()
            raise

    def _save_checkpoint(self):
        self.checkpoint.write_text(json.dumps(self.progress))
```

---

### 10. **Declarative Build** (Vindexfile → Buddyfile)

**LARQL Pattern**:
```dockerfile
FROM hf://base-model-vindex
PATCH hf://knowledge-patch@2.1.0
PATCH ./local-facts.vlp
INSERT ("Acme", "headquarters", "London")
EXPOSE browse inference
```

**Buddy Adaptation**:
```yaml
# Buddyfile.yml
base: "smartman-approved-state-v1.2"
receipts:
  - source: "github://shared/gdrive-auth@1.0"
  - source: "./local-preferences.rcpt"
  - inline:
      action: "configure_tool"
      tool: "browser"
      setting: "default_headless: true"
posture: "operator"  # witness | operator | authority
```

**Key Insight**: Configuration as code. Reproducible builds. Version everything.

**Apply To**:
- `Buddyfile.yml` - New file at repo root
- `runtime/config_loader.py` - Parse and apply Buddyfile
- `runtime/session_registry.py` - Session from Buddyfile

---

## Control Flow Patterns

### A. **Agent Role Split**

**LARQL**:
```
Planner:    Analyze query → determine operation
Executor:   Execute LQL statement → return results
Validator:  Check results → verify correctness
```

**Buddy**:
```
Sentinel:   Observe state → detect violations
Planner:    Analyze request → propose actions
Companion:  Execute tools → return results
SmartMan:   Approve/refuse → enforce law
```

---

### B. **Fallback Cascade**

**LARQL**:
```
1. Try walk FFN (mmap, fastest)
2. Fallback to sparse FFN (if no mmap)
3. Fallback to dense FFN (if no vindex)
```

**Buddy**:
```
1. Try BitNet local (offline, fast)
2. Fallback to cached response (if seen before)
3. Fallback to GPT-4 (if online)
4. Refuse with reason (if offline + no cache)
```

**Apply To**: `runtime/models/fallback_matrix.py`

---

### C. **Packet-Based Communication**

**LARQL**:
```python
# Query → Executor → Result
query_packet = {"type": "DESCRIBE", "entity": "France"}
result_packet = {"edges": [...], "latency_ms": 33}
```

**Buddy**:
```python
# Proposal → SmartMan → Receipt
proposal_packet = {"type": "file_write", "path": "...", "content": "..."}
receipt_packet = {"status": "approved", "hash": "abc123", "timestamp": ...}
```

**Apply To**: `runtime/bridge_packets.py`

---

## Memory Management Patterns

### 1. **Compact State**

**Principle**: State should be small enough to serialize, large enough to resume

**LARQL**: index.json (5KB) contains everything to reload vindex
**Buddy**: session_state.json contains posture + active receipts + tool configs

### 2. **Lazy Loading**

**Principle**: Don't load until accessed

**LARQL**: Attention weights load on first INFER, not on USE
**Buddy**: Models load when first needed, not on Buddy startup

### 3. **Demand Paging**

**Principle**: OS manages memory better than you

**LARQL**: Mmap 3.6GB file, RSS stays <500MB until accessed
**Buddy**: Mmap knowledge base, let OS decide what stays in RAM

---

## Testing Patterns

### 1. **Boundary Sweep**

**LARQL**: Test walk FFN at all 34 layer boundaries → 100% match
**Buddy**: Test permission gates at all transitions (witness→operator→authority) → 100% enforcement

### 2. **Regression Benchmarks**

**LARQL**: 5 standard prompts, track accuracy and latency
**Buddy**: 10 standard workflows, track success rate and receipt count

### 3. **Stress Testing**

**LARQL**: 10,000 rapid queries, check for memory leaks
**Buddy**: 1,000 rapid tool calls, check for permission bypass

---

## What NOT to Extract

❌ Don't extract:
- LARQL's LQL language syntax (Buddy has its own commands)
- LARQL's vindex file format (Buddy has compact state)
- LARQL's HuggingFace integration (Buddy uses GitHub/GDrive)
- LARQL's model architecture (Buddy uses BitNet/GPT-4)

✅ Do extract:
- Permission gating philosophy
- Immutable base + overlay pattern
- Zero-copy mmap strategy
- Batch processing through BLAS
- Profile-first optimization
- Template caching
- Fallback cascades
- Packet-based communication
- Resume-capable operations
- Declarative configuration

---

## Implementation Priorities for Buddy

**Week 1-2**: Core patterns
1. Immutable base + receipt overlay (`runtime/state/`)
2. Three-tier permissions (`runtime/permission_roundtrip.py`)
3. Profiler (`runtime/profiler.py`)

**Week 3-4**: Performance
4. Mmap for BitNet weights (`runtime/models/bitnet_adapter.py`)
5. Batch tool routing (`runtime/models/model_router.py`)
6. Template cache (`runtime/brain/planner_light.py`)

**Week 5-6**: Resilience
7. Resume capability (`runtime/session_registry.py`)
8. Fallback matrix (`runtime/models/fallback_matrix.py`)
9. Buddyfile config (`Buddyfile.yml` + loader)

---

## Success Metrics

Track these as you adapt patterns:

| Metric | Baseline | Target | LARQL Achievement |
|--------|----------|--------|-------------------|
| **Startup time** | ??? | <2s | LARQL: <10ms (mmap) |
| **Memory footprint** | ??? | <500MB | LARQL: 3.5GB→500MB (walk-only) |
| **Tool call latency** | ??? | <100ms | LARQL: 33ms (browse) |
| **Permission violations** | ??? | 0 | LARQL: 0 (level-gated) |
| **Receipt chain integrity** | ??? | 100% | LARQL: 100% (immutable base) |

---

## License & Attribution

**LARQL**: Apache-2.0 (source repository)
**This Document**: Adaptation guide for Buddy (patterns, not code)
**Philosophy**: Extract organs (logic), not identity (brand)

When implementing, cite the pattern source:
```python
# Pattern adapted from LARQL's immutable base + patch overlay
# See: https://github.com/Citry3g/larql/blob/main/docs/vindex-operations-spec.md
class CompactState:
    ...
```

---

## Questions to Answer Before Implementation

1. **Permission Model**: Does Buddy need three tiers or more?
2. **Receipt Format**: JSON? Binary? Encrypted?
3. **State Size**: How big is Buddy's compact state?
4. **Offline Requirements**: Must BitNet work 100% offline?
5. **Template Patterns**: What are Buddy's top 10 repeated workflows?
6. **Fallback Policy**: What happens when all models unavailable?
7. **Checkpoint Frequency**: Every N actions or every M seconds?

Answer these, then start implementing patterns that fit Buddy's law.

---

**Remember**: You are not building LARQL for Buddy. You are building Buddy, informed by LARQL's proven patterns. The model is the database → The state is the law. SmartMan always wins.
