# LARQL Donor Limits and Graft Rules

**Purpose**: Define what is reusable from LARQL for Buddy, what is pattern-only, and what must NOT be transplanted.

**Repo Law**:
- `Citry3g/larql` = donor-pattern / optimization / local knowledge assimilation harvest
- `Citry3g/smartman` = Buddy / SmartMan implementation root
- **SmartMan** = canonical mutation authority
- **Buddy** = operator layer
- **Shells** = compact-state renderers only

---

## What IS Reusable (Graft These)

### 1. Local Knowledge Assimilation Patterns

**Pattern**: Immutable base + append-only overlay
**LARQL**: Base vindex (readonly) + patches (.vlp files)
**Buddy Landing**:
- `runtime/patch_engine.py` - Patch overlay logic
- `runtime/receipt_engine.py` - Receipt chain validation
- `runtime/state/compact_state.py` - Immutable base state

**Graft Rule**: Extract the **overlay pattern**, not the vindex format. Buddy uses JSON receipts, not .vlp files.

---

### 2. Walk-Only / Low-Memory Mode

**Pattern**: Load only what's needed for current operation
**LARQL**: Walk mode (3.5GB) vs Full mode (16.6GB)
**Buddy Landing**:
- `runtime/models/model_router.py` - Posture-aware model loading
- `runtime/models/fallback_matrix.py` - Resource-aware fallback

**Graft Rule**: Adapt the **resource gating strategy**. Buddy postures (witness/operator/authority) determine what loads.

---

### 3. Template Cache / Profiling

**Pattern**: Cache fixed patterns, profile before optimizing
**LARQL**: Attention template cache (99% fixed heads), progressive profiler
**Buddy Landing**:
- `runtime/brain/workflow_cache.py` - Cache common tool workflows
- `runtime/profiler.py` - Lightweight bottleneck detection

**Graft Rule**: Cache **tool routing decisions**, not attention patterns. Profile Buddy's critical path (tool exec, LLM calls, state sync).

---

### 4. Session Continuation / Resumable Operations

**Pattern**: Checkpoint progress, resume after interruption
**LARQL**: --resume flag, progress.json checkpoints
**Buddy Landing**:
- `runtime/session_registry.py` - Session state management
- All long operations - Add checkpoint capability

**Graft Rule**: Every Buddy operation >10s must support resume. Use same checkpoint pattern.

---

### 5. Incremental Updates / Corpus Ingestion

**Pattern**: Streaming ingestion without full reload
**LARQL**: Extract with --resume, incremental probe updates
**Buddy Landing**:
- `runtime/source_registry.py` - Track ingested sources
- `runtime/parser_router.py` - Parse documents incrementally

**Graft Rule**: Buddy ingests local docs/code/notes incrementally. Never require full rebuild.

---

### 6. Provenance / Confidence Scoring

**Pattern**: Track source of knowledge, assign confidence
**LARQL**: Probe confidence scores, feature labels with sources
**Buddy Landing**:
- `runtime/brain/source_judge.py` - Assign trust levels to sources
- Receipt metadata - Track provenance chain

**Graft Rule**: Every Buddy fact includes source + confidence. SmartMan uses this for approval decisions.

---

### 7. Batch Processing Efficiency

**Pattern**: Batch similar operations through SIMD/BLAS
**LARQL**: 6 sequential gemv → 1 batched gemm (5× speedup)
**Buddy Landing**:
- `runtime/batch_processor.py` - Batch tool scoring
- `runtime/models/model_router.py` - Batch prompt processing

**Graft Rule**: Batch operations where possible. Profile to confirm gains.

---

### 8. Local-First Build Directives

**Pattern**: Declarative configuration for reproducible builds
**LARQL**: Vindexfile (FROM, PATCH, INSERT, EXPOSE)
**Buddy Landing**:
- `Buddyfile.yml` - Declarative Buddy configuration
- `runtime/config_loader.py` - Parse and apply Buddyfile

**Graft Rule**: Buddyfile syntax inspired by Vindexfile, but Buddy-specific semantics.

---

## What Is Pattern-Only (Study, Don't Copy)

### 1. LQL Language Syntax
**Why Pattern-Only**: LARQL-specific query language. Buddy has its own command structure.
**Study For**: How to design a domain-specific language with clear error messages.

### 2. Vindex File Format
**Why Pattern-Only**: Specific to transformer model weights. Buddy doesn't store model weights.
**Study For**: Mmap-first storage strategy, feature-major layout benefits.

### 3. FFN Graph Layer
**Why Pattern-Only**: Neural network internals. Buddy uses BitNet/GPT-4, not custom FFN.
**Study For**: How to optimize hot paths through profiling + iteration.

### 4. Gate KNN via BLAS
**Why Pattern-Only**: Model-specific operation. Buddy doesn't do gate searches.
**Study For**: How to use BLAS for batch efficiency (apply to tool routing instead).

---

## What Must NOT Be Transplanted (Hard Limits)

### ❌ 1. LARQL Product Identity

**Do NOT**:
- Call Buddy "LARQL for agents"
- Use LARQL branding/naming
- Claim Buddy is built on LARQL codebase
- Import LARQL as a dependency

**Reason**: Separate projects, separate identities. Buddy is Buddy.

---

### ❌ 2. Model-as-Database Philosophy

**Do NOT**:
- Treat LLM weights as queryable database
- Implement vindex extraction for Buddy
- Use model weight patches as primary storage

**Reason**: LARQL's core claim (model IS database) doesn't apply to Buddy. Buddy uses models as tools, not truth stores.

---

### ❌ 3. HuggingFace Integration

**Do NOT**:
- Integrate with HuggingFace Hub
- Download models from HF
- Use HF tokenizers/transformers library

**Reason**: Buddy uses BitNet (local) and GPT-4 (API), not HuggingFace models. Different ecosystem.

---

### ❌ 4. Training-Free Knowledge Insertion via Weight Modification

**Do NOT**:
- Modify BitNet weights directly
- Implement constellation insert
- Store knowledge in model weights

**Reason**: Buddy stores knowledge in receipts/state, not model weights. Different architecture.

---

### ❌ 5. Rust Implementation

**Do NOT**:
- Rewrite Buddy in Rust
- Use Cargo/Rust toolchain
- Import LARQL's Rust crates

**Reason**: Buddy is Python (runtime) + Lua (Source 2 shell). Rust is LARQL-specific.

---

### ❌ 6. Shell-as-Truth Behavior

**Do NOT**:
- Let Source 2 shell modify state directly
- Bypass SmartMan from visualization layer
- Store canonical state in game engine

**Reason**: Source 2 is display only. SmartMan is always authority. Shells never mutate truth.

---

## Exact Buddy Landing Zones (SmartMan Repo)

When grafting LARQL patterns into Buddy, use these exact file paths:

```
smartman/                                    (Buddy implementation root)
├── runtime/
│   ├── state/
│   │   └── compact_state.py                 ← Immutable base + receipt overlay
│   ├── source_registry.py                   ← Local corpus tracking
│   ├── parser_router.py                     ← Document ingestion
│   ├── patch_engine.py                      ← Patch overlay logic
│   ├── receipt_engine.py                    ← Receipt validation
│   ├── session_registry.py                  ← Session continuation
│   ├── profiler.py                          ← Bottleneck detection
│   ├── batch_processor.py                   ← Batch efficiency
│   ├── models/
│   │   ├── model_router.py                  ← Posture-aware routing
│   │   ├── fallback_matrix.py               ← Local → Remote fallback
│   │   └── bitnet_adapter.py                ← BitNet local inference
│   └── brain/
│       ├── source_judge.py                  ← Provenance/confidence
│       └── workflow_cache.py                ← Template caching
├── Buddyfile.yml                            ← Declarative config
└── runtime/config_loader.py                 ← Buddyfile parser
```

**Rule**: All LARQL patterns land in `smartman/runtime/`, never in LARQL repo itself.

---

## Graft Process (How to Extract)

### Step 1: Identify Pattern
Read LARQL docs, identify useful pattern (e.g., "immutable base + overlay").

### Step 2: Extract Logic
Write down the **behavior contract**, not the code:
```
Pattern: Immutable base + overlay
Behavior:
1. Base state is readonly (never modified)
2. Changes append to log (receipts.jsonl)
3. Current state = base + all receipts
4. Receipts can be rolled back
5. Base can be compacted (bake receipts in)
```

### Step 3: Adapt to Buddy
Map to Buddy concepts:
- LARQL "vindex" → Buddy "compact state"
- LARQL "patches" → Buddy "receipts"
- LARQL "COMPILE" → Buddy "compact base"

### Step 4: Implement in SmartMan
Write Python code in correct landing zone (`runtime/state/compact_state.py`).

### Step 5: Wire to SmartMan Authority
Ensure all mutations go through SmartMan approval gate.

### Step 6: Test in Buddy Posture
Verify works in witness/operator/authority modes.

---

## Anti-Patterns (What Went Wrong Before)

### ❌ Anti-Pattern 1: Repo Identity Confusion
**Mistake**: Agent kept asking "is this the Buddy repo?"
**Correct**: Accept LARQL is donor lane, route implementation to SmartMan.

### ❌ Anti-Pattern 2: Implementing Buddy in LARQL
**Mistake**: Writing Buddy code inside LARQL repository.
**Correct**: LARQL produces **donor notes**, SmartMan implements.

### ❌ Anti-Pattern 3: Claiming Non-Existent PRs
**Mistake**: Saying "PR #55 created" when GitHub API failed.
**Correct**: Only claim PR exists if visible on GitHub.

### ❌ Anti-Pattern 4: Transplanting Identity
**Mistake**: Calling Buddy "LARQL-based" or using LARQL naming.
**Correct**: Buddy is Buddy. LARQL informed it, doesn't define it.

### ❌ Anti-Pattern 5: Widening Shells Before Runtime Ready
**Mistake**: Designing Source 2 visualization before Buddy runtime works.
**Correct**: Build runtime spine first, shells later.

---

## Success Criteria

A successful donor harvest PR contains:

✅ **Pattern documentation** (this file)
✅ **Optimization roadmap** (OPTIMIZATION_ROADMAP.md)
✅ **Adaptation guide** (BUDDY_ADAPTATION_GUIDE.md)
✅ **Extractable organs** (EXTRACTABLE_ORGANS.md)

✅ **Clear boundaries**: What to graft, what to study, what to avoid
✅ **Exact landing zones**: File paths in SmartMan repo
✅ **No implementation**: Docs only, no Buddy code in LARQL
✅ **No confusion**: LARQL ≠ Buddy, donor ≠ recipient

❌ **No Source 2 implementation** in LARQL
❌ **No SmartMan law changes** in LARQL
❌ **No private/user data** in docs
❌ **No claims about PRs** unless visible on GitHub

---

## The Disciplined Truth

From the agent transcript:

**What Actually Happened**:
1. Agent created branch locally
2. Agent committed files locally
3. Push to GitHub **failed** (403 auth error)
4. PR creation **failed** (403 forbidden)
5. Nothing was published to GitHub

**Therefore**: PR #55 is **not** a live repo fact from that transcript.

**Correct Action**: Create **fresh docs-only PR** linked to issue #23, let GitHub assign the real number.

---

## Recommended Next Actions

### For This PR (LARQL Donor Harvest)
1. ✅ Complete these 4 docs
2. ✅ Link to issue #23
3. ✅ Submit as docs-only PR
4. ✅ Let GitHub assign actual PR number
5. ✅ Get it merged

### For SmartMan Repo (Buddy Implementation)
1. Create landing zone files (`runtime/state/compact_state.py`, etc.)
2. Implement patterns from LARQL docs
3. Wire to SmartMan authority
4. Test in Buddy postures
5. Profile and optimize
6. Deploy

### For Source 2 (Shell Layer)
1. **Wait** until Buddy runtime is fast and real
2. Then design 3D visualization
3. Keep it display-only
4. Never bypass SmartMan

---

## License & Attribution

**LARQL Patterns**: Apache-2.0 (studied for patterns)
**This Document**: Donor harvest guide for Buddy
**Buddy Implementation**: Your license in SmartMan repo

When using patterns in SmartMan, cite source:
```python
# Pattern adapted from LARQL's immutable base + patch overlay
# See: https://github.com/Citry3g/larql/blob/main/docs/vindex-operations-spec.md
```

---

## The Bottom Line

**LARQL** = excellent donor for local knowledge patterns
**Buddy** = separate project that benefits from those patterns
**This PR** = clean harvest documentation, not implementation

**SmartMan always wins.** Buddy is the operator. Shells are renderers. Models are tools.

The donor surgery is documentation-only. The actual grafting happens in `Citry3g/smartman`.

---

**End of donor limits and graft rules.**
