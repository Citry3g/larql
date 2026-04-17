# Advanced Experimental Patterns from LARQL

**Purpose**: Document advanced experimental patterns from LARQL research that may inform future Buddy capabilities. These are **not ready for immediate grafting** but represent valuable research directions.

**Status**: Research-grade patterns. Study for concepts, not implementation.

**Repo Law**: LARQL = donor research lane. These patterns stay in LARQL experiments/. If proven valuable through further research, extract to Buddy following the graft rules in `LARQL_DONOR_LIMITS_AND_GRAFT_RULES.md`.

---

## Pattern 1: Multi-Layer Knowledge Insertion

**Experiment**: `experiments/04_constellation_insert/multilayer.py`
**Status**: Proven (94.6% accuracy, 20% degradation on neighbors)
**Maturity**: Production-ready pattern, documented in `training-free-insert.md`

### Core Pattern

Instead of inserting knowledge at a single layer (which breaks existing facts), spread the insertion across multiple layers with smaller contributions at each layer.

**Math**:
```
Single layer:   1 feature × alpha=5.0  →  strong effect, breaks neighbors
Multi-layer:    8 features × alpha=0.25  →  same total (2.0), selective effect
```

**Result**:
- New fact: "The capital of Atlantis is" → Poseidon (94.6%)
- Existing fact: "The capital of France is" → Paris (60.5%, down from 80.5%)
- Trade-off: 16L × 0.12 gives 78% new fact, only 14pts degradation

### Why This Matters for Buddy

**NOT APPLICABLE**: Buddy doesn't modify model weights or store knowledge in neural layers.

**Study for conceptual pattern**: Distributed weak signals accumulate selectively based on context. The same principle applies to:
- Tool scoring: Multiple weak signals (file type, recent use, context match) combine
- Confidence aggregation: Multiple sources contribute small confidence adjustments
- Posture transitions: Multiple weak triggers accumulate to posture change threshold

**Buddy Adaptation** (if applicable):
```python
class DistributedSignalAggregator:
    """Pattern from LARQL multi-layer insert: weak signals accumulate selectively"""

    def aggregate_tool_scores(self, tool_id: str, context: Dict) -> float:
        """Aggregate multiple weak signals instead of single strong score"""
        signals = [
            self._file_type_match(tool_id, context) * 0.15,
            self._recent_use_score(tool_id) * 0.15,
            self._context_similarity(tool_id, context) * 0.20,
            self._success_rate(tool_id) * 0.25,
            self._user_preference(tool_id) * 0.25,
        ]
        return sum(signals)  # Weak signals accumulate
```

**Landing Zone** (if implemented): `smartman/runtime/brain/distributed_scoring.py`

---

## Pattern 2: Syntax-Circuit Routing

**Experiment**: `experiments/05_syntax_circuit_routing/`
**Status**: Research-stage hypothesis
**Maturity**: Needs validation

### Core Hypothesis

Early syntactic features predict which attention circuits activate, analogous to how trigram types routed to MoE experts in GPT-OSS.

**Claim**:
```
Syntax features (L0-12) detect pattern
  ↓
Predict attention circuit activation (L13-26)
  ↓
Replace attention with cached routing table
  ↓
Result: 39× speedup (1.15ms vs 45ms)
```

**Success Criteria**:
- Sparsity > 0.8: Most syntax features map to ≤3 circuits
- Category separation: Different patterns route to different head clusters
- Labeled features match: `wn:synonym` → synonym circuit

### Why This Matters for Buddy

**NOT DIRECTLY APPLICABLE**: Buddy doesn't have attention circuits or syntax features.

**Study for routing pattern**: Early cheap signals predict expensive operations.

**Buddy Adaptation**:
```python
class CheapSignalRouter:
    """Pattern from syntax-circuit routing: cheap features predict expensive paths"""

    def __init__(self):
        # Build routing table: cheap_signal → expensive_operation
        self.routing_table = self._build_routing_table()

    def route_tool_call(self, user_message: str) -> str:
        """Use cheap NLP features to predict which tool to call"""
        # Cheap: keyword extraction, entity detection (1ms)
        features = self._extract_cheap_features(user_message)

        # Route via table lookup (0.01ms) instead of full LLM call (200ms)
        if self._can_route_from_table(features):
            return self.routing_table.lookup(features)

        # Fallback: full LLM routing (expensive but accurate)
        return self._llm_route(user_message)

    def _extract_cheap_features(self, text: str) -> List[str]:
        """Fast pattern matching: file extensions, keywords, entities"""
        return [
            self._has_file_path(text),
            self._has_code_block(text),
            self._has_url(text),
            self._has_date_time(text),
        ]
```

**Landing Zone** (if validated): `smartman/runtime/tools/fast_router.py`

---

## Pattern 3: Token-Level Solver Interception

**Experiment**: `experiments/07_wasm_compute/phase1_pipeline.py`
**Status**: Phase 1 complete (Python solvers)
**Maturity**: Proven concept, Phase 2-3 TBD

### Core Pattern

Monitor LLM output token stream during generation. When a computable expression is detected:
1. Pause generation
2. Parse expression
3. Dispatch to deterministic solver
4. Inject correct answer tokens
5. Resume generation

**Phases**:
- Phase 1 (done): Python solvers, token interception
- Phase 2 (research): Residual-level dispatch
- Phase 3 (conditional): WASM runtime in Rust

**Result**: Math benchmarks improve because symbolic solver guarantees correctness.

### Why This Matters for Buddy

**HIGHLY APPLICABLE**: Buddy already has tool system. This is tool-calling at token level.

**Buddy already does this** via tool use, but at the **message level** (full response), not **token level** (mid-generation).

**Study for**: Streaming tool calls during generation.

**Buddy Adaptation**:
```python
class StreamingToolIntercept:
    """Pattern from LARQL WASM compute: intercept generation, call tool, resume"""

    def generate_with_tools(self, prompt: str, tools: List[Tool]) -> str:
        """Generate with mid-stream tool interception"""
        output_buffer = ""
        stream_parser = StreamParser(tools)

        for token in self.llm.generate_stream(prompt):
            output_buffer += token

            # Check if we hit a tool pattern (e.g., "```python\n" or "$math{")
            tool_call = stream_parser.detect_tool_pattern(output_buffer)

            if tool_call:
                # Pause generation
                self.llm.pause()

                # Execute tool
                tool_result = self._execute_tool(tool_call)

                # Inject result tokens
                output_buffer += tool_result

                # Resume generation with injected context
                self.llm.resume(context=tool_result)

        return output_buffer
```

**Landing Zone** (if implemented): `smartman/runtime/brain/streaming_tools.py`

**Implementation Priority**: Low. Message-level tool calling is sufficient for Buddy v1.

---

## Pattern 4: Gradient Anatomy for Layer Specialization

**Experiment**: `experiments/06_backprop_insert/experiment.py`
**Status**: Research-stage
**Maturity**: Hypothesis testing

### Core Hypothesis

During training, does backprop naturally assign different learning tasks to different layer bands?

**Measurement**:
- Syntax examples → FFN weight displacement concentrated in L0-3 (syntax band)?
- Factual examples → FFN weight displacement concentrated in L4-7 (knowledge band)?
- Formatting examples → FFN weight displacement concentrated in L8-11 (output band)?

**Claim**: If true, then INSERT operations should respect these natural bands.

### Why This Matters for Buddy

**NOT APPLICABLE**: Buddy doesn't train models or analyze gradients.

**Study for layer strategy**: Different types of knowledge live in different "bands" (early/mid/late layers).

**Buddy has no equivalent** — Buddy doesn't have "layers" in this sense. But the concept of **specialization by depth** could map to:
- Tool selection: Quick filters (early) → Deep reasoning (late)
- Posture transitions: witness (surface) → operator (action) → authority (judgment)

**No immediate graft target.**

---

## Pattern 5: Residual Stream Clustering

**Experiment**: `experiments/05_syntax_circuit_routing/residual_clustering.py`
**Status**: Exploratory
**Maturity**: Research

### Core Pattern

Cluster residual vectors by semantic category. If clusters are tight and well-separated, the model has learned distinct representations for different knowledge types.

**Use Case**: Determine which layer band handles which type of knowledge by examining where clusters form.

### Why This Matters for Buddy

**NOT APPLICABLE**: Buddy doesn't analyze neural residuals.

**Study for**: How to validate that a system has learned distinct internal representations.

**Buddy equivalent**: Validate that SmartMan's approval decisions cluster by category (file edits vs API calls vs data access).

**No immediate graft target.**

---

## Pattern 6: Template Caching with Attention Analysis

**Experiment**: Mentioned in `walk-boundary-sweep.md` and `inference-engine.md`
**Status**: Documented pattern (99% fixed heads)
**Maturity**: Production-ready

### Core Pattern

Profile attention patterns. If 99% of attention heads produce identical patterns across prompts, cache the attention computation and replace with lookup.

**Result**:
```
Without cache: Attention 84ms + FFN 206ms = 290ms
With cache:    Attention ~0ms + FFN 206ms = 206ms
```

**Already documented** in `BUDDY_ADAPTATION_GUIDE.md` as "Template cache / profiling" pattern.

---

## Pattern 7: Walk-Only Mode (Resource-Aware Operation)

**Experiment**: `ffn-graph-layer.md` documents `InferenceModel::load_walk_only()`
**Status**: Production
**Maturity**: Implemented

### Core Pattern

Detect available resources and load only what's needed:
- Full mode: 16.6GB (attention + FFN weights)
- Walk-only mode: 5.5GB (attention + vindex, drop FFN weights)

**Result**: 10.7GB savings with zero accuracy loss.

**Already documented** in `BUDDY_ADAPTATION_GUIDE.md` as "Walk-only / low-memory mode" pattern.

---

## What Is Ready vs What Is Research

### ✅ Ready to Graft (Already Documented)

These patterns are proven and documented in the main donor extraction docs:

1. **Multi-layer distributed signals** → `BUDDY_ADAPTATION_GUIDE.md` (pattern adapted)
2. **Template caching** → `BUDDY_ADAPTATION_GUIDE.md` (workflow cache)
3. **Walk-only mode** → `BUDDY_ADAPTATION_GUIDE.md` (posture-aware loading)
4. **Immutable base + overlay** → `EXTRACTABLE_ORGANS.md` (ImmutableStateManager)
5. **Progressive profiling** → `EXTRACTABLE_ORGANS.md` (ProgressiveProfiler)
6. **Resumable operations** → `EXTRACTABLE_ORGANS.md` (ResumableOperation)

### 🔬 Research-Stage (Study, Don't Graft)

These patterns are experimental and need more validation:

1. **Syntax-circuit routing**: Hypothesis needs validation (sparsity, separation)
2. **Gradient anatomy**: Neural-network-specific, no Buddy equivalent
3. **Residual clustering**: Analysis technique, not operational pattern
4. **Token-level tool interception**: Interesting but complex, message-level sufficient for v1

### ⚠️ Not Applicable to Buddy

These patterns are LARQL-specific and should NOT be transplanted:

1. **Training-free knowledge insertion** (modifies model weights)
2. **Gate KNN via BLAS** (neural network internals)
3. **Attention head analysis** (transformer-specific)
4. **FFN band specialization** (layer-based architecture)

---

## Decision Tree: Should I Graft This Pattern?

```
Is the pattern proven in LARQL experiments?
  ├─ No → Keep in research lane, don't graft
  └─ Yes → Is it already documented in main donor docs?
      ├─ Yes → Use existing documentation
      └─ No → Does it require neural network internals?
          ├─ Yes → Study for concepts, don't graft code
          └─ No → Does Buddy have an equivalent concept?
              ├─ No → No graft target
              └─ Yes → Extract pattern, map to Buddy concept, land in SmartMan
```

---

## Experimental Pattern Maturity Ladder

**Level 0: Hypothesis** → Research question, no code
**Level 1: Exploratory** → Code exists, results unclear
**Level 2: Validated** → Results confirm hypothesis, not optimized
**Level 3: Production** → Optimized, documented, integrated into main codebase
**Level 4: Donor-ready** → Documented in extraction guides, clear Buddy mapping

| Pattern | LARQL Level | Buddy Applicability | Graft Status |
|---------|-------------|---------------------|--------------|
| Multi-layer insert | 3 (Production) | Distributed scoring | ✅ Documented |
| Template caching | 3 (Production) | Workflow cache | ✅ Documented |
| Walk-only mode | 3 (Production) | Posture-aware loading | ✅ Documented |
| Syntax-circuit routing | 1 (Exploratory) | Fast routing table | 🔬 Research |
| Token-level interception | 2 (Validated Phase 1) | Streaming tools | 🔬 Research |
| Gradient anatomy | 1 (Exploratory) | None | ❌ N/A |
| Residual clustering | 1 (Exploratory) | Validation method | 🔬 Research |

---

## How to Advance a Pattern from Research → Donor

### Step 1: Prove It in LARQL

Run experiments, collect results, validate hypothesis. Document in `experiments/XX_name/README.md`.

### Step 2: Integrate into LARQL Core

If proven valuable, move from `experiments/` to core codebase (e.g., `larql-inference`, `larql-vindex`).

### Step 3: Write Production Docs

Document in `docs/` (e.g., `training-free-insert.md`, `walk-boundary-sweep.md`).

### Step 4: Extract for Buddy

Add to donor extraction docs:
- `BUDDY_ADAPTATION_GUIDE.md` (pattern mapping)
- `EXTRACTABLE_ORGANS.md` (code modules)

### Step 5: Implement in SmartMan

Write Buddy implementation in `Citry3g/smartman` repository, following exact landing zones.

---

## Recommended Research Priorities

If LARQL research continues, prioritize patterns that:

1. **✅ Have clear Buddy equivalents** (e.g., routing, caching, resource awareness)
2. **✅ Generalize beyond neural networks** (applicable to any AI system)
3. **✅ Are measurable** (clear success criteria, quantitative results)
4. **❌ Avoid neural-network internals** (gradient analysis, attention mechanics)

**High-value research areas for Buddy**:
- Fast routing via cheap signals (syntax-circuit pattern, generalized)
- Streaming tool interception (token-level pattern, adapted to message level)
- Template caching for tool workflows (proven in attention, adapt to tools)
- Distributed confidence scoring (multi-layer insert pattern, adapt to sources)

**Low-value for Buddy**:
- Gradient anatomy (no training in Buddy)
- Residual analysis (no neural layers in Buddy)
- Model weight modification (SmartMan is authority, not model weights)

---

## Crosswalk: LARQL Experiments → Buddy Use Cases

| LARQL Experiment | Core Finding | Buddy Equivalent | Status |
|------------------|--------------|------------------|--------|
| 01 Gate Synthesis | Embedding ≠ residual (cos=0.01) | Don't assume user intent == literal keywords | ✅ Known |
| 02 Manifold Dimensionality | 99% variance in 15D/4096D | Most tool decisions have low intrinsic dimensionality | 🔬 Validate |
| 03 Build Knowledge Layer | Construct layer from triples | Build SmartMan rules from user preferences | 🔬 Future |
| 04 Constellation Insert | Multi-layer weak signals | Distributed confidence scoring | ✅ Documented |
| 05 Syntax Circuit Routing | Cheap features predict expensive ops | Fast tool routing via keywords | 🔬 Research |
| 06 Backprop Insert | Gradients reveal specialization | Profile SmartMan to find approval patterns | 🔬 Future |
| 07 WASM Compute | Deterministic solvers > neural | Tools > LLM for deterministic tasks | ✅ Known |

---

## Anti-Patterns: What NOT to Do

### ❌ Don't Graft Neural Network Internals

**Bad**: Copy gate KNN code into Buddy
**Good**: Study the concept (batch scoring for efficiency), adapt to tool scoring

### ❌ Don't Implement Unvalidated Research

**Bad**: Implement syntax-circuit routing before proving it works
**Good**: Wait for validation, then extract proven pattern

### ❌ Don't Confuse Research with Product

**Bad**: Ship experimental Phase 1 token interception in Buddy v1
**Good**: Use proven message-level tool calling, research streaming for v2

### ❌ Don't Transplant Identity

**Bad**: Call Buddy's fast router "syntax-circuit routing from LARQL"
**Good**: "Fast routing via cheap signals (pattern inspired by LARQL research)"

---

## Success Criteria for Future Donor Extraction

A research pattern is ready for donor extraction when:

✅ **Proven**: Results confirm hypothesis with quantitative validation
✅ **Documented**: Production docs in `docs/`, not just `experiments/`
✅ **Generalizable**: Pattern applies beyond neural networks
✅ **Buddy-mappable**: Clear Buddy use case and landing zone
✅ **Not identity**: Can be adapted without copying LARQL branding/naming

---

## Bottom Line

LARQL experiments are a **research pipeline** for discovering patterns. Not all patterns are donor-ready. Not all patterns apply to Buddy.

**Current state**:
- ✅ 6 patterns extracted and documented (main donor docs)
- 🔬 3 patterns in research stage (syntax routing, token interception, gradient anatomy)
- ❌ 4 patterns not applicable (neural-network-specific)

**Process discipline**:
1. Prove pattern in LARQL experiments
2. Validate with production integration
3. Document in `docs/`
4. Extract to Buddy donor docs (if applicable)
5. Implement in SmartMan (separate repo)

**Repository boundary**: LARQL produces research findings. SmartMan implements Buddy. Never mix the two.

---

**End of advanced experimental patterns documentation.**
