# WalkFfn — Vindex Graph Walk with Interpretability

**File:** `crates/larql-inference/src/vector_index.rs`
**Status:** Production
**Speed:** Same as WeightFfn (delegates computation)
**Accuracy:** 100% bit-identical to dense

## Description

The production FFN backend for LARQL. Delegates all computation to `WeightFfn` (architecture-
correct dense FFN) and adds the vindex interpretability layer on top. For each layer, captures
a walk trace showing which features activated and what they mean.

This is the backend used by the LQL `INFER` statement.

## Architecture

```
Input x (post-attention residual)
  │
  ├─► WeightFfn::forward(layer, x)  →  exact FFN output
  │
  └─► VectorIndex::gate_knn(layer, x_last, top_k)  →  walk trace
        Feature IDs + gate scores + down_meta labels
```

Computation and trace are independent. The trace doesn't affect the output.

## Walk Trace

Each layer's trace contains:
- **Feature ID** — which FFN feature activated
- **Gate score** — how strongly it activated
- **Down meta** — what token this feature predicts (from the vindex)

Example for "The capital of France is":
```
L27: F9515  gate=+9.247  hears="Paris"   c=0.05
L26: F5040  gate=+7.880  hears="French"  c=0.08
L28: F8200  gate=-5.297  hears="France"  c=0.08
```

## Usage

```rust
use larql_inference::vector_index::{VectorIndex, WalkFfn};

let walk_ffn = WalkFfn::new(weights, &vindex, top_k);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
let trace = walk_ffn.take_trace(); // interpretability layer
```

## LQL

```sql
INFER "The capital of France is" TOP 5;
EXPLAIN WALK "The capital of France is";
```
