# harmorust

Rust implementation of the Harmony single-cell integration algorithm.

```rust
use harmorust::{run_harmony, HarmonyConfig, simulate_atlas, SimulateSpec};

let data = simulate_atlas(&SimulateSpec::default());
let z = harmorust::embeddings_from_rows(&data.embeddings)?;
let out = run_harmony(z.view(), &data.batch, &HarmonyConfig::default())?;
```

See the repository README for the seurust workflow and the web demo.
