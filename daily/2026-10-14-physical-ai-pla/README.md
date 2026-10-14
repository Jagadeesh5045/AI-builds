# Perception-Language-Action pipeline (toy embodied-AI demo)

**Trending:** NEURA Robotics unveiled [Neuron AI](https://www.unite.ai/neura-robotics-unveils-neuron-ai-foundation-model-for-physical-ai/) on 14 October 2026 — a multimodal foundation model for "Physical AI" whose Perception-Language-Action (PLA) architecture ties a robot's sight, hearing and touch to past events, encoding every modality into a common, time-aligned representation before conditioning it into a stream of action. Company description of the architecture; capability claims are theirs, not verified here.

**What this demo shows:** a small, honest, stdlib-only version of that pipeline on a simulated tabletop robot:

- **PERCEIVE** — four synthetic sensor encoders (vision patches + edges, audio RMS/ZCR/spectral features via a hand-rolled DFT, tactile pressure grid, joint proprioception) feed a `TimeAlignedBuffer` that resamples every modality onto a common timeline and flags stale streams.
- **REASON** — a seeded `SharedSpace` projects all modalities *and* the natural-language command into one 64-d unit-norm space, so cross-modal cosine queries work; `IntentParser` turns the command into a plan of robot primitives, refusing to guess when the mentioned object is not visible.
- **ACT** — an `ActionDecoder` executes the plan through a `SafetyGate` that enforces workspace bounds, speed and payload limits, a confidence threshold, and an emergency stop. Denials are explicit, never silent.

**Results (demo.py):** the command *"pick up the red cup and place it on the green tray"* succeeds — 6-step plan (move_to, grasp, lift, move_to, place, release), the cup ends exactly on the tray's coordinates. A second run with a 0.01 m/s speed limit is denied by the gate with an explicit reason.

## Files

| File | What it is |
|---|---|
| `pla.py` | Encoders, shared space, time buffer, intent parser, safety gate, action decoder, tabletop world, `run_mission()` |
| `demo.py` | Narrated run: perception alignment scores, plan, execution log, safety-gate demo |
| `test_pla.py` | 17 unit/integration tests |

## Run it

```bash
python3 demo.py        # narrated mission
python3 test_pla.py    # 17/17 tests passing (pure stdlib, no dependencies)
```

## Honest limits

- This is a teaching model, not a foundation model: the "shared space" is seeded random projections over hand-built features, not learned multimodal embeddings.
- The world is a 2D tabletop simulator; real PLA systems (Neuron AI, vision-language-action models) train on large robot datasets and handle 3D, contact dynamics and real sensor noise.
- The audio encoder's DFT is over 64 synthetic samples; the vision encoder paints bright patches rather than detecting objects.
