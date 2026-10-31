# TheCombRedux — a loop that changes the comb

A standalone browser-based **causal research sandbox** for connecting three ideas from the October 2026 repository experiments:

1. **MoebiusLoopMemory / LoopAImemory:** a compact, fading, rotating state retains information about earlier encounters.
2. **TheComb:** weighted teeth compute transformations. Here two independently **trained** sigmoid teeth perform copy and invert, and a state-dependent mixture selects between them.
3. **Workspace / J-space inspiration:** a shared, compact representation can control downstream computation. Here it is a **trained 4-class softmax readout** of the entire recurrent vector, **not** an Anthropic Jacobian lens.

**[Open the live HTML lab](https://anttiluode.github.io/TheCombRedux/)** (after enabling GitHub Pages for this repository) or download the repository and open **index.html** locally. No install, server, web API, GPU or account necessary. You may also use GitHub's raw-file download for both \`index.html\` and \`engine.js\`.

## Causal computational path

\`\`\`text
input bit x ───────────────┐
                          ▼
observed tool bit y ─► 16D rotating recurrent state s_t
diagnostic probe ────────►│ 4 fast + 4 slow 2D phase planes
                          ▼
             learned decoder softmax(W [1,s_t])
                          ▼
      workspace: P(world 0,tool 0), P(0,1), P(1,0), P(1,1)
                          │
                          ▼
             world-marginal = p[2]+p[3]
                          ▼
  learned comb = (1−p_world)*copy(x) + p_world*invert(x)
                          ▼
              probabilistic action + score
\`\`\`

Every one of the sixteen recurrent coordinates reaches the learned decoder through its weights. The comb output—not the separately computed reference—determines decisions. The true hidden world/tool values are used to generate events and score outcomes; **they are unavailable to the deployed model**. A standard Bayesian filter running in parallel is the known-model reference.

### Equations

The toy world is \`y = x XOR world XOR tool\` with noisy observations. A tool-calibration probe reports tool state and a world-reference probe reports world state. These probes break a real identifiability ambiguity: \`world=0,tool=0\` and \`world=1,tool=1\` can generate identical passive observations.

Four fast and four slow phase planes use fixed signed projections and a nonlinear input gate:

\`\`\`
v'_(2k:2k+2) = tanh(decay_k * Rotation(theta_k) * v_(2k:2k+2)
                 + (W_in * event)_(2k:2k+2) * (1 + .32*tanh(v_(i+5) % 16)))
\`\`\`

Fast planes: \`decay=.82, theta=.58\`. Slow planes: \`decay=.979, theta=.15\`. They are discrete time rotations, not a physical Möbius surface. The original signed-shift repo did **not** show a unique Möbius-topology advantage.

The workspace readout is \`p = softmax(W*[1,v])\`, where \`W\` is learned offline. The two comb operators each use \`sigmoid(b + w*(2x−1))\` and learn the distinct transformations from supervised synthetic examples. The comb outputs

\`\`\`
P(world output = 1) = (p00+p01)*copy(x) + (p10+p11)*invert(x).
\`\`\`

Training learns readout weights and operator parameters. It does **not** train the reservoir projections, invent new operators, or implement online weight updates. Training uses 340 simulated episodes of 50 steps with true latent labels; evaluation uses new independent seeds and receives no latent labels. \`model\` is frozen during live experiments.

## Use the simulator

- **Send one bit / Run 50 steps**: step the environment and inspect the learned state, operator mixture, predictions and metrics.
- **Flip WORLD / Flip TOOL**: intervene on the two hidden causes without giving the agent their true values.
- **Calibrate tool / Reference world**: purchase independent causal evidence.
- **Autonomous probes**: the current hand-written policy uses uncertainty in the model's previous workspace. It is **not trained** and does not include probe costs in the reported score.
- **Cut recurrent → workspace read**: retains all 16 values, but sends zeros to the learned decoder.
- **Erase 16 coordinates**: destroys previous state before every event.
- **Mask coordinates 2–15**: tests whether the other 14 dimensions matter.
- **Run matched 3-seed comparison**: fixed probe schedule, 320 steps per seed; reports accuracy and Brier score for intact and three ablations. Uses the **same random stream** for each arm.
- **Export JSON**: saves exact observed events, hidden truth for audit, learned probabilities, ring coordinates and operator weights.

### Calculation cautions from the previous LoopWorkspaceLab audit

- **No numerical 50% tie is silently broken**: an exactly equal action probability produces an undecided answer; the accuracy counter gives it no credit. Interpret Brier score as a better continuous metric around .5.
- **No incorrectly signed residual**: the figure plots belief and phase-plane change, not a signed prediction error that subtracts \`P(actual observation)\` from the bit value.
- A 16D vector must alter a decision path to qualify as more than decoration. It now drives the workspace and comb, and the test suite checks that masking or cutting the vector changes model predictions.

## Evidence and negative results

See [Gate 0 result ledger](results/GATE0_2026-10-31.md). In twelve held-out streams, the intact loop affected decisions but scored a worse Brier loss (0.29062) than the cut-read control (0.24998); the analytical Bayesian filter reached 81.93% accuracy versus the learner's 55.86%. **The intended memory architecture has not yet earned a performance advantage.**

## Verification

\`\`\`bash
node --check engine.js
node tests/engine.test.js
\`\`\`

No dependencies. GitHub Actions runs this on pushes and pull requests.

## What would constitute a stronger result?

The initial project is a **mechanistic instrument**, not evidence for a new superior AI. The four-state Bayesian filter knows the correct generative model and is a demanding reference. These gates are necessary before claims of advantage:

1. Compare against a properly trained **GRU / LSTM / EMA**, matched as closely as possible for parameters and compute.
2. Test held-out switches, changing noise, unknown probe timing, unseen observation codes and tool-reliability changes.
3. Fit and evaluate independent randomized training seeds, not just test-stream seeds.
4. Learn active diagnostic selection with explicit probe costs; compare to an oracle and to fixed schedules.
5. Replace the proxy workspace with **actual Jacobian-lens J-space** on a frozen language model, and establish causal read/write beyond output steering.
6. Learn and selectively reawaken richer procedures rather than a predefined two-operator family.

**No selfhood, consciousness, cortical mechanism or new J-space architecture is claimed.** The research question is whether a continuing state can select how information is processed, and when that helps more than a simpler recurrent system.

## Lineage

- [TheComb](https://github.com/anttiluode/TheComb) — fixed weighted teeth and time–space recomputation.
- [MoebiusLoopMemory](https://github.com/anttiluode/MoebiusLoopMemory) — fading circulating state plus contextual gating.
- [LoopAImemory](https://github.com/anttiluode/LoopAImemory) — dual-path temporal memory.
- [RoundAndRound](https://github.com/anttiluode/RoundAndRound) — learned response operators, selective retrieval and failed reconsolidation gates.
- [TransformerX](https://github.com/anttiluode/TransformerX) — history-dependent routing and matched controls.
- [Anthropic J-space](https://transformer-circuits.pub/2026/workspace/) — inspiration for a privileged shared interface; not implemented here.
