---
title: "Wrapping the LeRobot pipeline in an agent loop — LeAgent devlog (M0)"
date: 2026-07-04
authors: ratel
excerpt: "Instead of running collect→train→eval by hand, I wrapped it in a deterministic loop plus agents. Deep research to ground the design (24 claims confirmed, 1 refuted), then a one-day catalog of real-environment bugs on the way to a 3-cycle autonomous loop on a real GPU."
tags:
  - LeAgent
  - LeRobot
  - agents
  - automation
  - SmolVLA
---

> Development log for my side project [LeAgent](https://github.com/ratelcode/LeAgent). I try to keep measured/committed facts and my opinions distinguishable at the sentence level. 한국어 버전: [LeAgent 개발기 (M0)](/ko/blog/leagent_m0_devlog/)

> **Terms, one line each**
> - **LeRobot**: Hugging Face's robotics library — `lerobot-train` / `lerobot-eval` CLIs and the LeRobotDataset v3.0 format.
> - **SmolVLA**: a 450M-parameter vision-language-action base model, designed to be fine-tuned.
> - **LIBERO**: a simulation benchmark of 130 manipulation tasks, officially supported since LeRobot v0.4.0.
> - **Cycle**: one lap of collect→train→eval→decide in LeAgent.

---

## TL;DR

- While doing the SmolVLA tuning series I kept running the "collect data → train → evaluate → decide what's next" loop by hand. LeAgent is that loop as code: an orchestrator drives data/train/eval agents, and a dashboard shows the flow.
- Before writing code I ran a deep-research pass (108 sub-agents, 26 sources, 130 extracted claims → top 25 through 3-vote adversarial verification): **24 confirmed, 1 refuted**. The refuted one — "RoboGen works as an unattended, unbounded data flywheel" — became the core design rule: verification gates at every loop boundary, never assume unbounded automation.
- Key principle: **control flow is a plain Python state machine + SQLite, never an LLM.** LLMs only propose (task curation, knowledge distillation); promote/iterate/escalate/rollback is a pure function of eval deltas.
- All 55 unit tests passing means little until you run the real CLIs. lerobot 0.5.1 produced **10 issues that only showed up in the real environment** — cataloged below.
- The first real-GPU autonomous loop (mini M0: 3 cycles, data growing 8→16→32 episodes, RTX 5070 Ti) finished in 22 minutes — with success flat at 0%, which the loop misread as "policy ceiling" and escalated to a bigger model. That observation became the `escalate_floor` guard (a near-zero plateau means under-training → iterate). **A measurement fixing the design.**

---

## Why

Throughout the SmolVLA tuning series my workflow was: collect some data, run `lerobot-train`, run `lerobot-eval`, look at the numbers, and decide by feel whether to collect more data, change hyperparameters, or switch models. The decision criteria lived in my head; the decision history lived nowhere.

Turning the loop itself into code makes the criteria explicit, keeps the history, and lets it run overnight. That's LeAgent: an open-source orchestrator coordinating data/training/eval/improvement agents over the LeRobot pipeline, with a dashboard to watch the flow.

## Grounding the design in verified research first

Rather than start typing, I first ran a deep-research pass over 2024–2026 papers and the LeRobot codebase: 108 sub-agents pulled 130 claims from 26 sources, and the top 25 went through 3-vote adversarial verification (each vote actively trying to refute). Result: 24 confirmed, 1 refuted.

The confirmed claims became the skeleton:

| Verified fact | Design consequence |
|---|---|
| `lerobot-train`/`lerobot-eval` are scriptable CLI entrypoints | Agents are subprocess wrappers — no coupling to LeRobot internals |
| LeRobotDataset v3.0 is the single data contract for every stage | Agents communicate only via dataset/checkpoint references |
| AutoRT ran LLM orchestration on 52 real robots for 7 months — **with a constitution safety filter and humans in the loop** | A `constitution.yaml` gate (real-hardware and CVE-affected paths denied) |
| DexFlyWheel: IL → residual RL → success-filtered rollouts → augmentation scaled 1 demo to 2,000+ | The template for the M1 self-improvement loop |
| SmolVLA: 20k steps ≈ 4h on one A100; ~50 episodes per task variation needed | Loop cadence and data-collection targets |

And the single refuted claim — "RoboGen can serve as an unbounded, unattended data flywheel" (killed 1–2: sim-only, needs supervision, skill verification is an open bottleneck) — became the most important design rule. **Never assume unbounded automation.** Budgets (cycles, GPU-hours) are hard limits, and promoting a "blessed" checkpoint must stay human-reversible.

Opinion: just marking every design-doc claim as *verified* vs *recommendation* paid for itself within a day — everything that broke later was either in the recommendation zone (streaming, framework choices) or outside the docs entirely (CLI behavior).

## Architecture: LLMs propose, a pure function decides

```
Orchestrator (Python state machine + SQLite)
 ├─ Proposer       : what to collect next (LLM or deterministic — a swappable Protocol)
 ├─ Constitution   : sim-only gate; real-hardware / CVE paths denied + audited
 ├─ Data Agent     : resolve seed dataset, progressive episode schedule (M1: curate/amplify/augment)
 ├─ Train Agent    : lerobot-train wrapper (continues from the blessed checkpoint)
 ├─ Eval Agent     : lerobot-eval wrapper, LIBERO gate
 ├─ Knowledge Agent: distills each cycle into an OKF markdown wiki (Karpathy LLM-Wiki layers)
 └─ decide()       : promote / iterate / escalate / rollback — a pure function of eval deltas
```

Why the decision function is not an LLM: it must be reproducible, it must be unit-testable as a table, and I'm not letting a language model's mood burn GPU budget at 3am. The LLM adapter is provider-agnostic (`anthropic:* | openai:*[@base_url]`, including Ollama/vLLM) and **every flow has a deterministic fallback with no LLM configured at all**.

## The real-environment bug catalog — lerobot 0.5.1

55 unit tests (fake runners) all green; then the real `lerobot-*` subprocesses started. Everything I hit in one day:

| # | Symptom | Cause / fix |
|---|---|---|
| 1 | `--policy.path` absent from `--help` | The parser special-cases it; it works. Trust execution, not help text |
| 2 | Training refuses to start: `'policy.repo_id' argument missing` | Hub push is on by default in v0.5 → pass `--policy.push_to_hub=false` |
| 3 | `FileExistsError: Output directory ... already exists` | We pre-created it by writing our log inside — move logs one level up |
| 4 | Eval dies instantly: `batch size is greater than the number of eval episodes (50 > 2)` | Default eval batch is 50 → `--eval.batch_size=min(batch, episodes)` |
| 5 | eval_info.json parse failure | Real schema is `overall/per_group/per_task`; `pc_success` is 0–100 |
| 6 | Can't download a 70 GB dataset (HuggingFaceVLA/libero) for a smoke test | `--dataset.episodes=[0,1,...]` partial shard download |
| 7 | `--dataset.streaming=true` + SmolVLA → empty batches | **Streaming is incompatible with action-chunking policies** (SmolVLA/ACT/π0) — it can't serve delta-timestamp action windows. The exact failure mode behind the "rough edges" my research pass had flagged |
| 8 | Feature mismatch: policy wants `camera1/2/3`, LIBERO exposes `image/image2` | `--rename_map` on both train and eval; 2-of-3 cameras passes the subset check |
| 9 | CUDA OOM during eval on 16 GB | 5 parallel LIBERO envs (MuJoCo EGL rendering) + policy inference — cap eval batch at 2 |
| 10 | `egl-probe` build failure blocks LIBERO install | ① system EGL headers required ② CMake 4.x rejects the 2018-era CMakeLists → `CMAKE_POLICY_VERSION_MINIMUM=3.5` ③ `libero` prompts interactively on first import → init once with `echo N \| python -c "import libero.libero"` |

Some measurements: SmolVLA fine-tuning on an RTX 5070 Ti (16 GB) runs at ~0.4 s/step (batch 8); a 300-step training job takes ~130 s; a real 4-episode LIBERO eval takes ~308 s. A PushT smoke config (2D, no EGL) does a full cycle in ~1 minute and earned its keep as the pipeline-validation path.

## What the first autonomous loop taught me: a 0% plateau is not a policy ceiling

Mini M0 (300 steps/cycle, data growing 8→16→32 episodes, weights carried over from the blessed checkpoint) completed 3 cycles in 22 minutes. The decision cascade behaved exactly as designed: promote (first cycle = baseline) → iterate → plateau detected → **escalate to π0.5**.

That last decision is the problem. Success was 0.0% across all three cycles — obviously, at 300 steps × 32 episodes — and the loop read the flat signal as "this policy has hit its ceiling" and tried to move up to a 7 GB model. A plateau at zero almost always means under-training or too little data, not insufficient model capacity.

So the decision function gained a guard: **a plateau only escalates when the baseline clears `escalate_floor` (default 5%); below that, iterate.** The observation is pinned as a unit test (`test_zero_plateau_iterates_instead_of_escalating`).

Opinion: this is exactly why the loop should be code. The judgment I was making unconsciously by hand ("it's just under-trained") didn't exist in the code, and the autonomous loop exposed that gap within a day.

## The knowledge layer: accumulate lessons, not just data

The loop accumulates data (events, checkpoints) but not lessons. Following Karpathy's LLM-Wiki pattern (2026-04) and Google's OKF spec (2026-06): immutable raw sources (events.jsonl, SQLite) → an agent-maintained markdown wiki (`knowledge/` — per-task/per-policy pages, YAML frontmatter carrying run/cycle provenance) → a schema file (KNOWLEDGE.md). Observations auto-upgrade `observed-once → replicated` at two reproductions, and `human-confirmed` is never downgraded by the agent. Wiki pages feed the proposer as context — and **never touch control flow.**

## Where it stands, what's next

- As I write this, a full-scale M0 (20k steps/cycle, data 40→80→160 episodes, ~7 h) is running, watched live through the dashboard (`leagent dash` — cycle pipeline, eval chart, rollout videos, event log, knowledge browser).
- Next: the remaining M1 work (DexFlyWheel-style residual RL, RoboGene-style LLM curation). Real-robot work (M3) waits for lerobot 0.6.0, which fixes CVE-2026-25874 (a pickle RCE in the async-inference gRPC path).
- Code: [github.com/ratelcode/LeAgent](https://github.com/ratelcode/LeAgent) — going public around the time this post goes up.
