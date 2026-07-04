---
title: About
description: Who runs these evaluations and why — author, publishing principles, contact.
---

I'm [ratelcode](https://github.com/ratelcode), an engineer who has worked on
quality and evaluation of AI and robotics systems. Here I publish, under my own
name, reproducibility reports from running open VLA / imitation-learning
policies on a real robot.

## What I do

The core job is checking whether the numbers printed in papers and benchmark
tables also show up on a physical robot.

- Compare sim success rates of open policies (SmolVLA, π0, ACT, …) against
  measured success rates on an SO-101 arm.
- Measure how much success rates wobble across repeats and seed changes under
  identical conditions.
- Collect and publish failure cases that scores don't capture (silent failures).

The measurement method lives in the [evaluation protocol](/methodology/protocol/)
and the hardware in the [test rig](/methodology/test-rig/) page. Reports ship
with raw logs, code, configs, and seeds — if you doubt my numbers, run them
yourself.

## Publishing principles

First-hand measured data only; publish the artifacts needed for reproduction;
make concrete claims that can be wrong; publish negative results as-is; keep a
correction history when wrong. Posts that can't clear these five gates don't go
up. ([Full gates](/methodology/protocol/#publishing-gates))

AI does the labor — harness runs, log parsing, charts, translation. Experiment
design, data, claims, and the responsibility for being wrong stay human.

## Independence

- Everything on this site is personal opinion, unrelated to any employer.
- Evaluations use open models, open benchmarks, and my own hardware only.
- If an evaluated party ever provides money or favors, the report will say so.
  So far: none.

## Contact

Evaluation requests, reproduction questions, and error reports all welcome.

- Email: [ratelcodemoon@gmail.com](mailto:ratelcodemoon@gmail.com)
- GitHub: [@ratelcode](https://github.com/ratelcode) — issues/discussions work
- RSS: [/blog/rss.xml](/blog/rss.xml)
