---
title: About
description: Who runs these evaluations and why — author, publishing principles, contact.
---

I'm [ratelcode](https://github.com/ratelcode). I'm an engineer who has worked on the
quality and evaluation of AI and robotics systems, and here I publish, under my own
name, reproducibility reports from running open VLA / imitation-learning policies on
real robots.

## What I do

The main job is checking whether the numbers reported in papers and benchmark tables
actually hold up on a real robot.

- I compare the sim success rate and the SO-101 real-world success rate of open policies
  such as SmolVLA, π0, and ACT.
- I measure how much the success rate moves when I repeat the same conditions or change
  the seed.
- I collect and publish silent failures — failures the score doesn't catch.

How I measure is written up in the [evaluation protocol](/methodology/protocol/), and
the hardware in the [test rig](/methodology/test-rig/). Each report ships with the raw
logs, code, config, and seeds, so if you doubt my numbers you can run it yourself.

## Publishing principles

It must be first-hand data I measured myself; the artifacts needed to reproduce it must
be public; it must make specific, falsifiable claims; negative results go out as-is; and
corrections leave a visible history. A post that can't meet these five doesn't get
published.
([see the full gate](/methodology/protocol/#publication-gate))

I use AI for labor like running the harness, parsing logs, making charts, and
translation. Experiment design, the data, the claims, and the responsibility when
something is wrong are on me.

## Independence

- Everything here is my personal opinion and unrelated to any organization I belong to.
- Evaluations use only public models, public benchmarks, and hardware I own.
- If I receive money or other benefits from a party being evaluated, I state it in that
  report. So far there have been none.

## Contact

I welcome evaluation requests, reproduction questions, and bug reports.

- Email: [ratelcodemoon@gmail.com](mailto:ratelcodemoon@gmail.com)
- GitHub: [@ratelcode](https://github.com/ratelcode) — reach me via issues/discussions
- RSS: [/blog/rss.xml](/blog/rss.xml)
