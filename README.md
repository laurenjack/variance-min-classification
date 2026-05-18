# jl-research

A repo for my personal ML research projects, my recent focus has been studying how we went [from double descent to the scaling laws](https://jacklaurenson.ai/blog/from-double-descent-to-scaling-laws/).

## What's in here

- **General exploration of Overparameterization** - Trying to answer the question of why double descent happens, why overfitting in deep learning is benign, and how LLMs are different.
  
- **bias-variance decomposition of the log loss** - I came up with a way to decompose the log loss into bias and variance terms, and use it to explain what as happening as models are overparameterized.
  
- **Pre-trained LLMs are low variance** - Using the that decompostion, I show pretrained LLMs are definitatively low variance see [nanochat fork](https://github.com/laurenjack/nanochat-variance)
  
- **Variance minimization work** - Many experiments that originally motivated this line of work, studying how architecture and regularization shift model variance and reduces the loss.

## Running it

See [`CLAUDE.md`](./CLAUDE.md) for full setup.

## GPU infrastructure

Code is written in Pytorch, design to run on single instances with multiple NVIDIA GPUs. Experiments are run using MPS given their small size.

Personally, I use [RunPod](https://www.runpod.io), can recommend!

