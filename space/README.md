---
title: Diverse Sampling Backend
emoji: 🎲
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Exact-log-prob sampling for the diverse-sampling demo
---

Backend for the interactive demo of *A Model of Diverse Sampling from Language
Models* ([arXiv:2611.25003](https://arxiv.org/abs/2611.25003)):
https://timvieira.github.io/dpp-lm-demo/

`/sample` draws N completions from the locally annealed proposal p°_{T′} and
returns each one's exact log p(y) and log q(y) (end-of-string included) plus
unit-norm MiniLM embeddings (base64 float32). The page does the DPP part.

Runs on free CPU hardware with small models; on ZeroGPU the `@spaces.GPU`
decorator attaches a GPU and larger models become available.
