# Model Atlas · 模型图鉴

**See every layer of a model work on your own input.** 用你自己的输入，看模型每一层如何工作。

Model Atlas is a free website that teaches how neural networks work. Every lesson is playable: you give a small
model your own input (a drawing, a sentence, a few points on a plane, a game position), and the lesson walks you
through what happens inside, step by step and layer by layer, with the real numbers your input produces. Every
page is in English and 中文.

**Open it:** https://robintianqili.github.io/model-atlas/

<!-- Screenshot placeholder: after the first deploy, add a picture of a lesson here, for example
     ![A TinyCNN lesson step](docs/screenshot.png) -->
> _Screenshot coming soon: a lesson step from the live site._

## What you will find

- **35 live lessons:** 31 model pages and 4 training-concept pages. The catalog plans 103 models in 11 areas and 72 training concepts; pages that are not built yet show a short "Planned" card.
- **Real numbers.** Every panel with model numbers carries a badge: *live* (computed in your browser now), *precomputed* (computed offline from the same model) or *illustrative* (a labelled example).
- **Tiny models that run in your browser.** Most models are trained by this project (`pipeline/`) and are small enough to run, or to train live, on your own device.
- **Fully static.** Nothing is fetched from another host while you use the site, so it also works where large services are blocked.

### Live lessons

**Vision · 视觉**

- [TinyCNN](https://robintianqili.github.io/model-atlas/m/tinycnn/): Draw a doodle and watch each convolution find strokes, corners and shapes. (10-step lesson)
- [Adversarial examples (FGSM / PGD)](https://robintianqili.github.io/model-atlas/m/adversarial-tinycnn/): Nudge every pixel of your drawing a little along the gradient, and flip TinyCNN’s answer. (10-step lesson)
- [TinyViT](https://robintianqili.github.io/model-atlas/m/tinyvit/): The same drawing, read as patches with attention instead of convolutions. (11-step lesson)
- [Tiny U-Net segmentation](https://robintianqili.github.io/model-atlas/m/tiny-unet-seg/): Stamp or paint circles, squares and triangles, and watch a U-Net label every pixel. (11-step lesson)

**Language · 语言**

- [TinyGPT](https://robintianqili.github.io/model-atlas/m/tinygpt/): Type a prompt and follow every number from tokens to next-token probabilities. (12-step lesson)
- [TinyBERT (masked LM)](https://robintianqili.github.io/model-atlas/m/tinybert/): Mask any word and watch a model that reads in both directions fill it in. (10-step lesson)
- [word2vec (skip-gram)](https://robintianqili.github.io/model-atlas/m/word2vec/): Train word vectors on your own tiny corpus and watch similar words drift together. (11-step lesson)
- [Character RNN (LSTM)](https://robintianqili.github.io/model-atlas/m/char-rnn/): A network with memory reads text one character at a time. (11-step lesson)
- [Tiny seq2seq with attention](https://robintianqili.github.io/model-atlas/m/tinyseq2seq/): Convert date formats and watch the decoder look back at the input. (11-step lesson)
- [Tiny Mamba (state-space model)](https://robintianqili.github.io/model-atlas/m/tinymamba/): A sequence model without attention: a scan that decides what to remember. (10-step lesson)

**Time series · 时间序列**

- [TinyForecaster](https://robintianqili.github.io/model-atlas/m/tinyforecaster/): Draw a curve and train a forecaster on it in your browser. (10-step lesson)
- [ECG anomaly autoencoder](https://robintianqili.github.io/model-atlas/m/ecg-autoencoder/): Draw a heartbeat; the network flags the beats it cannot rebuild. (10-step lesson)
- [Kalman filter](https://robintianqili.github.io/model-atlas/m/kalman-filter/): The classic predict-and-correct loop: follow a cart through noisy sensor readings, one second at a time. (10-step lesson)

**Recommendation · 推荐**

- [Matrix factorization](https://robintianqili.github.io/model-atlas/m/matrix-factorization/): Rate ten movies and watch your taste become a point among the movies. (9-step lesson)
- [Two-tower retrieval](https://robintianqili.github.io/model-atlas/m/two-tower/): One tower for you, one for the movies; a dot product picks your shortlist from the whole catalog. (12-step lesson)

**Tabular and classic ML · 表格数据与经典机器学习**

- [MLP playground](https://robintianqili.github.io/model-atlas/m/mlp-playground/): Paint two classes of points and train a small network to separate them. (11-step lesson)
- [Classic-ML canvas](https://robintianqili.github.io/model-atlas/m/classic-ml/): Draw your own dataset and watch k nearest neighbours, a decision tree and k-means work on it, step by step. (12-step lesson)

**Reinforcement learning and control · 强化学习与控制**

- [Bandits and gridworld (tabular RL)](https://robintianqili.github.io/model-atlas/m/gridworld/): Pull slot-machine arms yourself, then build a maze and watch Q-learning solve it. (17-step lesson)
- [Cart-pole with REINFORCE](https://robintianqili.github.io/model-atlas/m/cart-pole/): Train a small network to balance a pole by trial and reward, and watch each step pull its push up or down. (12-step lesson)
- [Snake DQN](https://robintianqili.github.io/model-atlas/m/snake-dqn/): Q-values drawn right on the snake board, then one DQN update worked out with real numbers. (11-step lesson)
- [AlphaZero-lite tic-tac-toe](https://robintianqili.github.io/model-atlas/m/alphazero-ttt/): Guess a tiny network’s favourite move, watch a tree search check it, then play against the search. (11-step lesson)

**Graphs · 图**

- [Graph convolutional network (GCN)](https://robintianqili.github.io/model-atlas/m/gcn/): Edit the friendships of a karate club and watch messages pass between its members. (11-step lesson)
- [GAT / GraphSAGE / GIN](https://robintianqili.github.io/model-atlas/m/gnn-variants/): The same graph, different ways to listen to neighbours. (11-step lesson)
- [Node embeddings (DeepWalk / node2vec)](https://robintianqili.github.io/model-atlas/m/node2vec/): Random walks turn a graph into sentences for word2vec. (10-step lesson)

**Generative models · 生成模型**

- [2D flow matching / diffusion](https://robintianqili.github.io/model-atlas/m/flow-2d/): Paint a 2D shape and train a model that turns noise into it. (12-step lesson)
- [2D GAN](https://robintianqili.github.io/model-atlas/m/gan-2d/): Watch a generator and a discriminator push against each other on a 2D plane. (9-step lesson)
- [VAE with a 2D latent space](https://robintianqili.github.io/model-atlas/m/vae-2d/): Draw a star, a ladder, an umbrella, glasses or a T-shirt, then drag a point through the 2-D map it lives on. (10-step lesson)
- [Score matching and Langevin sampling](https://robintianqili.github.io/model-atlas/m/score-2d/): Arrows that point toward the data, and samplers that follow them. (10-step lesson)
- [Tiny DDPM with a U-Net](https://robintianqili.github.io/model-atlas/m/ddpm-unet/): Bury your drawing in noise, then watch a small U-Net turn fresh noise into a new drawing, step by step. (11-step lesson)

**Multimodal · 多模态**

- [TinyCLIP](https://robintianqili.github.io/model-atlas/m/tinyclip/): Draw a shape, type captions, and see which caption your drawing matches. (10-step lesson)

**Science · 科学**

- [Physics-informed network (damped oscillator)](https://robintianqili.github.io/model-atlas/m/pinn/): Fit a curve from a few data points plus the laws of physics. (10-step lesson)

**Training concepts · 训练概念**

- [Loss and gradient descent](https://robintianqili.github.io/model-atlas/train/loss-gradient-descent/) (Foundations of training): Score a line through eight points, beat the score by hand, then watch gradient descent turn the dials with the real numbers, and break it with a step that is too big. (9-step lesson)
- [Tokenization](https://robintianqili.github.io/model-atlas/train/tokenization/) (Pre-training objectives and data): Byte-pair encoding on your own text: bytes, pre-tokens, pair counts, merges, then real tokenizers. (7-step lesson)
- [Next-token prediction](https://robintianqili.github.io/model-atlas/train/next-token-prediction/) (Pre-training objectives and data): Your sentence from tokens to loss: windows, 1,024 probabilities, −log p, perplexity, training checkpoints, sampling. (7-step lesson)
- [Transfer learning and fine-tuning](https://robintianqili.github.io/model-atlas/train/transfer-learning/): From a forced guess to a new head: frozen features, one training step by hand, few-example curves, BERT at real scale. (8-step lesson)

The full catalog, including the planned pages, is on the site: [Atlas](https://robintianqili.github.io/model-atlas/atlas/) and [Training concepts](https://robintianqili.github.io/model-atlas/train/).

## Run it locally

You need Node 24 (with npm 11). From the repository root:

```
cd site
npm ci
npm run dev
```

Then open the address it prints (http://localhost:5173/ by default). Other commands, all from `site/`:

| Command | What it does |
| --- | --- |
| `npm run build` | builds the static site into `site/build/` |
| `npm run preview` | serves that build |
| `BASE_PATH=/model-atlas npm run build` | builds for a GitHub Pages project site (served under `/model-atlas/`) |
| `npm run check` | type-checks the site |
| `npm run check:catalog` | checks the catalog |
| `npm run check:build` | after a build: every page exists, no third-party requests, size budgets |
| `npm run check:atlas` | the DevKit tests |

Python is only needed to retrain or re-export a model: Python 3.10 or newer, then
`pip install -r pipeline/requirements.txt` (a CPU build of torch is enough for the tiny models).

## Build your own lesson: the DevKit

Every lesson on the site is built with the DevKit: a command-line tool (`npm run atlas -- <command>` in `site/`),
a Python export kit (`pipeline/atlas_kit`), shared views, and a lesson format whose checks keep the numbers real,
the text bilingual and every panel labelled with where its numbers come from.

- **On the site:** [Build your own / 自己动手](https://robintianqili.github.io/model-atlas/devkit/): the guide, the reference of ops and views, a tutorial that builds a small XOR lesson, and how to contribute.
- **The guide:** [`site/DEVKIT.md`](site/DEVKIT.md). The kit API: [`site/KIT.md`](site/KIT.md). The build contracts: [`CONTRACTS.md`](CONTRACTS.md).
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) explains how to add a model page or fix a lesson, and what a pull request needs.
- **For AI coding agents:** the [model-atlas-builder skill](.claude/skills/model-atlas-builder/) holds the DevKit procedure, the teaching quality bar and the check routine; Claude Code picks it up from `.claude/skills/`.

## Repository layout

```
site/                the website (SvelteKit 2, Svelte 5, static output) and the DevKit
  src/lib/catalog/   the catalog: areas, models, training concepts, glossary
  src/lib/explainers/<slug>/   one DevKit lesson package per model (the first pages live in src/lib/pages/)
  static/models/<slug>/        the shipped model files (weights, specs, tapes)
  DEVKIT.md, KIT.md  the DevKit guide and the kit API
pipeline/            Python: training and export scripts per model, atlas_kit, parity checks
research/            design briefs, the BERT facts the transfer-learning page cites, the DevKit afternoon test
LICENSES/            the third-party licence texts (vocabularies, fonts, KaTeX)
.github/workflows/   the GitHub Pages deployment
```

## Credits

Model Atlas is made by Robintianqili. It builds on open work by many others: datasets, tokenizer vocabularies, papers,
other explainers whose ideas shaped the pages, and the libraries and fonts the site is built with (Svelte, SvelteKit,
KaTeX, Fontsource; Fraunces, Atkinson Hyperlegible Next, JetBrains Mono). Each lesson credits its sources on its
page. The full list (69 entries, with licences and attributions) is in [LICENSING.md](LICENSING.md) and
[`site/static/licenses.json`](site/static/licenses.json).

## Licences

- **Code:** MIT, see [LICENSE](LICENSE).
- **Lesson text, articles, narration, figures and diagrams:** CC BY 4.0, see [LICENSE-CONTENT.md](LICENSE-CONTENT.md).
- **Datasets, vocabularies, model files derived from them, fonts and libraries** keep their own licences, see [LICENSING.md](LICENSING.md); the 8 licence texts of the files the site redistributes are in [LICENSES/](LICENSES/).
