# Licensing

Model Atlas uses two licences for its own work. Third-party material keeps its own licence.

| What | Licence | Where |
| --- | --- | --- |
| Code: everything that runs, builds or tests the site and the models (`site/`, `pipeline/`, `.github/`), its configuration and the developer documentation | MIT | [LICENSE](LICENSE) |
| Teaching content: lesson text, articles, narration, figures and diagrams, in English and Chinese, on the website and in this repository | CC BY 4.0 | [LICENSE-CONTENT.md](LICENSE-CONTENT.md) |
| Datasets, tokenizer vocabularies, model files derived from them, fonts, libraries and ideas from other projects | their own licences | the list below, [`site/static/licenses.json`](site/static/licenses.json) and the texts in [LICENSES/](LICENSES/) |

Many source files hold both code and teaching text (for example the narration in
`site/src/lib/explainers/<slug>/lesson.ts`, or the articles under `site/src/lib/pages/`). In those files the code is
under MIT and the text and figures are under CC BY 4.0.

## Crediting the teaching content

> Based on Model Atlas (https://robintianqili.github.io/model-atlas/) by Robintianqili, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Say what you changed, if anything.

## Model files and data

The model weights, tapes, reference files and tokenizer files under `site/static/` were produced by the code in
`pipeline/` from the datasets and vocabularies listed below. They keep the licences of their sources; follow the
licence and attribution of each source when you reuse them (for example TinyStories, CDLA-Sharing-1.0, and
Quick, Draw!, CC BY 4.0).

## Licence texts

The site redistributes a few third-party files: tokenizer vocabularies, the self-hosted fonts and KaTeX. Their
licence texts are in [LICENSES/](LICENSES/), with an index in [LICENSES/README.md](LICENSES/README.md):

- [LICENSES/OFL-1.1-Atkinson-Hyperlegible-Next.txt](LICENSES/OFL-1.1-Atkinson-Hyperlegible-Next.txt): Atkinson Hyperlegible Next (font)
- [LICENSES/OFL-1.1-JetBrains-Mono.txt](LICENSES/OFL-1.1-JetBrains-Mono.txt): JetBrains Mono (font)
- [LICENSES/MIT-KaTeX.txt](LICENSES/MIT-KaTeX.txt): KaTeX
- [LICENSES/MIT-GPT-2.txt](LICENSES/MIT-GPT-2.txt): GPT-2 tokenizer vocabulary (byte-level BPE, 50,257 tokens)
- [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt): bert-base-uncased vocabulary (WordPiece, 30,522 tokens)
- [LICENSES/MIT-tiktoken.txt](LICENSES/MIT-tiktoken.txt): tiktoken cl100k_base encoding
- [LICENSES/Apache-2.0-Qwen2.5.txt](LICENSES/Apache-2.0-Qwen2.5.txt): Qwen2.5 tokenizer (byte-level BPE, 151,665 tokens)

## Third-party sources

Generated from `site/static/licenses.json` (139 entries). "Ideas only" means that no code, text or
figures were copied; the source is credited because it shaped the page.

- **[TinyStories (TinyStoriesV2-GPT4)](https://huggingface.co/datasets/roneneldan/TinyStories)**: CDLA-Sharing-1.0.
  Used for: TinyGPT training data (first 100 MiB of TinyStoriesV2-GPT4-train.txt) and validation (TinyStoriesV2-GPT4-valid.txt); TinyGPT's BPE tokenizer was learned from the same text. word2vec's preset mini-corpora are short sentences from TinyStoriesV2-GPT4-valid.txt (about 1.9 KB, chosen by pipeline/word2vec/presets.py), plus a Chinese translation of one of them. Apart from those sentences, only trained weights, the tokenizer and short generated samples ship with the site, not the dataset.
  Attribution: Ronen Eldan and Yuanzhi Li, 'TinyStories: How Small Can Language Models Be and Still Speak Coherent English?' (2023), arXiv:2305.07759
- **[nanoGPT / minGPT (ideas)](https://github.com/karpathy/nanoGPT)**: MIT.
  Used for: Ideas for TinyGPT's model and training loop (pre-LN GPT-2 block, tied head, init scaling, AdamW + warm-up + cosine) and the small GPTs of the positional-encodings extrapolation sweep. Code written fresh for this project.
  Attribution: Andrej Karpathy, nanoGPT and minGPT (MIT License)
- **[minbpe and GPT-2 encoder (ideas)](https://github.com/karpathy/minbpe)**: MIT.
  Used for: Ideas for TinyGPT's byte-level BPE tokenizer (byte fallback, GPT-2-style pre-tokenize regex, rank-ordered merges). Code written fresh for this project.
  Attribution: Andrej Karpathy, minbpe (MIT); OpenAI, GPT-2 encoder.py (MIT)
- **[Transformer Explainer (ideas)](https://github.com/poloclub/transformer-explainer)**: MIT.
  Used for: Ideas for what the TinyGPT page traces and shows (Q/K/V, scores, causal mask, softmax, sampling with temperature / top-k / top-p). No code copied.
  Attribution: Cho et al., Georgia Tech Polo Club, Transformer Explainer (MIT License)
- **[Quick, Draw! dataset (numpy bitmaps)](https://github.com/googlecreativelab/quickdraw-dataset)**: CC BY 4.0.
  Used for: TinyCNN training, validation and test data (first 15,000 drawings of 12 classes: apple, bicycle, car, clock, fish, house, star, sun, tree, umbrella, plus held-out ladder and envelope for the transfer-learning page). Shipped: trained weights and a few hundred 28x28 drawings as presets, channel patches, fine-tuning examples and reference inputs (site/static/models/tinycnn/). Also the VAE’s training and test drawings (star, ladder, umbrella, eyeglasses, t-shirt), its latent map and presets (site/static/models/vae-2d/).
  Attribution: Quick, Draw! dataset by Google Creative Lab, https://quickdraw.withgoogle.com/data (CC BY 4.0)
- **[CNN Explainer (layout idea)](https://github.com/poloclub/cnn-explainer)**: MIT.
  Used for: Idea for the TinyCNN layout (TinyVGG: two blocks of conv-ReLU-conv-ReLU-maxpool, then a dense softmax layer) and for showing per-cell convolution arithmetic. No code copied.
  Attribution: Wang et al., Georgia Tech Polo Club, CNN Explainer (MIT License)
- **[Atkinson Hyperlegible Next (font)](https://github.com/googlefonts/atkinson-hyperlegible-next)**: OFL-1.1.
  Used for: Body text face (self-hosted woff2 via @fontsource-variable/atkinson-hyperlegible-next).
  Attribution: Copyright 2020-2024 The Atkinson Hyperlegible Next Project Authors (Braille Institute of America)
  Licence text: [LICENSES/OFL-1.1-Atkinson-Hyperlegible-Next.txt](LICENSES/OFL-1.1-Atkinson-Hyperlegible-Next.txt)
- **[JetBrains Mono (font)](https://github.com/JetBrains/JetBrainsMono)**: OFL-1.1.
  Used for: Monospace face for numbers, shapes and code (self-hosted woff2 via @fontsource-variable/jetbrains-mono).
  Attribution: Copyright 2020 The JetBrains Mono Project Authors
  Licence text: [LICENSES/OFL-1.1-JetBrains-Mono.txt](LICENSES/OFL-1.1-JetBrains-Mono.txt)
- **[Fontsource](https://github.com/fontsource/fontsource)**: MIT.
  Used for: npm packaging and @font-face CSS for the three self-hosted fonts.
  Attribution: Fontsource contributors
- **[KaTeX](https://github.com/KaTeX/KaTeX)**: MIT.
  Used for: Math rendering in formula cards (JavaScript, CSS and KaTeX fonts bundled and self-hosted).
  Attribution: Copyright (c) 2013-2020 Khan Academy and other contributors
  Licence text: [LICENSES/MIT-KaTeX.txt](LICENSES/MIT-KaTeX.txt)
- **[Svelte](https://github.com/sveltejs/svelte)**: MIT.
  Used for: UI framework; its runtime is bundled into the site.
  Attribution: Copyright (c) 2016-present Svelte contributors
- **[SvelteKit](https://github.com/sveltejs/kit)**: MIT.
  Used for: App framework and static site generation (adapter-static); its client router is bundled into the site.
  Attribution: Copyright (c) 2020 Svelte contributors
- **[GLTR (idea)](https://github.com/HendrikStrobelt/detecting-fake-text)**: Apache-2.0.
  Used for: Idea for coloring each token of the learner's text by the rank the language model gave it (top 1 / 10 / 100 / beyond) on the next-token-prediction page. Reimplemented with TinyGPT and the site's sequential color scale; no GLTR code or text copied.
  Attribution: Hendrik Strobelt, Sebastian Gehrmann, Alexander M. Rush, GLTR: Statistical Detection and Visualization of Generated Text (ACL 2019 demo), Apache License 2.0
- **[AnimatedLLM (idea)](https://github.com/kasnerz/animated-llm)**: MIT.
  Used for: Idea for replaying a language model's training from random weights to fluent text with precomputed traces (the checkpoint replay on the next-token-prediction page). Our replay uses TinyGPT's own checkpoints; no code copied.
  Attribution: Zdeněk Kasner, AnimatedLLM (MIT License)
- **[BertViz (ideas)](https://github.com/jessevig/bertviz)**: Apache-2.0.
  Used for: Idea for the TinyGPT page's layers x heads grid of small attention maps with click-to-zoom and the query/key dot-product detail. No code copied; the page draws its own maps from the live model.
  Attribution: Jesse Vig, BertViz
- **[The logit lens (ideas)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)**: ideas only (blog post; no code or text used).
  Used for: Idea for the TinyGPT page's logit lens: applying the final LayerNorm and the output head to the residual stream after every sub-layer. Computed live by the page's own code.
  Attribution: nostalgebraist (2020), interpreting GPT: the logit lens
- **[GPT-2 tokenizer vocabulary (byte-level BPE, 50,257 tokens)](https://github.com/openai/gpt-2)**: Modified MIT (OpenAI GPT-2).
  Used for: Tokenization page: GPT-2 token bytes, converted to static/pages/tokenization/tok/gpt2.tokb (from huggingface.co/openai-community/gpt2 tokenizer.json)
  Attribution: OpenAI, GPT-2 (Radford et al., 2019)
  Licence text: [LICENSES/MIT-GPT-2.txt](LICENSES/MIT-GPT-2.txt)
- **[bert-base-uncased vocabulary (WordPiece, 30,522 tokens)](https://github.com/google-research/bert)**: Apache-2.0.
  Used for: Tokenization page: BERT WordPiece vocabulary, static/pages/tokenization/tok/bert-base-uncased.txt (vocab.txt from huggingface.co/google-bert/bert-base-uncased)
  Attribution: Google Research, BERT (Devlin et al., 2019)
  Licence text: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)
- **[tiktoken cl100k_base encoding](https://github.com/openai/tiktoken)**: MIT.
  Used for: Tokenization page: GPT-4 / GPT-3.5 cl100k_base token bytes, converted to static/pages/tokenization/tok/cl100k_base.tokb; tiktoken's byte-pair merge and pre-tokenizer regex re-implemented in TypeScript (realtok.ts)
  Attribution: OpenAI, tiktoken
  Licence text: [LICENSES/MIT-tiktoken.txt](LICENSES/MIT-tiktoken.txt)
- **[Qwen2.5 tokenizer (byte-level BPE, 151,665 tokens)](https://huggingface.co/Qwen/Qwen2.5-0.5B)**: Apache-2.0.
  Used for: Tokenization page: Qwen2.5 token bytes, converted to static/pages/tokenization/tok/qwen2.5.tokb; pre-tokenizer regex and NFC normalizer re-implemented in TypeScript
  Attribution: Qwen Team, Alibaba Cloud
  Licence text: [LICENSES/Apache-2.0-Qwen2.5.txt](LICENSES/Apache-2.0-Qwen2.5.txt)
- **[Hugging Face tokenizers (BERT normalizer, BertPreTokenizer, WordPiece; ideas)](https://github.com/huggingface/tokenizers)**: Apache-2.0.
  Used for: Tokenization page: WordPiece algorithm and BERT text normalization re-implemented in TypeScript (realtok.ts); used as the parity reference
  Attribution: Hugging Face
- **[Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset)**: CC BY 4.0.
  Used for: TinyViT and llava-toy training and test drawings (TinyCNN’s 10 classes and split) and the preset drawings
  Attribution: Quick, Draw! dataset by Google Creative Lab
- **[TensorFlow Playground](https://github.com/tensorflow/playground)**: Apache-2.0.
  Used for: Layout ideas for the neuron heatmaps and the network diagram; reimplemented on the Model Atlas autograd core
  Attribution: Daniel Smilkov and Shan Carter, Google (TensorFlow Playground)
- **[Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed., 2018)](http://incompleteideas.net/book/the-book-2nd.html)**: Book, CC BY-NC-ND 2.0; ideas only (no text, figures or code used).
  Used for: the cliff board (Example 6.6) and the order of ideas: bandits, then Q-learning
  Attribution: Richard S. Sutton and Andrew G. Barto
- **[Zachary’s karate club (from NetworkX)](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html)**: NetworkX: BSD-3-Clause; the graph is the published data of Zachary (1977).
  Used for: The graph: 34 members, 78 friendships (unweighted) and the faction each member joined
  Attribution: W. W. Zachary, “An information flow model for conflict and fission in small groups”, Journal of Anthropological Research 33(4), 452–473 (1977); graph from NetworkX karate_club_graph()
- **[Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)](https://arxiv.org/abs/1609.02907)**: Paper; ideas only (no code used).
  Used for: The graph layer Â (H Wᵀ) + b with Â = D^−1/2 (A + I) D^−1/2, the two-layer semi-supervised setup, and the limits step’s remark that two or three layers work best (their appendix B on model depth)
  Attribution: Thomas N. Kipf and Max Welling
- **[Li, Han & Wu, Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning (AAAI 2018)](https://arxiv.org/abs/1801.07606)**: Paper; ideas only (no code used).
  Used for: The over-smoothing picture of the limits step: averaging with Â again and again makes the members’ rows alike
  Attribution: Qimai Li, Zhichao Han and Xiao-Ming Wu
- **[Lipman et al., Flow Matching for Generative Modeling (ICLR 2023)](https://arxiv.org/abs/2210.02747)**: Paper; idea only (no text, figures or code used).
  Used for: the training objective: a network regresses the velocity of paths from noise to data
  Attribution: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
- **[Liu et al., Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (ICLR 2023)](https://arxiv.org/abs/2209.03003)**: Paper; idea only (no text, figures or code used).
  Used for: straight paths x_t = (1 − t)·x0 + t·x1 with target x1 − x0, and few-step Euler sampling
  Attribution: Xingchao Liu, Chengyue Gong, Qiang Liu
- **[CLIP: Radford et al., Learning Transferable Visual Models From Natural Language Supervision (2021)](https://arxiv.org/abs/2103.00020)**: Paper (arXiv); ideas only (no code, weights or data used).
  Used for: the two towers, the batch score table, the symmetric InfoNCE loss and the learned temperature, reimplemented at toy size
  Attribution: Alec Radford, Jong Wook Kim, Chris Hallacy and others (OpenAI)
- **[Hyndman & Athanasopoulos, Forecasting: Principles and Practice (3rd ed., 2021)](https://otexts.com/fpp3/)**: Book, free online; ideas only (no text, figures or code used).
  Used for: the seasonal-naive baseline and checking forecasts on held-out values
  Attribution: Rob J. Hyndman and George Athanasopoulos
- **[Raissi, Perdikaris & Karniadakis, Physics-informed neural networks (J. Comput. Phys. 378, 2019)](https://doi.org/10.1016/j.jcp.2018.10.045)**: Journal article; ideas only (no text, figures or code used).
  Used for: The idea of a physics-informed loss: a data term plus the residual of the governing equation at collocation points, with derivatives by automatic differentiation
  Attribution: Maziar Raissi, Paris Perdikaris and George Em Karniadakis
- **[Ben Moseley, harmonic-oscillator-pinn](https://github.com/benmoseley/harmonic-oscillator-pinn)**: MIT; ideas only (no code copied).
  Used for: The teaching setup: a damped harmonic oscillator, a plain network trained on early points against a PINN that extrapolates
  Attribution: Ben Moseley
- **[Koren, Bell & Volinsky, “Matrix Factorization Techniques for Recommender Systems” (IEEE Computer, 2009)](https://doi.org/10.1109/MC.2009.263)**: Article; ideas only (no text, figures, code or data used).
  Used for: the model (a rating as the dot product of a user vector and an item vector) and its stochastic-gradient-descent training
  Attribution: Yehuda Koren, Robert Bell and Chris Volinsky
- **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (ideas)](https://arxiv.org/abs/1810.04805)**: ideas only (paper; no code or text used).
  Used for: the input layout ([CLS], [SEP], segments) and the 15% / 80-10-10 masking rule; code written fresh
  Attribution: Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova (2019)
- **[wevi: word embedding visual inspector](https://github.com/ronxin/wevi)**: MIT.
  Used for: The idea of training skip-gram live on a tiny corpus with both weight matrices, the softmax and a PCA map in view; reimplemented, no code copied
  Attribution: Xin Rong (wevi; and “word2vec Parameter Learning Explained”, arXiv:1411.2738)
- **[word2vec (skip-gram)](https://arxiv.org/abs/1301.3781)**: Paper (ideas only; no text, figures or code used).
  Used for: The skip-gram model this page trains (with a full softmax instead of the paper’s faster approximations)
  Attribution: Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space” (2013)
- **[GAN Lab](https://github.com/poloclub/ganlab)**: Apache-2.0.
  Used for: Visualization ideas (discriminator heat map, generator manifold, gradient arrows, slow-motion steps); reimplemented on the Model Atlas autograd core
  Attribution: Minsuk Kahng, Nikhil Thorat, Duen Horng (Polo) Chau, Fernanda Viégas and Martin Wattenberg (GAN Lab, Georgia Tech and Google)
- **[TensorFlow.js examples: cart-pole (REINFORCE)](https://github.com/tensorflow/tfjs-examples/tree/master/cart-pole)**: Apache-2.0.
  Used for: The idea and the setup (a policy network giving p(left), discounted and normalized rewards, training in the browser); reimplemented on the Model Atlas autograd core, no code copied
  Attribution: The TensorFlow.js authors, Google
- **[Barto, Sutton & Anderson (1983), Neuronlike adaptive elements that can solve difficult learning control problems](https://doi.org/10.1109/TSMC.1983.6313077)**: Paper; ideas only (the cart-pole equations and constants, as OpenAI Gym states them).
  Used for: The cart-pole physics: masses, pole length, 10 N pushes, 0.02 s steps, the 12° and 2.4 m limits
  Attribution: Andrew G. Barto, Richard S. Sutton and Charles W. Anderson
- **[R2D3, A visual introduction to machine learning (2015)](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)**: Ideas only (no text, figures or code used).
  Used for: the idea of showing each tree split on the data next to the growing tree
  Attribution: Stephanie Yee and Tony Chu (R2D3)
- **[VAE Explainer](https://github.com/xnought/vae-explainer)**: MIT.
  Used for: Ideas only (no code): the reparameterization trick as its own step, and dragging a point through the latent space
  Attribution: VAE Explainer (github.com/xnought/vae-explainer), MIT License
- **[Neural Machine Translation by Jointly Learning to Align and Translate (ideas)](https://arxiv.org/abs/1409.0473)**: ideas only (paper; no code or text used).
  Used for: additive (Bahdanau) attention: scores v·tanh(W_q s + W_k h), softmax weights, the context vector; code written fresh
  Attribution: Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio (2015)
- **[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (ideas)](https://arxiv.org/abs/1406.1078)**: ideas only (paper; no code or text used).
  Used for: the GRU and the encoder-decoder without attention (the bottleneck baseline); PyTorch’s nn.GRU is used as is
  Attribution: Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk and Yoshua Bengio (2014)
- **[R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems” (Journal of Basic Engineering, 1960)](https://doi.org/10.1115/1.3662552)**: Article; ideas only (no text, figures, code or data used).
  Used for: the predict and update equations of the filter
  Attribution: Rudolf E. Kálmán
- **[Hamilton, Ying & Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)](https://arxiv.org/abs/1706.02216)**: Paper; ideas only (no code used).
  Used for: The GraphSAGE layer with the mean aggregator and concatenation
  Attribution: William L. Hamilton, Rex Ying and Jure Leskovec
- **[Xu, Hu, Leskovec & Jegelka, How Powerful are Graph Neural Networks? (ICLR 2019)](https://arxiv.org/abs/1810.00826)**: Paper; ideas only (no code used).
  Used for: The GIN layer (sum, 1 + ε, MLP) and why a sum tells apart more neighbourhoods than a mean
  Attribution: Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka
- **[Veličković et al., Graph Attention Networks (ICLR 2018)](https://arxiv.org/abs/1710.10903)**: Paper; ideas only (no code used).
  Used for: The GAT layer: LeakyReLU scores and a softmax over the neighbours (one head, as PyTorch Geometric’s GATConv)
  Attribution: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio
- **[Jain & Wallace, Attention is not Explanation (NAACL 2019)](https://arxiv.org/abs/1902.10186)**: Paper; ideas only (no code used).
  Used for: The limits step: attention weights compared with the effect of removing an input
  Attribution: Sarthak Jain and Byron C. Wallace
- **[AlphaZero (Silver et al., Science 2018)](https://doi.org/10.1126/science.aar6404)**: Paper; ideas only (no code or figures used).
  Used for: The method: a policy/value network trained by self-play, Monte Carlo tree search with the PUCT rule, visit counts as the policy target; its simulation and TPU counts in the limits step
  Attribution: David Silver, Thomas Hubert, Julian Schrittwieser et al. (DeepMind)
- **[AlphaGo Zero (Silver et al., Nature 2017)](https://doi.org/10.1038/nature24270)**: Paper; ideas only (no code or figures used).
  Used for: The search details: select, expand and evaluate, back up; Dirichlet noise at the root and the eight board symmetries in training
  Attribution: David Silver, Julian Schrittwieser, Karen Simonyan et al. (DeepMind)
- **[The number of legal Go positions (Tromp, 2016)](https://tromp.github.io/go/legal.html)**: Web page; one number quoted.
  Used for: The count of legal Go positions in the limits step
  Attribution: John Tromp
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces (ideas)](https://arxiv.org/abs/2312.00752)**: ideas only (paper; no code or text used).
  Used for: the selective scan (input-dependent Δ, B, C; diagonal A = −exp(A_log); softplus Δ), its initialization and the selective-copying task; code written fresh
  Attribution: Albert Gu and Tri Dao (2023)
- **[The Hidden Attention of Mamba Models (ideas)](https://arxiv.org/abs/2403.01590)**: ideas only (paper; no code or text used).
  Used for: reading the scan of one channel as a matrix of look-back weights (step 9); code written fresh
  Attribution: Ameen Ali, Itamar Zimerman and Lior Wolf (2024)
- **[Goodfellow, Shlens & Szegedy, Explaining and Harnessing Adversarial Examples (ICLR 2015)](https://arxiv.org/abs/1412.6572)**: arXiv paper; ideas only (no text, figures or code used).
  Used for: The fast gradient sign method (FGSM) and adversarial training
  Attribution: Ian J. Goodfellow, Jonathon Shlens and Christian Szegedy
- **[Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks (ICLR 2018)](https://arxiv.org/abs/1706.06083)**: arXiv paper; ideas only (no text, figures or code used).
  Used for: Projected gradient descent (PGD) inside an ε box, and adversarial training with it
  Attribution: Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras and Adrian Vladu
- **[U-Net: Ronneberger, Fischer and Brox, Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)](https://arxiv.org/abs/1505.04597)**: Paper (arXiv); ideas only (no code, weights or data used).
  Used for: the down path, the bottleneck, the up path with concatenated skip connections and the per-pixel softmax, reimplemented at toy size
  Attribution: Olaf Ronneberger, Philipp Fischer and Thomas Brox
- **[Ho, Jain and Abbeel, Denoising Diffusion Probabilistic Models (NeurIPS 2020)](https://arxiv.org/abs/2006.11239)**: Paper; idea only (no text, figures or code used).
  Used for: the forward process, the noise-prediction loss, the sampling formula with σ_t² = β_t, the linear β schedule, sinusoidal timestep embeddings and the EMA of weights
  Attribution: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **[Nichol and Dhariwal, Improved Denoising Diffusion Probabilistic Models (ICML 2021)](https://arxiv.org/abs/2102.09672)**: Paper; idea only (no text, figures or code used).
  Used for: scaling the linear schedule to fewer steps (1000 / T) and sampling on a subset of the steps (respacing)
  Attribution: Alex Nichol, Prafulla Dhariwal
- **[Song and Ermon, Generative Modeling by Estimating Gradients of the Data Distribution (NeurIPS 2019)](https://arxiv.org/abs/1907.05600)**: Paper; idea only (no text, figures or code used).
  Used for: a noise-conditional score network trained at many noise levels, and annealed Langevin sampling with step α·σ²/σ_L²
  Attribution: Yang Song, Stefano Ermon
- **[Vincent, A Connection Between Score Matching and Denoising Autoencoders (Neural Computation 2011)](https://doi.org/10.1162/NECO_a_00142)**: Paper; idea only (no text, figures or code used).
  Used for: denoising score matching: the target (x − x̃) / σ² for a noisy point x̃
  Attribution: Pascal Vincent
- **[Grover & Leskovec, node2vec: Scalable Feature Learning for Networks (KDD 2016)](https://arxiv.org/abs/1607.00653)**: Paper; ideas only (no code used).
  Used for: The biased second-order walk with the return parameter p and the in-out parameter q, and the walks-to-skip-gram recipe (here with a full softmax instead of negative sampling)
  Attribution: Aditya Grover and Jure Leskovec
- **[Perozzi, Al-Rfou & Skiena, DeepWalk: Online Learning of Social Representations (KDD 2014)](https://arxiv.org/abs/1403.6652)**: Paper; ideas only (no code used).
  Used for: The idea of treating random walks on a graph as sentences for word2vec (node2vec with p = q = 1)
  Attribution: Bryan Perozzi, Rami Al-Rfou and Steven Skiena
- **[Yi, Yang, Hong, Cheng, Heldt, Kumthekar, Zhao, Wei & Chi, “Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations” (RecSys 2019)](https://doi.org/10.1145/3298689.3346996)**: Article; ideas only (no text, figures, code or data used).
  Used for: the two-tower retrieval model trained with in-batch negatives (softmax over the items of a batch), at toy size
  Attribution: Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei and Ed Chi (Google)
- **[Covington, Adams & Sargin, “Deep Neural Networks for YouTube Recommendations” (RecSys 2016)](https://doi.org/10.1145/2959100.2959190)**: Article; ideas only (no text, figures, code or data used).
  Used for: the two stages: candidate retrieval by dot products over the whole catalog, then ranking of the short list
  Attribution: Paul Covington, Jay Adams and Emre Sargin (Google)
- **[Anomagram](https://github.com/victordibia/anomagram)**: MIT.
  Used for: Ideas only (no code or data): an autoencoder trained on normal ECG beats of 140 points flags the beats it cannot rebuild; drawing a beat, the rebuild overlay, the threshold, the error histogram and the ROC curve
  Attribution: Anomagram by Victor Dibia (Cloudera Fast Forward Labs), MIT License
- **[Human-level control through deep reinforcement learning (Mnih et al., Nature 2015)](https://doi.org/10.1038/nature14236)**: Paper; ideas only (no code or figures used).
  Used for: The method: a Q-network trained from a replay buffer against a periodically copied target network, with ε-greedy exploration
  Attribution: Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al. (DeepMind)
- **[TensorFlow.js examples: snake-dqn](https://github.com/tensorflow/tfjs-examples/tree/master/snake-dqn)**: Apache-2.0; idea only, reimplemented (no code or weights copied).
  Used for: The idea of the page: snake on a small board read as planes by a convolutional Q-network, with three moves relative to the heading
  Attribution: The TensorFlow.js authors (Google)
- **[micrograd (ideas)](https://github.com/karpathy/micrograd)**: MIT.
  Used for: Ideas for the backpropagation page: a graph of scalar operations whose backward pass adds each edge's share into the parent's gradient (grad +=), and the numerical gradient check. Code written fresh for this project; every number comes from the site's own autograd
  Attribution: Andrej Karpathy, micrograd (MIT License)
- **[Adam: A Method for Stochastic Optimization (Kingma and Ba, 2014)](https://arxiv.org/abs/1412.6980)**: Paper; ideas only (no text, figures or code used).
  Used for: The Adam update and its bias correction, as the kit implements it (the PyTorch form)
  Attribution: Diederik P. Kingma and Jimmy Ba
- **[Decoupled Weight Decay Regularization (Loshchilov and Hutter, 2017)](https://arxiv.org/abs/1711.05101)**: Paper; ideas only (no text, figures or code used).
  Used for: AdamW: weight decay applied to the weights directly, outside the Adam step
  Attribution: Ilya Loshchilov and Frank Hutter
- **[Cyclical Learning Rates for Training Neural Networks (Smith, 2015)](https://arxiv.org/abs/1506.01186)**: Paper; ideas only (no text, figures or code used).
  Used for: The learning-rate range test: raise the rate every step and read the loss curve
  Attribution: Leslie N. Smith
- **[On the difficulty of training recurrent neural networks (Pascanu, Mikolov and Bengio, 2013)](https://arxiv.org/abs/1211.5063)**: Paper; ideas only (no text, figures or code used).
  Used for: The exploding-gradient cliff of a one-unit recurrence and gradient clipping by norm
  Attribution: Razvan Pascanu, Tomas Mikolov and Yoshua Bengio
- **[Why Momentum Really Works (Goh, Distill 2017)](https://distill.pub/2017/momentum/)**: CC-BY-4.0; ideas only (no text, figures or code used).
  Used for: The idea of showing momentum on a long, narrow valley next to plain gradient descent
  Attribution: Gabriel Goh
- **[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5; ideas)](https://arxiv.org/abs/1910.10683)**: ideas only (paper; no code or text used).
  Used for: span corruption: 15% of tokens in spans of mean length 3, sentinels (the random-span mask is written fresh)
  Attribution: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li and Peter J. Liu (2020)
- **[BART: Denoising Sequence-to-Sequence Pre-training (ideas)](https://arxiv.org/abs/1910.13461)**: ideas only (paper; no code or text used).
  Used for: text infilling: 30% of tokens masked in spans with Poisson(3) lengths, one [MASK] per span, the whole text as the target
  Attribution: Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer (2020)
- **[Masked Autoencoders Are Scalable Vision Learners (MAE; ideas)](https://arxiv.org/abs/2111.06377)**: ideas only (paper; no code or text used).
  Used for: masked autoencoding: 75% of the patches hidden, the hidden pixels as the target
  Attribution: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár and Ross Girshick (2022)
- **[A Simple Framework for Contrastive Learning of Visual Representations (SimCLR; ideas)](https://arxiv.org/abs/2002.05709)**: ideas only (paper; no code or text used).
  Used for: contrastive learning with two altered views of one image
  Attribution: Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton (2020)
- **[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA; ideas)](https://arxiv.org/abs/2301.08243)**: ideas only (paper; no code or text used).
  Used for: predicting hidden patches as another network’s numbers instead of pixels
  Attribution: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun and Nicolas Ballas (2023)
- **[RoBERTa: A Robustly Optimized BERT Pretraining Approach (ideas)](https://arxiv.org/abs/1907.11692)**: ideas only (paper; no code or text used).
  Used for: the pitfall on next-sentence prediction: removing it matched or slightly improved results on later tasks
  Attribution: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer and Veselin Stoyanov (2019)
- **[Efficient Training of Language Models to Fill in the Middle (ideas)](https://arxiv.org/abs/2207.14255)**: ideas only (paper; no code or text used).
  Used for: fill in the middle: prefix and suffix moved in front of the middle
  Attribution: Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek and Mark Chen (2022)
- **[CPC: van den Oord et al., Representation Learning with Contrastive Predictive Coding (2018)](https://arxiv.org/abs/1807.03748)**: Paper (arXiv); ideas only (no code, weights or data used).
  Used for: the InfoNCE loss: a softmax over one positive and the negatives, and its bound of log N
  Attribution: Aaron van den Oord, Yazhe Li, Oriol Vinyals (DeepMind)
- **[McCloskey and Cohen, Catastrophic Interference in Connectionist Networks (1989)](https://doi.org/10.1016/S0079-7421(08)60536-8)**: Paper; idea only (no text, figures or code used).
  Used for: the phenomenon the page shows: training on a new task interferes with an old one
  Attribution: Michael McCloskey, Neal J. Cohen
- **[Kirkpatrick et al., Overcoming catastrophic forgetting in neural networks (PNAS 2017)](https://arxiv.org/abs/1612.00796)**: Paper; idea only (no text, figures or code used).
  Used for: the pull toward the old weights (here with the same weight for every parameter, a simplified EWC)
  Attribution: James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, et al.
- **[Robins, Catastrophic Forgetting, Rehearsal and Pseudorehearsal (Connection Science, 1995)](https://doi.org/10.1080/09540099550039318)**: Paper; idea only (no text, figures or code used).
  Used for: replay (rehearsal): training on stored old examples together with the new task
  Attribution: Anthony Robins
- **[Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)](https://arxiv.org/abs/2106.09685)**: Paper; ideas only (no text, figures or code used).
  Used for: The update W0 + (α/r)·B·A with W0 frozen, B starting at zero and a fixed α, and the idea that the change a task needs has a low intrinsic rank
  Attribution: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang and Weizhu Chen
- **[microsoft/LoRA (loralib)](https://github.com/microsoft/LoRA)**: MIT; idea only (no code copied).
  Used for: The initialization convention: A drawn like a PyTorch linear layer's weights (uniform in ±1/√d_in), B all zeros
  Attribution: Microsoft Corporation
- **[Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (NeurIPS 2023)](https://arxiv.org/abs/2305.14314)**: Paper; ideas only (no text, figures or code used).
  Used for: The QLoRA bar of the memory step: the frozen weights stored with 4 bits each, the adapter trained as in LoRA
  Attribution: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer
- **[Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**: Paper; published model configuration only.
  Used for: The sizes of GPT-2 small (12 blocks, width 768, MLP width 3,072, 50,257 tokens, 1,024 positions) for the parameter counter and the memory bars
  Attribution: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever
- **[Quantization from the ground up (ideas)](https://ngrok.com/blog/quantization)**: ideas only (no code or text used; no licence stated).
  Used for: the bit-level format widgets, an outlier that breaks the scale, and judging a quantized model by perplexity and KL divergence; code written fresh
  Attribution: Sam Rose, ngrok blog (2026)
- **[llama.cpp block quantization (ideas)](https://github.com/ggml-org/llama.cpp)**: MIT (ideas only; no code used).
  Used for: blocks of 32 weights that each get their own scale; code written fresh
  Attribution: Georgi Gerganov and the ggml contributors
- **[van den Oord, Vinyals and Kavukcuoglu, Neural Discrete Representation Learning (NeurIPS 2017)](https://arxiv.org/abs/1711.00937)**: Paper; idea only (no text, figures or code used).
  Used for: the VQ-VAE: nearest-code quantization, the straight-through gradient, the commitment loss and the moving-average codebook update
  Attribution: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
- **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (ideas)](https://arxiv.org/abs/1701.06538)**: ideas only (paper; no code or text used).
  Used for: the mixture-of-experts layer with a learned top-k router; code written fresh
  Attribution: Noam Shazeer et al. (2017)
- **[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (ideas)](https://arxiv.org/abs/2006.16668)**: ideas only (paper; no code or text used).
  Used for: top-2 routing in a Transformer and expert capacity (slots dropped when an expert is full); code written fresh
  Attribution: Dmitry Lepikhin et al. (2020)
- **[Switch Transformers (ideas)](https://arxiv.org/abs/2101.03961)**: ideas only (paper; no code or text used).
  Used for: the load-balancing loss E · Σ f·P with weight 0.01 and the capacity factor; code written fresh
  Attribution: William Fedus, Barret Zoph and Noam Shazeer (2021)
- **[Mixtral of Experts (ideas)](https://arxiv.org/abs/2401.04088)**: ideas only (paper; no code or text used).
  Used for: top-2 of the experts with a softmax over the kept scores (the gates), and training without dropping tokens; code written fresh
  Attribution: Albert Q. Jiang et al. (2024)
- **[PointNet (Qi, Su, Mo and Guibas, CVPR 2017)](https://arxiv.org/abs/1612.00593)**: idea only (no code or data used).
  Used for: the architecture (shared per-point MLP, max pooling, classifier) and the idea of critical points
  Attribution: C. R. Qi, H. Su, K. Mo, L. J. Guibas, “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation”
- **[C. E. Rasmussen and C. K. I. Williams, “Gaussian Processes for Machine Learning” (MIT Press, 2006), chapters 2 and 5](https://gaussianprocess.org/gpml/)**: Book; ideas only (no text, figures, code or data used).
  Used for: the posterior equations, the Cholesky-based prediction (algorithm 2.1) and the log marginal likelihood
  Attribution: Carl Edward Rasmussen and Christopher K. I. Williams
- **[Schulman, Wolski, Dhariwal, Radford & Klimov (2017), Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: Paper; ideas only.
  Used for: The clipped objective min(r·A, clip(r, 1 − ε, 1 + ε)·A), several epochs of minibatch updates per batch
  Attribution: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov
- **[Schulman, Moritz, Levine, Jordan & Abbeel (2016), High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)**: Paper; ideas only.
  Used for: GAE: δ = r + γ·V(s′) − V(s), A_t = δ_t + γλ·A_(t+1)
  Attribution: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan and Pieter Abbeel
- **[CleanRL: ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl)**: MIT.
  Used for: The training recipe (separate actor and critic, a state-independent log-std, normalized advantages, shuffled minibatch epochs); reimplemented on the Model Atlas autograd core, no code copied
  Attribution: Shengyi Huang and the CleanRL contributors
- **[Bordes et al., Translating Embeddings for Modeling Multi-relational Data (NeurIPS 2013)](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)**: Paper; ideas only (no code, text or figures used).
  Used for: TransE: entities as points and relations as translations (h + r ≈ t), the L2 distance, the margin ranking loss with one corrupted triple per fact, and link prediction by ranking (here without the paper’s step that rescales entities to length 1)
  Attribution: Antoine Bordes, Nicolas Usunier, Alberto Garcia-Durán, Jason Weston and Oksana Yakhnenko
- **[Wang et al., Knowledge Graph Embedding by Translating on Hyperplanes (AAAI 2014)](https://doi.org/10.1609/aaai.v28i1.8870)**: Paper; ideas only (no code, text or figures used).
  Used for: The analysis of why TransE struggles with one-to-many, many-to-one and many-to-many relations, which the limits step shows with live numbers
  Attribution: Zhen Wang, Jianwen Zhang, Jianlin Feng and Zheng Chen (TransH)
- **[Fourier Neural Operator (Li et al., 2021)](https://arxiv.org/abs/2010.08895)**: Paper; ideas only (no text, figures or code used).
  Used for: The architecture (lift, Fourier layers with a spectral and a pointwise path, projection) and the idea of keeping only the lowest modes; no code or data
  Attribution: Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021
- **[An Interactive Introduction to Fourier Transforms (Jez Swanson)](https://github.com/Jezzamonn/fourier)**: MIT; ideas only (no code copied).
  Used for: Ideas only (no code): drawing a curve and watching it rebuilt from a few waves as the number of kept waves changes
  Attribution: Jez Swanson (Jezzamonn), MIT License
- **[Kang & McAuley, “Self-Attentive Sequential Recommendation” (ICDM 2018)](https://doi.org/10.1109/ICDM.2018.00035)**: Article; ideas only (no text, figures, code or data used).
  Used for: the SASRec model (item and position embeddings, causal self-attention blocks, scores against the shared item embeddings), at toy size
  Attribution: Wang-Cheng Kang and Julian McAuley (UC San Diego)
- **[Attention Is All You Need (ideas)](https://arxiv.org/abs/1706.03762)**: ideas only (paper; no code or text used).
  Used for: the encoder-decoder Transformer: scaled dot-product attention, several heads, the causal mask in the decoder and cross-attention to the encoder; the sinusoidal positional encoding and the √d embedding scale (positional-encodings); code written fresh
  Attribution: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin (2017)
- **[CRNN (Shi, Bai and Yao, 2015)](https://arxiv.org/abs/1507.05717)**: Paper (arXiv); ideas only (no code, weights or data used).
  Used for: the architecture: convolutional features cut into columns, a bidirectional LSTM over the columns and a CTC output, reimplemented at toy size
  Attribution: Baoguang Shi, Xiang Bai and Cong Yao, “An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition”
- **[Connectionist temporal classification (Graves et al., ICML 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)**: Paper; ideas only (no text, figures or code used).
  Used for: the blank symbol, the collapse rule, the forward algorithm over the label with blanks and the CTC loss (training uses PyTorch’s ctc_loss; the page’s own forward algorithm is checked against it)
  Attribution: Alex Graves, Santiago Fernández, Faustino Gomez and Jürgen Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks”
- **[Dinh, Sohl-Dickstein and Bengio, Density estimation using Real NVP (ICLR 2017)](https://arxiv.org/abs/1605.08803)**: Paper; idea only (no text, figures or code used).
  Used for: affine coupling layers y′ = y·e^s + t with alternating masks, the log-determinant Σs, the exact log-likelihood and sampling by inverting
  Attribution: Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
- **[Dinh, Krueger and Bengio, NICE: Non-linear Independent Components Estimation (ICLR 2015 workshop)](https://arxiv.org/abs/1410.8516)**: Paper; idea only (no text, figures or code used).
  Used for: coupling layers that keep one part of the input and are inverted without inverting their network
  Attribution: Laurent Dinh, David Krueger, Yoshua Bengio
- **[Ross, Gordon & Bagnell (2011), A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)**: Paper; ideas only.
  Used for: DAgger: let the learner drive, label the visited states with the expert, aggregate the datasets and retrain
  Attribution: Stéphane Ross, Geoffrey J. Gordon and J. Andrew Bagnell
- **[Ross & Bagnell (2010), Efficient Reductions for Imitation Learning](https://proceedings.mlr.press/v9/ross10a.html)**: Paper; ideas only.
  Used for: Compounding error of behavior cloning: mistakes take the learner to states the expert never showed
  Attribution: Stéphane Ross and J. Andrew Bagnell
- **[Pomerleau (1988), ALVINN: An Autonomous Land Vehicle in a Neural Network](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)**: Paper; ideas only.
  Used for: Behavior cloning: a neural network learns steering from a driver’s demonstrations
  Attribution: Dean A. Pomerleau
- **[E(n) Equivariant Graph Neural Networks (Satorras, Hoogeboom and Welling, ICML 2021)](https://arxiv.org/abs/2102.09844)**: idea only (no code or data used).
  Used for: the EGNN layer: messages from atom features and squared distances, coordinate updates along relative positions, the node update
  Attribution: V. G. Satorras, E. Hoogeboom, M. Welling, “E(n) Equivariant Graph Neural Networks”
- **[Morse potential (P. M. Morse, Physical Review 1929)](https://doi.org/10.1103/PhysRev.34.57)**: idea only (a textbook formula).
  Used for: the shape of the toy pair energy that scores the training molecules (made-up parameters, not real chemistry)
  Attribution: P. M. Morse, “Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels”
- **[L. R. Rabiner, “A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition” (Proceedings of the IEEE, 1989)](https://doi.org/10.1109/5.18626)**: Article; ideas only (no text, figures, code or data used).
  Used for: the forward and Viterbi recursions and their notation (α, δ, ψ)
  Attribution: Lawrence R. Rabiner
- **[A. J. Viterbi, “Error bounds for convolutional codes and an asymptotically optimum decoding algorithm” (IEEE Transactions on Information Theory, 1967)](https://doi.org/10.1109/TIT.1967.1054010)**: Article; ideas only (no text, figures, code or data used).
  Used for: the Viterbi algorithm
  Attribution: Andrew J. Viterbi
- **[Wikipedia, “Hidden Markov model” (the example of a distant friend whose activities depend on the weather)](https://en.wikipedia.org/wiki/Hidden_Markov_model)**: CC BY-SA 4.0; the idea of the example only (no text, figures or numbers used: the tables here are our own).
  Used for: the setting of the story: hidden weather behind walk, shop and clean
  Attribution: Wikipedia contributors
- **[Bai, Kolter & Koltun, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (2018)](https://arxiv.org/abs/1803.01271)**: Paper; ideas only (no text, figures or code used).
  Used for: the TCN: causal dilated convolutions in residual blocks
  Attribution: Shaojie Bai, J. Zico Kolter and Vladlen Koltun
- **[van den Oord et al., WaveNet: A Generative Model for Raw Audio (2016)](https://arxiv.org/abs/1609.03499)**: Paper; ideas only (no text, figures or code used).
  Used for: dilations that double layer by layer, and the receptive field they give
  Attribution: Aäron van den Oord and co-authors
- **[Hannan & Rissanen, Recursive estimation of mixed autoregressive-moving average order (Biometrika, 1982)](https://doi.org/10.1093/biomet/69.1.81)**: Paper; ideas only (no text, figures or code used).
  Used for: ARIMA-lite: the two-stage least-squares fit of the AR and MA coefficients
  Attribution: E. J. Hannan and J. Rissanen
- **[LLaVA: Liu et al., Visual Instruction Tuning (2023) and Improved Baselines with Visual Instruction Tuning (2023)](https://arxiv.org/abs/2304.08485)**: Papers (arXiv); ideas only (no code, weights or data used).
  Used for: a frozen vision encoder read at its second-to-last layer without the [CLS] row, the two-layer GELU projector, image tokens placed in the language model’s sequence and the loss on the answer only, at toy size
  Attribution: Haotian Liu, Chunyuan Li, Qingyang Wu, Yuheng Li and Yong Jae Lee
- **[bitsandbytes NF4 data type (the 16 NF4 levels)](https://github.com/bitsandbytes-foundation/bitsandbytes)**: MIT.
  Used for: the table of the 16 NF4 levels (fmt.ts NF4), recomputed from normal quantiles in pipeline/number-formats/flib.py; the rounding code is written fresh
  Attribution: QLoRA (Dettmers, Pagnoni, Holtzman and Zettlemoyer, 2023) and the bitsandbytes contributors
- **[FP8 formats for deep learning; OCP Microscaling (MX) formats (ideas)](https://arxiv.org/abs/2209.05433)**: ideas only (no code or text used).
  Used for: the e4m3 and e5m2 encodings (e4m3 without infinity, one NaN code) and FP4 e2m1 with a scale per block of 32; code written fresh and checked against PyTorch’s float8 dtypes
  Attribution: Micikevicius et al. (2022); Open Compute Project MX specification v1.0 (2023)
- **[Mixed precision training (ideas)](https://arxiv.org/abs/1710.03740)**: ideas only (no code or text used).
  Used for: fp32 master weights and loss scaling, shown on one weight and on TinyGPT’s gradients; code written fresh
  Attribution: Micikevicius et al. (2018)
- **[The Curious Case of Neural Text Degeneration (ideas)](https://arxiv.org/abs/1904.09751)**: ideas only (paper; no code or text used).
  Used for: top-p (nucleus) sampling, and repetition under greedy decoding
  Attribution: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes and Yejin Choi (2020)
- **[Hierarchical Neural Story Generation (ideas)](https://arxiv.org/abs/1805.04833)**: ideas only (paper; no code or text used).
  Used for: top-k sampling
  Attribution: Angela Fan, Mike Lewis and Yann Dauphin (2018)
- **[Prompt caching (ideas)](https://ngrok.com/blog/prompt-caching/)**: ideas only (no code or text used; no licence stated).
  Used for: an attention grid that fills as tokens are generated, with the cached work shown as skipped; code written fresh
  Attribution: Sam Rose, ngrok blog (2025)
- **[Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)**: Paper; ideas only (no text, figures or code used).
  Used for: RoPE: each 2-D pair of a query and a key turned by position × θ_i with θ_i = 10000^(−2i/d), so that their dot product depends on the distance only
  Attribution: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen and Yunfeng Liu
- **[Press et al., Train Short, Test Long: Attention with Linear Biases (ALiBi, ICLR 2022)](https://arxiv.org/abs/2108.12409)**: Paper; ideas only (no text, figures or code used).
  Used for: ALiBi’s bias −m × distance with the head slopes 2^(−8/H), 2^(−16/H), …, and the train-short, test-long comparison
  Attribution: Ofir Press, Noah A. Smith and Mike Lewis
- **[Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (EMNLP 2023)](https://arxiv.org/abs/2305.13245)**: Paper; ideas and published numbers only (no text, figures or code used).
  Used for: Grouped-query attention, converting a multi-head checkpoint by mean pooling each group’s key and value heads, and uptraining with 5% of the pretraining steps; the page repeats both on TinyGPT
  Attribution: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón and Sumit Sanghai
- **[Shazeer, Fast Transformer Decoding: One Write-Head is All You Need (2019)](https://arxiv.org/abs/1911.02150)**: Paper; ideas and published numbers only (no text, figures or code used).
  Used for: Multi-query attention: one key head and one value head shared by every query head
  Attribution: Noam Shazeer
- **[DeepSeek-AI, DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)](https://arxiv.org/abs/2405.04434)**: Paper; ideas and published numbers only (no text, figures or code used).
  Used for: Multi-head latent attention and the DeepSeek-V2 preset: 60 layers, 128 heads of width 128, a cached latent of 512 plus a 64-wide RoPE key per token and layer
  Attribution: DeepSeek-AI
- **[Touvron et al., Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)](https://arxiv.org/abs/2307.09288)**: Paper; ideas and published numbers only (no text, figures or code used).
  Used for: The Llama 2 7B (32 layers, 32 heads, MHA) and Llama 2 70B (80 layers, 64 heads, 8 KV heads) presets, head width 128
  Attribution: Hugo Touvron, Louis Martin, Kevin Stone and the Llama 2 team (Meta)
- **[Llama Team, AI @ Meta, The Llama 3 Herd of Models (2024)](https://arxiv.org/abs/2407.21783)**: Paper; ideas and published numbers only (no text, figures or code used).
  Used for: The Llama 3 8B preset: 32 layers, 32 heads, 8 KV heads, head width 128
  Attribution: Aaron Grattafiori, Abhimanyu Dubey and the Llama 3 team (Meta)
- **[Almazrouei et al., The Falcon Series of Open Language Models (2023)](https://arxiv.org/abs/2311.16867)**: Paper; ideas and published numbers only (no text, figures or code used).
  Used for: The Falcon-7B preset: 32 layers, 71 query heads of width 64, multi-query attention
  Attribution: Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi and the Falcon team (TII)
- **[AI Notes: Initializing neural networks (ideas)](https://www.deeplearning.ai/ai-notes/initialization/)**: ideas only (no code or text used).
  Used for: the idea of showing per-layer histograms of a deep network under zero, too small, Xavier and too large initialization; network, data and code written fresh
  Attribution: Katanforoosh and Kunin, deeplearning.ai (2019)
- **[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ideas)](https://arxiv.org/abs/2003.10555)**: ideas only (paper; no code or text used).
  Used for: the point that a masked model learns from only the chosen ~15% of positions per example, and the further-reading link
  Attribution: Kevin Clark, Minh-Thang Luong, Quoc V. Le and Christopher D. Manning (2020)
