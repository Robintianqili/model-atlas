# Design brief: an integrated, playable model-visualization site

Date: 2026-10-05 (revision 2). Scope: one free, static site, in English with a Chinese toggle. It covers models from every application area plus a Training Concepts track, and each page uses a CNN-Explainer-style view driven by the learner's own input.

Starting point: the builder has a single GPU for training and precomputation, GitHub Pages for hosting, and existing animated BERT explainer slides and a BERT reproduction (MRPC/RTE).

Ground rule: every number, license and URL below comes from the research evidence or from the re-checks done for this revision. Labels used:
- **unverified**: we could not confirm it.
- **derived**: arithmetic on verified numbers.
- **design target**: our own plan, not a fact.
- **(young)**: a repository with at most 1 star, created in 2026. We take ideas from it but do not treat its claims as evidence.

**Summary**
- **There is an opening, but it is narrower than "nobody integrates".**
  - Polo Club's explainers (CNN, Transformer, Diffusion, GAN Lab) set the quality bar. They are separate apps.
  - Google PAIR's LIT is the closest "one UI, many models, your own datapoints" tool. It covers text, image and tabular models, but needs a Python server or notebook and is aimed at practitioners.
  - TensorSpace (2019; 5 pretrained models in 3D; last push 2022) and NNvisual (2026; claims Transformer, CNN, MLP and diffusion in vanilla JS; 0 stars) are small precedents.
  - What no one offers is the following combination, which is what sets us apart:
    - (1) breadth across 11 application areas in one visual grammar, on the learner's own input;
    - (2) an interactive Training Concepts track (pre-training, fine-tuning, PEFT, alignment, compression, evaluation, inference);
    - (3) provenance honesty (Live / Precomputed / Illustrative, with precision and parity tolerance);
    - (4) zh/en content;
    - (5) static hosting that survives without servers.
- **Recommended approach.** A SvelteKit static site with one shared component kit and one visual grammar. Every model plugs into a `ModelRunner` with three backends:
  - precomputed traces (always available, the phone default);
  - live inference in a Web Worker (our own JS forward pass for tiny models; ONNX Runtime Web or Transformers.js for Hub models);
  - live in-browser training on a small in-house autograd/tensor core.
- **GPU strategy.** Train tiny exact models ourselves (≤1 MB) so every on-screen number is computed in the browser, within a stated tolerance of PyTorch. Precompute traces for anything larger. Use one shared backbone family (TinyGPT; nanochat at small depth for real-scale lifecycle traces) so the pre-training, SFT, RL, LoRA and quantization pages compare the same model.
- **Scope is frozen for v1** at the MVP plus one flagship live model per remaining area (11 model pages in total) and about 11 concept pages. Every other page in §5 and §6 is listed as "planned" and gets built only when the shared kit already covers about 90% of it.
- **MVP:** 4 models from 4 areas (TinyCNN, TinyGPT, keyword spotting, 2D flow matching) and 3 concept pages (tokenization, next-token prediction, transfer learning/fine-tuning).

---

## 1. Landscape at a glance

Depth: **model** = whole-model outputs, **layer** = per-layer or per-head values, **neuron** = per unit, weight or arithmetic step.

| # | Project | Area | Model families / concepts | Custom input? | Depth | Runs how | License | The one idea worth taking |
|---|---|---|---|---|---|---|---|---|
| 1 | [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | Vision | CNN (TinyVGG, 10 classes, 64×64) | Yes: upload file or URL; presets | Neuron (kernel arithmetic) | In-browser (TF.js 1.4, Svelte 3; 79.7 KB weights) | MIT | Three-level semantic zoom (overview → layer → one neuron's arithmetic), with hover-linking between output cell, input window and kernel |
| 2 | [ConvNetJS demos](https://cs.stanford.edu/people/karpathy/convnetjs/) | Vision / foundations | MLP, CNN, autoencoder, DQN, coordinate-MLP "painting" | Partial: click 2D points, upload image (painting demo), editable architecture text | Neuron/filter, including gradients | In-browser training (plain JS) | MIT | Live training that shows activations, activation gradients, weights and weight gradients together, plus a "hidden layer warps the plane" view |
| 3 | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | Vision explainability | CNN, ViT, Swin, detection, segmentation | n/a (offline library) | Chosen layer × class | Precompute (PyTorch) | MIT | One formula card, CAM = ReLU(Σ w_k A_k), where each method is just a different way to choose w_k. Bake low-resolution maps for every layer × method × class |
| 4 | [ViT-Explainer](https://vit-explainer.vercel.app/) | Vision | ViT | Yes per paper (upload); live URL is now a user-study portal | Layer / head / token | Server | unverified (code not released) | A tiny 3×3 patch grid so every token is visible; vision logit lens; guided and free modes |
| 5 | [WebGPU ViT attention](https://lyonsno.github.io/webgpu-vit-attention/) (young: 1 star, created and last pushed 2026-06-25) | Vision | ViT (DeiT-Tiny) | Yes: drag-and-drop image | Layer / head | In-browser (WGSL, ~5 MB weights) | BSD-3-Clause | Suggests a real ViT can run fully static on GitHub Pages, with layer and head sliders (unvetted; reproduce before relying on it) |
| 6 | [Segment Anything WebGPU](https://huggingface.co/spaces/Xenova/segment-anything-webgpu) | Vision | SAM (SlimSAM-77) | Yes: upload, positive/negative point clicks | Model (stage split) | In-browser (Transformers.js, 20.7 MB fp16) | Apache-2.0 (example code and SlimSAM weights) | "Encode once, decode per click", felt through latency. It hides the 3 candidate masks, IoU scores and embedding; we would show them |
| 7 | [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Language | GPT-2 small | Yes: any prompt (12-word cap); temperature, top-k, top-p | Layer/head down to matrix operations | In-browser (ORT Web WASM, 63 chunks = 656,662,664 B ≈ 626 MiB fp32) plus 5 precomputed examples | MIT (8,822 stars; pushed 2026-06-06) | Sankey overview that expands in place, and a "textbook" tour whose pages drive the visualization. Caveat: its embedding, Q/K/V and MLP stripes are `Math.random()` placeholders, which we must not copy |
| 8 | [LLM Visualization](https://bbycroft.net/llm) | Language | GPT (nano-gpt live; GPT-2/3 structure only) | No (fixed letter-sorting task) | Every weight and activation cell | In-browser (WebGL2) | MIT (homepage and third-party assets excluded) | Separate colors for weights and activations; a narrated walkthrough that drives the camera; true-scale size comparison |
| 9 | [AnimatedLLM](https://animatedllm.github.io) | Language / pre-training | Decoder LLMs, next-token prediction | No (curated inputs; offline scripts for new ones) | Layer | Precomputed JSON traces | MIT | Run real models offline, export traces, animate them statically; simple and detailed modes; Chinese UI |
| 10 | [BertViz](https://github.com/jessevig/bertviz) | Language | BERT, GPT-2, T5 | Yes, in a notebook only | Head; neuron view down to q/k elements | Notebook/server, exports static HTML | Apache-2.0 | A layers × heads thumbnail grid with click-to-zoom, and a q·k neuron view. Ship as uint8 (57 KB gzipped vs 5.27 MB HTML for 29 tokens) |
| 11 | [Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) | Language / pre-training | BPE, WordPiece, SentencePiece | Yes: any text, 13 tokenizers or any Hub id | Input stage | In-browser (Web Worker) | Apache-2.0 (code; tokenizer files vary) | Re-tokenizes on every keystroke, keeps state in the URL. We would add side-by-side comparison, a byte view and Chinese |
| 12 | [wevi](https://ronxin.github.io/wevi/) | Language | word2vec CBOW / skip-gram | Yes: your own context/target mini-corpus | Every weight | In-browser training | MIT | Show the whole tiny model, with both weight matrices updating, next to a PCA plot |
| 13 | [Visualizing Memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/) | Language / sequence | GRU, LSTM, Nested LSTM | No | Per-input-character connectivity | Precomputed | MIT | A connectivity heatmap over the text answers "how far back does it look?"; compares cell types on one sentence |
| 14 | [TF.js Speech Commands](https://storage.googleapis.com/tfjs-speech-model-test/2019-01-03a/dist/index.html) | Speech/audio | Keyword-spotting CNN (18 words), transfer learning | Yes: microphone; record your own words | Model (all layers extractable) | In-browser (TF.js, 5.9 MB) | Apache-2.0 (weights' license unverified) | Backbone for a keyword-spotting (KWS) explainer: spectrogram → conv maps → softmax, plus a frozen-base fine-tune from 8 clips per word |
| 15 | [Whisper Web](https://huggingface.co/spaces/Xenova/whisper-web) | Speech | Whisper (encoder-decoder ASR) | Yes: microphone or file | Model | In-browser (Transformers.js) | MIT (Whisper code and weights are MIT per openai/whisper) | Whisper-tiny runs client-side, so a Whisper explainer can process the learner's own voice |
| 16 | [Kokoro TTS WebGPU](https://huggingface.co/spaces/webml-community/kokoro-webgpu) | Speech | TTS (82M, StyleTTS2 fine-tune) | Yes: any text, 28 voices | Model (pipeline stages visible in JS) | In-browser (kokoro-js; q8 92 MB) | Apache-2.0 (eSpeak NG upstream is GPL-3.0; whether kokoro-js bundles it is unverified) | Text normalization → IPA → tokens → audio; an fp32/q8/q4 A/B comparison makes quantization audible |
| 17 | [Anomagram](https://anomagram.fastforwardlabs.com/) | Time series | Autoencoder anomaly detection | Yes: draw an ECG; design and train the net | Layer config; input vs reconstruction | In-browser (TF.js) | MIT | Draw a signal, see the reconstruction and its error, with a live threshold, error histogram and ROC |
| 18 | [Two-tower recommender demo](https://thanushthrony.github.io/projects/recommender-demo.html) | Recommendation | Two-tower retrieval | Yes: pick movies you like | Stage | User tower live, item tower precomputed | none listed | The static-site recommender pattern. Ideas only |
| 19 | [TensorFlow Playground](https://playground.tensorflow.org) | Classic ML / foundations | MLP, features, regularization | Partial: dataset, features, architecture, editable weights | Neuron and weight | In-browser (hand-written TS) | Apache-2.0 | A heatmap of each neuron over the input plane; full state in the URL hash, with `*_hide` flags for embedding in lessons |
| 20 | [MLU-Explain](https://mlu-explain.github.io/) | Classic ML | Regression, trees, random forests, metrics, double descent, RL | Partial: sliders, scroll | Model to neuron (simple nets) | In-browser / precomputed | CC BY-SA 4.0 articles + modified MIT code (GitHub reports NOASSERTION); repo **archived**, last push 2024-11-12 | Card-gallery home page and pinned-figure scrollytelling as the article template |
| 21 | [Beginner's RL Playground](https://awjuliani.github.io/web-rl-playground/) | RL | Q-learning, SARSA, Monte Carlo, actor-critic, successor representation | Yes: edit cells, rewards, all hyperparameters | State / action | In-browser | MIT | One map the learner builds, many algorithms, switchable cell-display modes |
| 22 | [TF.js cart-pole](https://storage.googleapis.com/tfjs-examples/cart-pole/dist/index.html) | RL | REINFORCE policy gradient | Partial: hidden sizes, hyperparameters | Model + training curve | In-browser training (769 params) | Apache-2.0 | A complete ~150-line REINFORCE loop, ready to wrap in a step-through view of returns, advantages and gradients |
| 23 | [GNN 101](https://visual-intelligence-umn.github.io/GNN-101/) | Graphs | GCN, GAT, GraphSAGE | Partial: dataset, model, molecule | Layer → node → matrix cell | In-browser (ORT Web, 25–113 KB models) | none specified | ONNX runs the real forward pass while JSON weights let the page redo each equation; node-link and matrix views. Ideas only |
| 24 | [GAN Lab](https://poloclub.github.io/ganlab/) | Generative | GAN | Yes: draw a 2D distribution; edit hyperparameters | Component + per-sample gradient | In-browser training (TF.js) | Apache-2.0 | One shared 2D space showing the discriminator heatmap, generator manifold and gradient arrows, plus a slow-motion mode with numbered steps |
| 25 | [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/) | Generative / multimodal | Stable Diffusion (CLIP, U-Net, VAE, CFG) | No (13 preset prompts) | Component × timestep | Precomputed | MIT | Precompute the heavy model, then compare prompt pairs with a timestep scrubber and UMAP latent trajectories |
| 26 | [Diffusion Explorer](https://alechelbling.com/Diffusion-Explorer/) | Generative | Flow matching (live page shows flow matching only; DDPM reported on main, not re-verified) | Yes: brush a 2D distribution and train | Process (samples and density) | In-browser training (TF.js worker, 8,706-param MLP) | none (1,190 stars; pushed 2026-09-05; still unlicensed) | Individual paths and density contours shown together; time as a universal scrubber. Re-implement only |
| 27 | [VAE Explainer](https://xnought.github.io/vae-explainer/) | Generative | VAE | Yes: draw a digit; drag the latent point | Stage | In-browser (TF.js) | MIT | Code lines synced with graph highlights; the reparameterization trick as its own visual step |
| 28 | [Transformers.js examples](https://github.com/huggingface/transformers.js-examples) | Multimodal / vision | CLIP, DINOv2 attention, SmolVLM, Florence-2, Janus, SAM | Yes: images and text | Layer/head (attention example); otherwise model | In-browser (worker, WebGPU or WASM) | Apache-2.0 | Re-export a model "with-attentions" and read the extra outputs by name; post ImageBitmaps from the worker |
| 29 | [AlphaFold DB + Mol*](https://alphafold.ebi.ac.uk/) | Science | Protein structure, pLDDT, PAE | Yes: any protein name or UniProt id | Model outputs | Precomputed (CORS-open API) | Mol* MIT; AFDB data license unverified | Real predictions inside a static site; a PAE-heatmap selection highlights residues in 3D |
| 30 | [PINNLab](https://pinn-lab.vercel.app) (young: 0 stars, single-day repo created 2026-04-21) | Science | Physics-informed neural networks | Partial: choose the PDE and hyperparameters | Model + loss terms | In-browser (TF.js) | MIT | Loss-decomposition bars and a λ slider make "physics as a loss term" tangible |
| 31 | [micrograd](https://github.com/karpathy/micrograd) (+ [Autograd UI](https://autograd.hewliyang.com)) | Concept: backprop | Autograd, MLP | Python only; UI has fixed examples | Per scalar operation | none / in-browser | MIT / none | A ~150-line engine as the single source of truth; step backward one chain-rule edge at a time |
| 32 | [Backprop Explainer](https://xnought.github.io/backprop-explainer/) | Concept: backprop | MLP, gradient descent | Partial: drag weights, train | Neuron and weight gradients | In-browser (TF.js) | MIT | The learner tunes by hand first, then gradient descent takes over; chain-rule terms color-coded; "nudge" arrows |
| 33 | [Why Momentum Really Works](https://distill.pub/2017/momentum/) | Concept: optimization | Gradient descent, momentum, step size | Partial: α/β sliders, clickable map | Optimizer | In-browser | CC-BY-4.0 | Two sliders drive every linked figure; a clickable (α, β) convergence map |
| 34 | [Sequence Toy](https://sequence.toys) | Concept: pre-training dynamics | Transformer/LSTM/GRU; LR schedules, optimizers, MLM vs CLM | Partial: task + every hyperparameter | Layer (activations and gradients) | In-browser training (WebGPU) | MIT | Real training in a browser tab; "You could try…" links set a config and point at the control; config lives in the URL |
| 35 | [GLTR](http://gltr.io) | Concept: next-token prediction | GPT-2 surprisal | Yes: paste text (demo server) | Per-token output | Server | Apache-2.0 | Color each token by the rank the model gave it; hover to see what it expected instead |
| 36 | [Kingy.ai Atlas](https://kingy.ai/atlas/lora-rank/) | Concept: LoRA, quantization | LoRA (rank, storage, VRAM) | Yes (LoRA page): k, d, rank r, alpha, storage precision, identical-targets toggle. No per-weight value entry | Parameter counts (analytic) | none (analytic) | Only three named diagrams are CC BY 4.0 (v2026-09-07): "Follow the low-rank path", "Rank has a storage slope", "Parameters are not VRAM". Tool and text are not covered | LoRA drawn as a narrow detour around the frozen W₀; "parameters are not VRAM" |
| 37 | [AIFromZero / DeepLearningFromZero](https://github.com/dev48v/ai-from-zero) (young: 0 stars, created 2026-06-23) | Concepts: fine-tuning, alignment, compression | LoRA, RLHF, DPO, distillation, quantization, pruning, transfer | Partial: sliders; the learner labels preference pairs | Per weight (toy) | In-browser toy training | MIT | A LOOK / UNDERSTAND / BUILD template; "you are the labeller"; a KL-leash β slider that shows Goodhart's law |
| 38 | [Crucible](https://github.com/The-CarL/llm-lifecycle) (young: 0 stars, created 2026-07-08) | Concepts: LLM lifecycle | Pre-train → SFT → RLHF/DPO/GRPO → int8 deploy | Partial | Per-stage token distributions | Precomputed + int8 ONNX live | none | One "persistent probe" followed through every stage; size-budgeted replay JSON; schema-to-TypeScript contract. Its "nothing is faked" claim comes only from its README |
| 39 | [Teachable Machine](https://teachablemachine.withgoogle.com/) | Concept: transfer learning | MobileNet features + trainable head (image, audio, pose) | Yes: webcam, uploads, microphone, your own classes | Model | In-browser | Apache-2.0 (community repo; app source unverified) | Train on your own data in seconds. Adding a layer view would turn it into a lesson |
| 40 | [AMU applet collection](https://bnaskrecki.faculty.wmi.amu.edu.pl/nnets/applets/) | Integrated curriculum | Perceptron → CNN → RNN/LSTM → Transformer → BERT/GPT, tokenizers | Yes (varies by applet) | Weight / gate / head | In-browser (ORT Web, TF.js) | unverified | Tiny from-scratch models (<1 MB ONNX) that match PyTorch exactly; "click a character to mask it" |
| 41 | [LIT, Learning Interpretability Tool](https://pair-code.github.io/lit/) | Multi-area (text; preliminary image and tabular) | Classification, regression, seq2seq, masked LM, span labeling, multi-head models | Yes: your own datapoints; counterfactual generation | Model, plus salience, attention, embeddings, TCAV | Python server or notebook (Colab/Jupyter); hosted demos (Palmer Penguins tabular, GLUE BERT) | Apache-2.0 (3,669 stars; pushed 2026-10-05) | The closest "one UI, many models, your own input" tool. Borrow datapoint and model comparison, the counterfactual editor and the salience/embedding linking. Its GLUE BERT demo is a link target for the builder's MRPC/RTE work |
| 42 | [TensorSpace](https://tensorspace.org/) | Vision / generative | LeNet, AlexNet, YOLOv2-tiny, ResNet-50, ACGAN | unverified (demos use presets) | Layer (3D) | In-browser (TF.js + Three.js) | Apache-2.0 (5,198 stars; last push 2022-12-05) | An early "many pretrained models, one layer grammar" precedent, now unmaintained. 3D looks impressive but hides values; our 2D grammar should beat it on legibility |
| 43 | [NNvisual](https://j4den.com/nnvisual/) (young: 0 stars, created 2026-08-07) | Foundations / multi-family | Site lists Transformer, CNN, MLP/perceptron, diffusion; repo describes "build a small MLP and watch it learn" | unverified | unverified | In-browser, vanilla JS, zero dependencies; site claims "real models running in your browser" | MIT | A single-page family switcher. Re-check its depth before any novelty claim |
| 44 | [Quantization from the ground up](https://ngrok.com/blog/quantization) (Sam Rose, 2026-03-25) | Concept: compression | fp32/fp16/bf16/fp8/fp4 formats; symmetric vs asymmetric; block quantization; outliers | Partial: bit-level sliders and controls | Per value | In-browser | none stated (ideas only) | Bit-level format widgets, an outlier that breaks the scale, before/after next-token distributions, and evaluation by perplexity, KL divergence and benchmarks side by side |
| 45 | [Prompt caching](https://ngrok.com/blog/prompt-caching/) (Sam Rose, 2025-12-16) | Concept: inference | Attention, embeddings, KV cache, prefix reuse | Partial: hover/click per cell | Per token | In-browser | none stated (ideas only) | An attention grid that fills as tokens are generated; cached K/V shown as skipped matrix work; a 1D → 3D embedding toggle |
| 46 | [MLSysBook Playground](https://mlsysbook.ai/games/) (Harvard) | Concepts: training systems, efficiency | 14 games, e.g. Tensor Tetris (training memory), Pulse Prune, Quantization Sharp Shot, MoE Router, Gradient Lander (batch size vs LR), KV Cache Packer, Roofline Rider, All-Reduce Rhythm | Yes (gameplay) | Concept | In-browser | CC BY-NC-SA 4.0 (ideas only) | Turn a constraint (memory, bandwidth, precision) into a game with a fail state |
| 47 | [AI Notes: Initialization](https://www.deeplearning.ai/ai-notes/initialization/) (+ Optimization note) | Concept: initialization | Zero / too small / appropriate / too large init; Xavier, uniform and normal on MNIST | Partial: init choice, depth, layer width, activation | Layer (weights, gradients, cost) | In-browser (TF.js + D3) | © deeplearning.ai 2021 (ideas only) | Pick an init and watch vanishing or exploding gradients, the cost curve and the decision boundary; a 5-layer MNIST net trains in the page |
| 48 | [Google ML Crash Course exercises](https://developers.google.com/machine-learning/crash-course/exercises) | Curriculum | Regression, classification, neural nets, embeddings (word2vec), intro to LLMs incl. fine-tuning, distillation and prompt engineering | Partial (in-page exercises) | Concept / model | In-browser | unverified | The benchmark curriculum learners will compare us with; it has a Simplified-Chinese version (developers.google.cn). Link target for /paths |
| 49 | [nanochat](https://github.com/karpathy/nanochat) | Concept: LLM lifecycle (offline) | Tokenizer training, pre-training, SFT, RL, evaluation, chat | n/a (Python) | — | Precompute (PyTorch; single GPU works via gradient accumulation; `--depth` and `--device-batch-size` shrink it) | MIT (58,431 stars; pushed 2026-09-07) | One codebase, one tokenizer and one architecture through every stage: real checkpoints for the lifecycle track |
| 50 | [ManimML](https://github.com/helblazer811/ManimML) | Animation tooling | Manim animations of CNNs, VAEs and more | No | Layer | Offline video | MIT (3,513 stars; last push 2024-06-22) | Generate reduced-motion stills and article figures from the same model spec; fits the builder's animated-slide workflow |
| 51 | [LLM-RL-Visualized (大模型算法图解)](https://github.com/changyeyu/LLM-RL-Visualized) | Concepts: LLM, RL, alignment (Chinese) | 100+ original diagrams (SFT, DPO, PPO, GRPO, decoding, …) | No | Diagram | Static images | "Other" (NOASSERTION; terms unverified; ideas only); 4,933 stars, pushed 2026-09-10 | Chinese terminology and visual vocabulary for the alignment track |

Also relevant but not in the table:
- Neuronpedia (link-out target for interpretability).
- The YOLO "10647 Times" visualization (GPL-3.0, ideas only).
- Distill GNN intro (Apache-2.0 code); Explained Visually PCA (MIT); Understanding UMAP (Apache-2.0).
- tfjs-examples snake-DQN; Q-Learning Lab.
- HF Chat Template Playground; rasbt/LLMs-from-scratch; ESM Atlas fold API.
- Chrome Music Lab Spectrogram; Distill CTC.
- d2l.ai; labml.ai; Netron; Model Explorer; the Asimov Neural Network Zoo.
- Keras.js (current status unverified).
- deep-learning-visualized / 图解深度学习 (see §3).

---

## 2. What makes the best ones work

1. **Semantic zoom with context kept.** Go from overview to layer to unit to arithmetic without leaving the page.
   - CNN Explainer: overview, then elastic layer view, then kernel math.
   - Transformer Explainer: Sankey overview, then block, then head, then a matrix-multiply popover.
   - GNN 101: four explicit levels.
   - BertViz: model, head and neuron views.
   - Model Explorer: collapsible module hierarchy.
2. **The learner's own input drives a real forward pass.**
   - Upload an image: CNN Explainer, WebGPU ViT.
   - Type a prompt: Transformer Explainer.
   - Speak: Speech Commands, Whisper Web.
   - Click prompts: SAM.
   - Draw the data itself: GAN Lab, Diffusion Explorer.
   - Bring your own classes: Teachable Machine.
   - Edit your own datapoints: LIT.

   Breadth without real input (toy-only multi-family sites, AI-generated "lab" sites) teaches far less.
3. **Bidirectional hover-linking.**
   - Output cell ↔ input window and kernel: CNN Explainer, ezyang's Convolution Visualizer.
   - Token ↔ attention row: Transformer Explainer; ngrok's prompt-caching grid.
   - PAE region ↔ 3D residues: AlphaFold DB + Mol*.
   - Query line ↔ heatmap tile: Sequence Toy.
   - Equation symbol ↔ same symbol in other equations: labml.
   - Datapoint ↔ embedding ↔ salience: LIT.
4. **Instant first, live later.** Show precomputed examples immediately, switch to the live model once it is cached, and serve precomputed only on phones.
   - Transformer Explainer: 5 cached runs while 626 MiB downloads.
   - LLM Attention Visualizer: precomputed by default, live generation opt-in.
   - AnimatedLLM, Diffusion Explainer, Crucible (young).
5. **Step players and slow motion tied to the algorithm's steps.**
   - GAN Lab: 5 numbered D steps and 5 G steps, 1,250 ms each.
   - CNN Explainer: convolution auto-play, scrubbed by hovering.
   - Autograd UI: one chain-rule edge per click.
   - Diffusion Explorer: a time slider where each frame is a pure `stateAt(t)`.
6. **Equations with the real numbers filled in, colored like the diagram.**
   - CNN Explainer KernelMathView.
   - Backprop Explainer (w := 1 − 0.01·(−7.98)).
   - Q-Learning Lab.
   - Distill GNN "click a node to see its update equation".
   - labml symbol highlighting.
   - ngrok quantization's bit-level float widgets.
7. **Train it yourself, live, on a tiny model that finishes in seconds.**
   - TF Playground, ConvNetJS, wevi, GAN Lab, Diffusion Explorer.
   - TF.js cart-pole (769 parameters), Sequence Toy, Teachable Machine.
   - deeplearning.ai's initialization note (MNIST in the page).
   - nn.ameo.dev and Diffusion Explorer train in a Web Worker to keep the UI smooth.
8. **Contrast and counterfactual pairs.**
   - Erase a region and re-classify: Interactive Classification.
   - Generate and compare counterfactual datapoints: LIT.
   - Prompt pairs differing by one keyword: Diffusion Explainer, PAIR essays.
   - Same image explained as "dog" vs "cat": pytorch-grad-cam.
   - GRU vs LSTM on the same sentence: Distill memorization.
   - Human vs sampled text: GLTR.
   - 1/2/4/8/16 Euler steps: Rectified Flow explainer.
   - Model before vs after quantization: ngrok.
9. **Direct manipulation of internals.**
   - Edit a weight: TF Playground.
   - Drag weights by hand: Backprop Explainer.
   - Turn units on and off: GANPaint.
   - Steering sliders: Neuronpedia.
   - Drag the latent point: VAE Explainer.
   - LoRA scale as a slider: Concept Sliders.
   - Flip individual bits: ngrok quantization.
10. **A guided tour plus free play.**
    - Transformer Explainer: 20 textbook pages with on/out hooks.
    - ViT-Explainer: guided and free modes.
    - bbycroft: narrated walkthrough.
    - GNN 101: intro.js tours.
    - Sequence Toy: "You could try…" preset links.
11. **Tool and article linked both ways.** CNN Explainer's (i) buttons, Distill's interleaved figures, MLU-Explain's pinned scrollytelling, labml's side-by-side math and code.
12. **Shareable state.** TF Playground (URL hash and hide flags), Tokenizer Playground (`?tokenizer=&text=`), Sequence Toy (dotted config params), Chat Template Playground (`?modelId=&template=`).
13. **Honesty about what is real.**
    - Crucible's (young) "nothing is faked" labels, as an idea.
    - Anti-pattern: Transformer Explainer's `Math.random()` vectors.
    - Caution: the Virtual Labs transfer-learning curves are simulated.
    - Good practice: micrograd's tests against PyTorch, AMU's bit-exact ONNX, Steerscope's logit-lens tests.
14. **One visual vocabulary across many models.**
    - Asimov Zoo: a legend of about 10 glyphs.
    - Jay Alammar: colored vector strips.
    - Raschka gallery: uniform data-driven cards.
    - Polo Club: one design language.
    - d2l: progressive figure sequences (lstm-0 … lstm-3).
    - TensorSpace: one 3D layer grammar for 5 models.
15. **A visible shape and parameter budget.** d2l prints shapes per layer; ConvNetJS prints parameter counts per layer ("8x5x5x1+8 = 208"); the Speech Commands dense layer holds 96% of the parameters; bbycroft compares sizes at true scale.
16. **Constraints made playable.** MLSysBook's games (memory Tetris, pruning, per-layer precision) and Kingy's "parameters are not VRAM" turn a budget into a fail state the learner can hit.

---

## 3. Gaps, and why an integrated site is worth building

**Close to our idea already (be honest):**
- **Polo Club family** (CNN, Transformer, Diffusion explainers and GAN Lab).
  - The closest in quality, but separate apps with no shared catalog, covering four families.
  - For GPT-2, Transformer Explainer already does most of what we would do. Link to it or fork it rather than compete, and set ourselves apart with a tiny exact model, training, and concept links.
  - Its stack is not drop-in. CNN Explainer is Svelte 3 + Rollup 1 + TF.js 1.4. Transformer Explainer is Svelte 5 / Kit 2 but pins `@xenova/transformers` ^2.17 and `onnxruntime-web` ^1.23.
- **LIT (Google PAIR).**
  - The closest to "one UI, many models, your own input". Text, preliminary image and tabular support; salience, attention, embeddings, counterfactuals, model and datapoint comparison, TCAV.
  - It needs a Python server or notebook. It targets practitioners debugging their own models rather than novices learning how each architecture computes, and it does not animate per-layer arithmetic.
  - We reuse its comparison and counterfactual patterns and link to its GLUE BERT demo.
- **TensorSpace** (and Keras.js, status unverified). Several pretrained models (LeNet, AlexNet, YOLOv2-tiny, ResNet-50, ACGAN) in one 3D layer grammar, in the browser. The site is online, but the repo has not been pushed since 2022-12-05, and it covers only vision and one GAN.
- **NNvisual** (young). A 2026 vanilla-JS site whose homepage lists Transformer, CNN, MLP and diffusion "running in your browser"; the repo description mentions only an MLP builder. Its depth and custom input are unverified. It is a direct small analogue, so re-check it before launch claims.
- **AMU applet gallery.** One curriculum from perceptron to BERT/GPT, with small ONNX models in the browser and a final test. Its weaknesses:
  - no published license or source;
  - mixed fidelity (the Bahdanau applet is simulated analytically);
  - foundations and language only;
  - applets are separate pages.
- **deep-learning-visualized (图解深度学习).**
  - Created 2026-09-20 (2 stars). The nearest Chinese analogue: 9 stages and 77 chapters per the site (79 per the repo description), with slider widgets and a strict chapter scaffold.
  - No trained models and no custom input.
  - Licensing is conflicting. The site says code MIT and content CC BY-NC-SA 4.0, but the repo LICENSE file is CC BY-NC-SA 4.0 only (GitHub: NOASSERTION). Treat content as ideas only, and ask the author before reusing any code.
- **Google ML Crash Course.** The mainstream curriculum, with in-page exercises (neural nets, word2vec embeddings, LLM fine-tuning and distillation) and a Simplified-Chinese version. It is the bar our bilingual paths will be compared with. Link to it; do not duplicate it.
- **Gradient's "Learn ML Visualized Interactive"** (234 chapters). Toy problems, author unknown, cold-start 503s.
- **Neural-Visualizer.** Same tabs for every family, but toy networks only, and GPL-3.0.
- **d2l.ai and labml.ai.** Excellent text and code, not interactive.
- **MLU-Explain** (archived 2024-11-12), **PAIR Explorables** and **Distill.**
  - Beautiful essays, but not on your own data.
  - Distill paused in 2021. Its editorial names volunteer-editor burnout, mentorship overhead (editors often spent 50+ hours improving a single article), and "the amount of effort it takes to produce these articles and the unusual combination of scientific and design expertise required", plus a belief that self-publication is the better model.
  - Lesson for us: per-page production cost must be driven down by shared components.

**What actually sets us apart** (integration alone is not novel):
- (a) breadth across 11 application areas in one grammar, on the learner's own input;
- (b) a Training Concepts track that is interactive, not a set of calculators;
- (c) provenance honesty with precision and tolerance on every panel;
- (d) zh/en;
- (e) static durability.

**Gaps nobody fills well:**
1. **One site that spans application areas** with one grammar and one navigation, at novice level and statically hosted. LIT spans modalities, but it is a server tool for interpretability practitioners. TensorSpace spans a few vision models and is unmaintained.
2. **Areas with no CNN-Explainer-style, own-input, per-layer explainer:**
   - speech (Whisper, wav2vec 2.0, TTS internals, neural audio codecs);
   - time-series Transformers and TCNs;
   - recommender systems;
   - object detection;
   - segmentation internals;
   - CLIP and multimodal;
   - U-Net;
   - normalizing flows;
   - Mamba/SSM and MoE with custom input;
   - deep-RL network internals and AlphaZero;
   - a static browser BERT explainer (exbert.net is down; BertViz needs a notebook; LIT needs a server);
   - a real small seq2seq or encoder-decoder Transformer on typed input;
   - 3D vision, video, robotics/embodied policies.
3. **The training-concepts track.**
   - **Already well covered; we link and extend:** tokenization and next-token prediction. Number formats and quantization (ngrok, 2026). Initialization (deeplearning.ai AI Notes). KV cache / prompt caching (ngrok). Systems constraints as games (MLSysBook, NC license).
   - **Weak or absent:**
     - next-sentence prediction;
     - vision self-supervised learning (MAE, DINO, SimCLR; existing Spaces are broken or asleep);
     - scaling-law sweeps and data mixtures;
     - catastrophic forgetting;
     - layer-wise LR decay;
     - adapters, BitFit, prefix/prompt tuning;
     - LP-FT and feature distortion;
     - multi-task and intermediate-task training;
     - in-context learning vs fine-tuning;
     - evaluation (contamination, calibration, pass@k);
     - interpretability methods (patching, probing, SAEs) on a model the learner just used.
   - Existing fine-tuning tools are analytic calculators (Kingy, gpu_poor), toy simulations (dev48v, alignment-lab, Virtual Labs) or precomputed replays (Crucible). Only Teachable Machine and Speech Commands fine-tune a real network on the learner's own data, and only for classification heads.
4. **Links between models and concepts.** No site connects "this LayerNorm in GPT" to a LayerNorm concept page, or "masking" across BERT, MAE, wav2vec 2.0 and ESM-2.
5. **Hybrid fidelity done honestly.** Most tools are either live but tiny, or real but frozen. No site combines both with clear provenance badges.
6. **Durability.**
   - Server-backed tools die: OpenAI Microscope (DNS gone), exbert.net, the LSTMVis live server, seq2seq-vis.io, gandissect, GLTR's old `/dist/`, the exploRNN site.
   - Static tools survive: TF Playground, ConvNetJS, Distill, wevi.
   - Archived repos stop getting fixes: MLU-Explain.
   - Curated lists such as ML-Tokyo Interactive_Tools carry dead links.
7. **Bilingual (zh/en) and mobile-first.**
   - Chinese material mostly lists the English tools; deep-learning-visualized and LLM-RL-Visualized are exceptions without live models.
   - Most explainers are desktop-only: GAN Lab has no touch support, CNN Explainer collapses at 800 px, and Transformer Explainer skips the model on mobile.

---

## 4. Recommended product

Working title: "Model Atlas / 模型图鉴". Check for name collisions before launch, since "Weight Atlas" and "Kingy.ai Atlas" exist.

### 4.1 Information architecture

```
/                    Home: input launcher + interactive "zoo map" + learning paths
/atlas/              Atlas by application area (11 areas)
/atlas/<area>/       Area page: models in tiers, "start here", prerequisites, planned cards
/m/<model-slug>/     Model page (fixed template, §4.2)
/train/              Training Concepts track (ordered sections, §6)
/train/<slug>/       Concept page (fixed template, §4.4)
/compare/            Comparison views (§4.6), deep-linkable
/glossary/           Bilingual glossary (single source of terminology for zh/en and graph.json)
/paths/              Curated paths (d2l order; "LLM lifecycle"; "vision"; "audio"; links to MLCC)
/about/              How numbers are computed (live / precomputed / illustrative, precision, tolerance), licenses, credits
```

**Home.**
- **Input launcher** (formerly "universal input box"). Type, draw, speak or drop a file, and the site lists every model that accepts that input, each with a precomputed preview thumbnail and its exact preprocessing (e.g. "TinyCNN sees 64×64 RGB; TinyCLIP sees 32×32 grayscale"). It loads **only** the model the learner picks: one drawing must never trigger four model downloads.
- **Interactive zoo map.** Asimov-style glyph diagrams of each family; click one to open its page.
- **MLU-style card gallery** grouped by area, with a one-line hook and a status badge per card: Live, Precomputed, or Planned.
- **Learning paths.** Follow the d2l chapter order for foundations, then branch by area.

**Atlas.** Eleven areas. Some names are widened so that new families fit without new areas:
1. vision (images, video, 3D)
2. language
3. speech/audio
4. time series
5. recommendation
6. tabular/classic ML
7. reinforcement learning and control (incl. robotics/embodied)
8. graphs
9. generative
10. multimodal
11. science

Each card shows the input type, provenance mode, download size and prerequisites.

**Linking models and concepts.** A single `graph.json` knowledge graph:
- **Nodes:** models, concepts, glossary terms, external references (d2l zh/en section URLs, labml pages, papers, MLCC units).
- **Typed edges:** `prerequisite`, `uses-component` (e.g. `tiny-gpt → layernorm`), `trained-with` (e.g. `tiny-bert → masked-lm`), `compares-to`, `fine-tuned-from`, `implemented-in`.

Pages render this as:
- concept chips on the exact layer card that uses them;
- "See this concept in…" lists on concept pages;
- "Before this, read…" banners.

### 4.2 Per-model page template (same order on every page)

1. **Learning goals and pre-question.** 2–3 goals and one "predict first" question, answered again at the end (§9 study plan).
2. **Fact strip:** area, family, exact model, parameter count, download size, runtime, input type, numeric precision, and a provenance badge:
   - **Live**: computed in your browser now, with precision and parity tolerance, e.g. "Live · fp32 · max abs diff vs PyTorch ≤ 1e-5" (design target).
   - **Precomputed**: computed offline from the same model.
   - **Illustrative**: made-up numbers, used rarely and always labeled.
   - Quantized runs are labeled "approximate", never "exact".
3. **Input bar** for the modality, with presets in both zh and en (§4.5).
4. **Overview graph,** generated from a model spec JSON (§7) rather than hand-placed.
5. **Shared layer inspector** (§4.3), opened by clicking any node.
6. **Unit/arithmetic view:** formula card with live numbers and an operation animator (auto-play, plus hover to scrub). Shown at full depth for own tiny models. For Hub models it appears only where we re-exported the needed internals (§7).
7. **Output panel:** probabilities, boxes, masks, audio or samples. Cheap post-processing (temperature, thresholds, NMS) is recomputed without re-running the model.
8. **"Try this":** 2–4 guided experiments, each a URL that sets state and highlights the control, with "predict first, then check" prompts and one misconception check.
9. **Training tab,** where feasible: "train it yourself" for toy models, or a checkpoint replay.
10. **Article:** scrollytelling below the tool, linked by (i) buttons both ways.
11. **Code tab:** labml-style aligned PyTorch lines for the highlighted node, plus deep links to d2l (zh and en) and labml.
12. **Concepts used here, Compare with, Next.**
13. **Checkpoint:** the post-question plus 1–2 questions answered by finding a value in the widget. Credits and licenses.

### 4.3 Shared layer inspector (one component, every model)

A drawer with tabs, stepped with `[` and `]` (ActLens) or arrow keys:
- **Values:** a tensor view (grid, strip or tile stack) with a color-scope switch for Unit, Layer, Global or z-score (CNN Explainer and Sequence Toy), and a "show true grid" toggle for upsampled maps.
- **Stats:** histogram, mean/std/min/max, sparsity, shape.
- **Math:** the operation that produced this tensor, with live numbers.
- **What it responds to** (precomputed): feature-visualization icon (Lucent), top dataset patches, top audio clips, top text snippets.
- **Compare:** the same tensor for a second input, a second model or a second checkpoint, with a diff mode on the diverging scale. Datapoint-comparison and counterfactual workflow adapted from LIT.
- **Code:** the line that computed it.

### 4.4 Concept page template

1. Learning goals and pre-question.
2. Hook question.
3. **Toy:** a simulation where every variable is visible.
4. **Real:** apply the idea to a model page, e.g. fine-tune the KWS model you just used.
5. **Compare:** side-by-side runs.
6. **Pitfalls and misconceptions:** e.g. "attention is not explanation", "CAMs need sanity checks", "parameters are not VRAM".
7. Post-question.
8. **Links** to models and further reading (ngrok, deeplearning.ai, MLCC, d2l, labml, LIT, Neuronpedia as relevant).

### 4.5 Custom input, per input type

| Input type | Capture | Shown as "layer 0" (exactly what the model receives) | Fallback / presets | Notes |
|---|---|---|---|---|
| Text | Textarea; tokenized live in a worker on every keystroke | Token chips (alternating colors), IDs, bytes for byte-level BPE, special tokens added by the post-processor | zh/en presets; `?text=` deep link | Per-model length caps (e.g. TinyGPT context 64). Hover for ID, bytes and vocab rank. A Chinese preset is mandatory (e.g. 我爱自然语言处理) |
| Image | Upload, drag-drop, paste, URL (with CORS check and a clear error), webcam | Crop/resize/normalize preview at the true input size (e.g. 64×64, 224×224) | Images we own or CC0 (not Tiny ImageNet) | Test images smaller than the input size (CNN Explainer has an unfinished branch there). Images never leave the device; say so |
| Drawing | Pointer-event pad with touch and pen support | The exact training preprocessing (invert, center, resize, crop) shown as a tensor | Preset strokes | ConvNetJS's upload path skipped grayscale and cropping; do not repeat that |
| Audio | Microphone (record or stream), file upload (decode and resample), bundled clips | Waveform, then the exact spectrogram or log-mel the model sees (e.g. 43×232 for KWS) | A no-microphone path is mandatory; self-recorded presets | Spectrogram as step 0 for every audio page (Chrome Music Lab idea, reimplemented) |
| Numbers / time series | Draw a curve on a canvas, paste CSV, bundled series | Windowed tensor with highlighted input and target windows | Sine/ECG/seasonal presets | Avoid external API keys (the jinglescode demo needs one) |
| Tabular | Form with ranges and defaults; "random row" button | Normalized feature vector | Dataset rows (license checked) | Highlight the tree path or feature contributions |
| 2D points | Shared Playground2D: brush spray (GAN Lab, Diffusion Explorer) and click-to-label (ML Playground) | The point set and its train/test split | Moons, circles, spiral, smiley, three modes | One engine reused by MLP, classic ML, GAN, VAE, flow, diffusion, flow matching, forgetting, federated toys |
| Graph | Node-link editor, editable adjacency matrix (Distill GNN) | Feature matrix X and adjacency A | Molecule presets | Node-link and matrix views stay in sync |
| Game state | Playable board: bandit arms, gridworld editor, tic-tac-toe, cart-pole, snake | State vector or board tensor | Preset maps | Step and auto-play; agent vs you; reset to any state |
| Protein sequence | Amino-acid text with validation; click to mutate or mask | Residue tokens | Sample proteins | Third-party fold API with a cached fallback |
| Ratings / preferences | Rate ~10 items; choose A vs B | User vector / preference pair | Preset user profiles | Feeds the recommendation and reward-model pages |
| Conversation | JSON5 messages editor | Rendered chat template, then tokens with loss mask | Chat presets | SFT page (§6) |

### 4.6 Comparison views

- **Same input, two models:** tokenizers side by side; TinyCNN vs TinyViT on one drawing; GRU vs LSTM connectivity; CNN CAM vs ViT rollout.
- **Same model, two inputs:**
  - prompt or sentence pairs (PAIR, Diffusion Explainer);
  - images with and without a region erased (Interactive Classification);
  - LIT-style generated counterfactuals (swap a word, perturb a feature);
  - clean vs adversarially perturbed drawing.
- **Same model, two training states:** base vs fine-tuned vs LoRA-merged; an epoch scrubber; the same probe prompt carried through pre-training, SFT and RL/DPO (nanochat checkpoints; Crucible's probe idea).
- **Same model, two precisions:** fp32 vs q8 vs q4, comparing activations, outputs, size and speed, plus perplexity and KL divergence (ngrok idea).
- **Architecture diffs:** Raschka-style cards (GPT-2 → Llama-style: RoPE, GQA, MoE) generated from the same spec format.

All comparisons use the same diff encoding: the diverging scale applied to the difference, plus a numeric summary.

### 4.7 One visual grammar (drawn the same everywhere)

| Element | Encoding | Source of convention |
|---|---|---|
| Flow direction | Left to right on desktop, top to bottom on phones. Forward pass is solid edges with flowing dashes; backward pass runs right to left with dashed edges | TF Playground dashes; GAN Lab flow paths |
| Vector | Horizontal strip of cells; long dimensions elided with "… 768" and an exact shape label like `[T×768]` always visible | Alammar strips; Transformer Explainer labeled scale; d2l shape printouts |
| Matrix / feature maps | Grid; a CNN feature map is a stack of tiles; a sequence is one column per token, headed by its token chip | CNN Explainer; BertViz |
| Signed values (activations, weights, gradients, diffs) | One diverging scale centered at 0: blue negative, neutral midpoint, orange positive; perceptual (OKLCH) and colorblind-checked; never jet | Sequence Toy's z-scored diverging map; avoid pytorch-grad-cam's jet |
| Non-negative values (probabilities, attention, magnitudes, densities) | One single-hue sequential ramp (purple) | Transformer Explainer probability color |
| Tensor role | Activation: plain cells on the data path. Parameter: cells in a double-bordered slab with a **W** chip. Gradient: dashed outline with a **∇** chip, placed next to its tensor. Optimizer state: small **m, v** badges. Cached tensor (KV cache): slab with a **⟲** chip | llm-viz weight vs activation colors; ConvNetJS four views; lilipads optimizer-state glyphs; ngrok KV-cache view |
| Training status | Frozen: desaturated with a lock. Trainable: accent outline. LoRA/adapter: a thin side branch labeled with its rank. Quantized: stepped edge with a bit-width badge | Virtual Labs gray-frozen convention; Kingy LoRA detour |
| Attention | Query rows × key columns; causal mask as a hatched upper triangle; heads in a layers × heads grid; Q/K/V frames always blue/red/green (frames only, never cell fill) | BertViz model view; Transformer Explainer Q/K/V colors |
| Outputs | Top-k horizontal bars; the chosen token highlighted; temperature, top-k and top-p cutoffs drawn as lines | Transformer Explainer; sampling explorable |
| Classes | One fixed 10-color categorical palette; class k has the same color in predictions, scatter plots and decision regions | — |
| Real vs generated | Real data in green, generated samples in magenta (purple is reserved for sequential values) | Adapted from GAN Lab |
| Spatial overlays (CAM, attention on patches, masks, adversarial perturbation) | Translucent overlay with an opacity slider and a "true grid" toggle | tfjs-gradcam opacity slider; pytorch-grad-cam lesson |
| Numbers | Monospace, 3 significant digits; full precision on hover, tap or focus | — |
| Formulas | KaTeX prerendered at build time; each symbol takes the color of its tensor in the diagram | Better than labml, which links equations only to other equations |
| Provenance | Every panel carries Live, Precomputed or Illustrative, plus precision (fp32/fp16/q8/q4) and, for Live own models, the parity tolerance | Crucible idea; extended |
| Motion | Every animation is a pure `stateAt(t)` function, so play, scrub and screenshot agree; with reduced motion it becomes a step list (stills can be rendered offline, e.g. with ManimML or Playwright) | Diffusion Explorer's Tempus timelines |

---

## 5. Model catalog by application area

Tiers:
- **P0** = MVP.
- **V1** = flagship that completes v1 (one per area).
- **P1** = next after v1.
- **P2** = later.

P1 and P2 pages are **planned**. They appear as cards with a "Planned" badge, and each is built only when the shared kit already covers about 90% of it.

"Own" means a tiny model we train on a single GPU; its parameter counts are design targets. Hub model sizes come from the evidence or HF trees checked 2026-10-05 unless marked unverified. Before shipping any Hub model, check its model-card license.

Hub models are shown at **model level live, with precomputed internals,** unless we re-export them with extra outputs and self-host them (§7). Per-neuron depth is reserved for own tiny models.

### Vision (images, video, 3D)

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| P0 | **TinyCNN** (own, CNN Explainer TinyVGG layout: 4 conv 3×3×10, 2 max-pool, dense 10; 64×64×3) | ≈20k params (derived from 79.7 KB fp32) | Own JS forward pass from binary weights, checked against PyTorch and ORT Web | Drawing pad, upload, webcam, presets | Crop preview → each conv (kernel slides with live arithmetic, per-channel intermediate sums + bias) → ReLU → pool → flatten (index mapping) → softmax | All live. Precompute per-channel feature-visualization icons and top dataset patches (Lucent), plus a per-class "how the class is built" graph (Summit idea) |
| P1 | **Adversarial examples on TinyCNN** (FGSM/PGD) | — (reuses TinyCNN) | Own JS (input gradient in the autograd core) | Your drawing | ε slider; the perturbation amplified as an overlay; logits moving until the class flips; PGD steps on the step player | Live |
| P1 | **ResNet-18** (HF ONNX port; Xenova/resnet-50 is the larger option) | ≈11.7M (derived from 47 MB fp32); q8 12 MB. ResNet-50: 25.8 MB quantized | ORT Web or Transformers.js | Upload, webcam | 4 stage feature maps, residual add, global average pool; live CAM at the last block (exact for a GAP + linear head: ∂logit/∂A = W[c,k]/(H·W)); layer slider. Needs a re-export with stage outputs | Live forward + CAM. Precompute Grad-CAM++, ScoreCAM, LayerCAM, EigenCAM per preset image × layer × top-3 classes (pytorch-grad-cam), stored as raw low-resolution uint8 maps |
| P1 | **TinyViT** (own: 3×3 patch grid + CLS, 4 layers, d=64, 4 heads; same classes as TinyCNN) | Design ≈0.2M params, <1 MB | Own JS / ORT Web | Same inputs as TinyCNN (side-by-side CNN vs ViT) | Patchify (shown as a conv whose stride equals its kernel), position embeddings, Q/K/V per head, attention on the patches, MLP, CLS logit lens per layer | All live |
| P1 | **DINOv2-small with registers, with-attentions** (onnx-community export; its base_model is `facebook/dinov2-with-registers-small-imagenet1k-1-layer`, i.e. with an ImageNet linear head) | ≈23M (derived from 91.7 MB fp32); q4 20.0 MB, q8 25.1 MB, q4f16 15.5 MB | Transformers.js worker (WebGPU, falling back to WASM) | Upload, drag-drop | CLS → patch attention per layer and head; attention rollout computed in JS; PCA of patch features; top-5 ImageNet classes from the extra head | Live; precompute gradient rollout for presets. ONNX repo declares no license; upstream Facebook card license unverified (check before self-hosting) |
| P2 | **DeiT-Tiny** (WGSL, BSD-3 project, young) as a no-ONNX alternative | ~5 MB weights | Raw WebGPU | Image | Layer and head sliders | Live |
| P2 | **SlimSAM-77-uniform** | ≈10M (derived from 39.9 MB fp32); fp16 20.7 MB; quantized 13.8 MB | Transformers.js | Upload, positive/negative points, boxes (`input_boxes` supported) | Per-stage timers; 64×64×256 embedding as a PCA image and click-similarity heatmap; all 3 candidate masks with IoU; 256×256 logits with a threshold slider | Live |
| P2 | **Detector:** Xenova/yolos-tiny, or an own tiny YOLO-style model exported with raw per-scale grids | YOLOS-tiny ≈6.6M (derived from 26.2 MB fp32); fp16 13.3 MB; q8 9.66 MB; q4 7.81 MB / own design ≤5 MB | ORT Web / Transformers.js | Upload, webcam | Grid cells as small classifiers, objectness per scale, all boxes before NMS, animated NMS with IoU and confidence sliders | Live for the own model; precompute saliency per layer |
| P2 | Planned backlog: tiny U-Net / FCN segmentation (own); monocular depth (Hub model, choice and size unverified); OCR (own CRNN + CTC); neural style transfer; super-resolution; pose estimation; video (tiny 3D CNN vs frame-wise CNN); 3D (PointNet on drawn point clouds; a 2D "NeRF on a line/plane" toy; Gaussian-splat toy) | Design ≤5 MB each for own models | Own JS / ORT Web | Photo, webcam, drawing, point cloud | Per-task heads over a shared backbone view | Live (own) / precomputed (Hub) |

### Language

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| P0 | **Tokenizers:** `@huggingface/tokenizers` 0.2.0 (Apache-2.0; 361 KB unpacked on npm; 36.6 KB min+gz, unverified) + tokenizer.json files: bert-base-cased 669 KB, claude 1.77 MB, t5-small 2.42 MB, gpt-4 (cl100k) 4.23 MB, llama3 9.08 MB | — | Worker | Text | Normalizer → pre-tokenizer → model (merges) → post-processor; UTF-8 byte view | Live (lazy-load each tokenizer.json, show its size) |
| P0 | **TinyGPT** (own, character level, 2 layers, 4 heads, d=64, context 64, bilingual corpus with verified license) | Design ≈0.1M params, ≈0.4 MB fp32 | Own JS forward pass, checked against PyTorch and ORT Web | Text | Embedding row lookup, positions, LayerNorm, Q/K/V, scores, causal mask, softmax, weighted V, output projection, residual, MLP, logits, sampling. Every number is computed live | Live. Precompute training checkpoints for a training replay |
| P0 (data) / P1 (live) | **GPT-2 small** (124M) | fp32 500 MB; fp16 251 MB; q8 128 MB. distilgpt2: q8 84.9 MB, fp16 165 MB | Precomputed traces; optional opt-in live via Transformers.js (HF-hosted) | Curated prompts (+ custom when live) | The TinyGPT diagram at real scale (768-d, 12 layers, 12 heads); link to Transformer Explainer | Precompute on a GPU (hooks), uint8 attention |
| P1 | **TinyBERT** (own; same corpus and size as TinyGPT; MLM, with 80/10/10 masking) | Design ≈0.1M | Own JS / ORT Web | Text; click a token to mask it | Bidirectional vs causal attention; [MASK] prediction; segment embeddings | Live |
| P1 | **BERT-base** (110M) including the builder's own MRPC/RTE checkpoints | fp32 438 MB / q8 110 MB / q4f16 96 MB | Precomputed traces in BertViz layout | Curated sentence pairs, including the builder's slide examples | Layers × heads grid; neuron view extended past softmax (×V, output projection, residual, LayerNorm, FFN); [CLS] head; A→B / B→A filters | Precompute only. Read `outputs.attentions` with eager attention; store uint8. Link to LIT's GLUE BERT demo |
| P1 | **word2vec** (own skip-gram, wevi-style, V≈20–50, d=5–10) | <10 KB | Own JS training | Your own mini-corpus | Both weight matrices updating, softmax, PCA | Live training. Optional GloVe 50-d vectors for custom axes (word2viz pattern; GloVe license unverified) |
| P1 | **Character RNN / GRU / LSTM** (own, hidden 32–64) | Design ≈10–40k | Own JS (in-browser training feasible, as RecurrentJS shows) | Text | Unrolled steps, gate bars, cell-state line, hidden-state heatmap under the text, input-connectivity saliency | Live |
| P1 | **Sentence embeddings and retrieval** (small Hub bi-encoder such as MiniLM-L6, size unverified; plus an own toy cross-encoder) | unverified | Transformers.js | Your own sentences and a query | Mean pooling → normalized vectors → cosine matrix; bi-encoder vs cross-encoder; brute force vs ANN (HNSW-style graph walk); a 3-step RAG pipeline (retrieve → stuff prompt → generate, generation precomputed) | Live embeddings; precomputed generation |
| P2 | **Tiny seq2seq + attention** (own GRU encoder-decoder with Bahdanau attention on a synthetic task, e.g. date formats) and **tiny encoder-decoder Transformer** (T5-style, same task, cross-attention) | Design ≈50k–0.2M | Own JS / ORT Web | Typed source text | Encoder states, scores, softmax, context vector, decoder step, cross-attention, clickable beam-search tree | Live (fills the seq2seq-vis gap without a server) |
| P2 | **Tiny Mamba / SSM** (own, 2 layers, d=32, copy and selective-copy tasks) | Design <0.1M | Own JS (the scan is simpler in JS than in ONNX) | Tokens | Δ, A, B, C per token; scan animation; hidden state | Live |
| P2 | **Tiny MoE** (own, 4 experts, top-2) + routing traces from a small open MoE (choice unverified) | Design <0.2M | Own JS + precomputed | Text | Router logits; layer × expert grid per token; cumulative routing heatmap | Live (tiny) / precomputed (real) |
| P2 | **Logit lens on a small modern LLM** | e.g. SmolLM2-135M (WebLLM lists 360 MB VRAM at q0f16) | Precomputed; link to Steerscope and Neuronpedia | Curated prompts | Per-layer "when did it decide" ladder | Precompute |
| P2 | Planned backlog: NER / sequence labeling with a CRF layer (own tiny BiLSTM-CRF; Viterbi animation); reasoning models and agents/tool use (precomputed traces from a nanochat-scale model) | Design <0.2M | Own JS / precomputed | Text | Emission vs transition scores; Viterbi lattice | Live / precomputed |

### Speech / audio

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| P0 | **Speech-commands KWS "18w"** (retrain our own copy with the Apache-2.0 tfjs-models notebook so the weights' provenance is ours; cite Warden 2018, arXiv:1804.03209; dataset license unverified here) | 1,468,684 params; 5.9 MB fp32 | TF.js (direct port), wrapped in an all-layer-outputs `tf.model` | Microphone, WAV upload (JS STFT unit-tested against `sc_preproc_model`), self-recorded clips | 2048-point FFT → 232 bins × 43 frames → z-norm → conv1 8@42×225 → conv2 32@20×109 → conv3 32@9×51 → conv4 32@3×22 → flatten 704 → dense 2000 (96% of the parameters) → softmax over 20 | Live; precompute top clips per channel |
| P1 | **Whisper-tiny.en** (Xenova port; OpenAI code and weights MIT) | 39M params. q8: encoder 10,124,913 B + merged decoder 30,727,382 B ≈ 40.9 MB total (derived). fp16: 16.5 MB + 59.6 MB. The single decoder file (≈29.3 MiB) exceeds the 25 MB/MiB per-file caps, so it must be chunked (§7) | Transformers.js / ORT Web with chunk loader | Microphone, file | 30 s padding → 80-bin log-mel → conv stem → encoder layers → decoder token by token → cross-attention alignment (DTW timestamps); an attention-window slider | Live forward (desktop opt-in, "Load live model (41 MB)"); precompute internals and "listen to what a neuron hears" top clips (Reid's idea) |
| P2 | **wav2vec2-CTC** (HF port; model and size unverified) | unverified | Transformers.js / precomputed | Microphone, file | Frame-level outputs collapsing into text; an interactive CTC lattice ("type a word, see every valid alignment and the α table fill in"); masked-frame contrastive objective | Precompute model; live CTC toy |
| P2 | **Kokoro-82M TTS** | q8 92.4 MB; q8f16 86.0 MB; fp16 163.2 MB; fp32 325.5 MB | kokoro-js (HF-hosted) | Text, voice | Normalization diff → IPA → token chips → waveform + spectrogram; voice-vector interpolation; re-export with durations and F0 for alignment views | Live opt-in; precompute durations and F0 for presets. Resolve the eSpeak NG (GPL-3.0) bundling question first |
| P2 | **CREPE** pitch tracker (MIT; capacity and size unverified) | unverified | TF.js | Sing or hum into the microphone | The full 360-bin output distribution over time as a heatmap; bins → weighted-average decoding | Live |
| P2 | Planned backlog: neural audio codec / residual vector quantization (own tiny RVQ on log-mels, "how audio becomes tokens"); general audio classification (YAMNet/AST, availability unverified); voice-activity detection; speaker embeddings; source separation; music generation (precomputed) | Design ≤2 MB own | Own JS / precomputed | Microphone, file | Codebook stages adding residual detail; bitrate vs quality slider | Live (own) / precomputed |

### Time series

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| P1 | **Scalar LSTM** (1 unit; every weight visible) | 16 params | Own JS | Preset sequences; editable weights | Every gate and the cell state per time step | Live |
| V1 | **TinyForecaster** (own LSTM, univariate, hidden 16) | Design <10k | Own JS (worker training on the autograd core) | Drawn curve, CSV paste, bundled series | Sliding windows, gate bars per step, prediction vs truth, multi-step rollout error growth | Live training and inference |
| P1 | **ECG autoencoder** (port of Anomagram, MIT) | unverified (small) | TF.js port or own core | Draw an ECG; design the net | Encoder → bottleneck → decoder; reconstruction overlay; error histogram; threshold; ROC | Live |
| P1 | **Kalman filter / linear state-space baseline** (own JS) | — | Own JS | Same inputs | Predict/update steps with the covariance ellipse; compare with LSTM | Live |
| P2 | **Tiny TCN / Transformer / PatchTST-style forecaster** (own) + ARIMA baseline (own JS; the arima JS library has no license) | Design <50k | Own JS | Same as above | Dilated-conv receptive field; patching; attention over past steps; comparison with ARIMA | Live |
| P2 | **Forecasting foundation models** (Chronos / TimesFM; sizes and licenses unverified) | unverified | Precomputed | Bundled series | Tokenized values (Chronos) or patches (TimesFM); quantile fan charts; zero-shot vs our trained forecaster | Precompute |
| Prereq | Fourier widget (Jezzamon, MIT) | — | JS | Draw a wave | Components slider | Live |

### Recommendation

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| V1 | **Matrix factorization** (own; rank 2 for display, rank 8 for accuracy; a MovieLens slice if its license allows, otherwise synthetic data) | Design (users + items) × k, <200 KB | Own JS SGD in a worker | Rate ~10 movies | Rating matrix filling in; your 2-D user vector moving among item factors; the dot product; loss | Live training |
| P2 | **Two-tower** (own: item tower precomputed, user tower as live JSON weights) | Design <1 MB | Own JS | Pick liked items | Towers → embeddings → dot product → top-k retrieval → ranking stage | Hybrid |
| P2 | **CTR models** (own factorization machine, Wide&Deep / DLRM-style toy) and **sequential recommenders** (own tiny SASRec) | Design <0.5 MB | Own JS | Click history | Feature crosses as pairwise dot products; causal attention over your click history | Live |
| P2 | Item embedding map | — | regl/WebGL (WizMap pattern) | Click an item | Neighbors in the learned space | Precomputed |

### Tabular / classic ML

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| V1 | **MLP playground** (port of TF Playground `nn.ts`, Apache-2.0; add softmax + cross-entropy) | ≤6 layers × 8 neurons | Own TS in a worker | 2D points | Per-neuron input-space heatmaps, weights as edges, gradient overlay, "trace one point" with numbers | Live |
| P1 | **Classic-ML canvas:** kNN, logistic regression, linear and kernel SVM (the kernel trick as a lift to 3D), decision tree, random forest, gradient boosting, naive Bayes, k-means, GMM/EM, DBSCAN, isolation forest, PCA, t-SNE (tsnejs), UMAP (umap-js, Apache-2.0) | — | Pure JS | One shared "draw your own dataset" canvas | Tree splits with particles falling through (R2D3 idea), boosting residuals as point size, support vectors, centroid and EM steps, isolation-path lengths, PCA "find the best angle yourself", t-SNE small multiples over perplexity | Live |
| P2 | **Gaussian processes, HMMs, metric learning** (siamese network + triplet loss on drawn shapes) | — / design <100 KB | Pure JS / own core | 2D points, sequences, drawings | GP posterior band updating per clicked point; HMM forward/Viterbi lattice; embedding pulling positives together | Live |
| P2 | **Gradient-boosted trees on tabular data** (TF-DF adult GBT, 71,537 B, or our own) | ~72 KB | TF-DF in the browser | Feature form | Path highlight per tree, cumulative sum; SHAP waterfall (§6.7) | Live |

### Reinforcement learning and control

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| V1 | **Tabular gridworld** (own; ideas from web-rl-playground (MIT) and REINFORCEjs (no license)), opening with a **multi-armed bandit** section (ε-greedy, UCB, Thompson) | Q-table | Own JS | Pull arms yourself; edit cells and rewards; set γ, ε, α | Bandit: regret curve, posterior bars. Gridworld: value heatmap, policy arrows, the Q/SARSA update with numbers filled in | Live |
| P1 | **Cart-pole REINFORCE** (tfjs-examples, Apache-2.0) | 769 params at hidden 128 | TF.js port in a worker (not driven by `requestAnimationFrame`, which stalls when the tab is hidden) | Hyperparameters; play/step | State vector → hidden → p(left) → sampled action; discounted returns; normalized advantages; gradient contribution colored by the sign of the advantage | Live training |
| P2 | **Snake DQN** (retrain our own copy) | 1,120,019 params; 4.5 MB | TF.js | Play/step | Q-values drawn on the board; conv features | Live inference |
| P2 | **AlphaZero-lite tic-tac-toe** (own policy/value net + MCTS) | Design ≈10k | Own JS | Play your own moves | Priors vs visit counts, PUCT, value head, MCTS phases | Live |
| P2 | Planned backlog: PPO on continuous control (2D reacher, own); model-based RL / world model (own tiny dynamics net, imagined rollouts); imitation learning and offline RL (behavior cloning from your own demonstrations in the gridworld); robotics/embodied: 2D diffusion policy toy reusing the flow-matching engine; VLA models as a precomputed walkthrough | Design <0.5 MB own | Own JS / precomputed | Demonstrations you draw or play | Policy distribution over actions; compounding error in BC vs DAgger | Live (own) / precomputed |

### Graphs

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| V1 | **GCN** (own, PyTorch Geometric, molecule graph classification; GNN 101's comparable models are 25–113 KB) | Design <120 KB | Own JS from JSON weights (ORT Web for parity) | Graph editor, molecule presets | Message passing per node, aggregation choice, update equation with numbers, node-link ↔ matrix, per-layer embedding scatter, readout | Live |
| P1 | **GAT / GraphSAGE / GIN** on the same node (Distill comparison idea); GAT link prediction | Design <120 KB each | Same | Same | Attention coefficients per edge; aggregator comparison | Live |
| P2 | Planned backlog: node embeddings (DeepWalk/node2vec random walks → skip-gram, reusing word2vec); graph transformers; E(3)-equivariant GNNs (rotate a molecule, outputs rotate with it); knowledge-graph embeddings (TransE as vector arithmetic) | Design <200 KB | Own JS | Graph editor | Random-walk corpus; equivariance check | Live |

### Generative

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| P0 | **2D flow matching / diffusion** (own MLP: [x, y, t] → 64 → 64 → 64 → 2 with ELU, as in Diffusion Explorer, reimplemented) | 8,706 params ≈ 35 KB | Own autograd core in a worker | Brush a 2D distribution | Samples, density contours, trajectories, warped mesh, velocity field. Drill-down: click a point to see the denoiser MLP's activations and predicted velocity, plus the target x_t = (1−t)x₀ + t·x₁ | Live training; KB-sized pretrained presets for instant first paint |
| P1 | **2D GAN** (port of GAN Lab's raw-op MLPs, Apache-2.0) | Presets 0.5–2.3 KB | TF.js 4.x (direct port) | Same brush | Discriminator heatmap, generator manifold, gradient arrows, slow-motion steps, JS/KL divergence charts | Live |
| P1 | **VAE with a 2D latent** (VAE Explainer, MIT, MNIST-like data) | unverified (small) | TF.js port / ORT Web | Draw a digit; drag the latent point | Encoder → μ, log σ² → reparameterization → decoder; latent map | Live |
| P1 | **Score matching / Langevin sampling** (own, 2D) | Design <10k | Own core | Same brush | Score field arrows; Langevin chains with a noise slider; link to DDPM equivalence | Live |
| P2 | **VQ-VAE** (own tiny) · **RealNVP 2D flow** (own) · **PixelCNN / MaskGIT-style token generator** (own tiny, 16×16) | Design <0.5 MB | Own JS | Draw / brush | Codebook nearest-code lookup; coupling layers warping the plane; raster vs parallel masked decoding order | Live |
| P2 | **Tiny DDPM with a U-Net** (own, 28×28; labml `ddpm/unet.py`, MIT) and **tiny DiT** (own, same data) | Design ≤8 MB | ORT Web | Seed or draw | Down/up path, skip connections, noise prediction per timestep; DiT patch tokens with adaLN conditioning; consistency / few-step distillation compared by step count | Live for a few steps; precompute full chains |
| P2 | **Stable Diffusion** (Diffusion Explainer pattern on a GPU; live SD-Turbo is optional and unverified) | — | Precomputed | Prompt pairs, guidance, seed | Text encoder, U-Net steps, cross-attention conditioning maps, CFG, VAE decode, UMAP trajectories | Precompute |
| P2 | **ACGAN generator** (tfjs-examples) | 2,657,897 params; 10.6 MB | TF.js | Latent sliders + class | Generator layers | Live |
| P2 | History corner: Hopfield network, RBM, energy-based model (own 2D) | <10 KB | Own JS | Draw patterns | Energy descent; Gibbs sampling | Live |

### Multimodal

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| V1 | **TinyCLIP** (own: synthetic shapes × captions such as "red circle", "blue square") | Design <1 MB | Own JS / ORT Web | Draw a shape, type captions | Two encoders → normalized embeddings → N×N similarity → temperature → InfoNCE diagonal → zero-shot | Live (the default; real CLIP is too large for auto-load) |
| P1 | **CLIP ViT-B/32** (Xenova/clip-vit-base-patch32) | ≈151M params (derived from 605.8 MB fp32). Quantized: full 153.7 MB, vision 89.1 MB, text 64.5 MB | Precomputed by default; explicit opt-in live from HF (may be unreachable from China) | Your own photos and captions (when live); curated sets (precomputed) | The same matrix view with real images; zero-shot bars; embedding map | Precomputed first; live opt-in on desktop only |
| P2 | **VLM** (SmolVLM / Florence-2 from the Transformers.js examples; sizes unverified) + an own LLaVA-style connector toy (frozen TinyViT → MLP projector → TinyGPT) | unverified / design <2 MB | Transformers.js / precomputed / own JS | Image + question | Image tokens entering the LM sequence; precomputed cross-attention per generated token | Mixed |
| P2 | **CLAP-style audio-text contrastive** (own tiny on KWS words × captions) | Design <1 MB | Own JS | Speak a word, type captions | Same similarity-matrix view as TinyCLIP | Live |

### Science

| Tier | Model | Params / download | Runtime | Input | Layers or steps to animate | Precomputed vs live |
|---|---|---|---|---|---|---|
| V1 | **PINN, damped oscillator** (own; port of Ben Moseley's MIT notebook) | Design <5k | Own autograd core (needs second derivatives; jax-js as a fallback) | Drag observation points; damping; λ | Plain NN vs PINN side by side; data-loss vs physics-loss bars; residual heatmap | Live training |
| P1 | **ESM-2 t6 8M** (Xenova/esm2_t6_8M_UR50D; 6 layers) | ≈7.6M (derived from 30.3 MB fp32); fp16 15.4 MB; q8 8.06 MB | Transformers.js (self-hosted q8) | Amino-acid sequence; click to mask or mutate a residue | Masked-residue distribution (protein "BERT"); attention heads | Live (ESM-2 weights license unverified; check first). Structure from the ESMFold API (CORS `*`, worked 2026-10-05) shown in Mol*, with cached fallbacks |
| P2 | **AlphaFold page:** AFDB API + Mol* + a PAE heatmap linked to 3D; architecture walkthrough linking to The Illustrated AlphaFold | — | Precomputed API | Protein name or accession | Single vs pair representation; triangle-update animation on a small protein | Precomputed |
| P2 | Planned backlog: neural operators (own tiny FNO on 1D heat/Burgers: Fourier layer as "filter in frequency space"); weather GNNs (GraphCast/GenCast, precomputed walkthrough); molecular property prediction and generation (reuses the GCN page); DNA language model (k-mer tokens; reuses tokenizer views); protein design (ProteinMPNN/RFdiffusion, precomputed); medical-image segmentation (reuses the U-Net page) | Design <2 MB own | Own JS / precomputed | Drawn initial condition, molecule editor, sequences | Spectral truncation slider; message passing on a mesh | Live (own) / precomputed |

---

## 6. Concepts and methods catalog (Training Concepts track)

Each concept page uses the template in §4.4. "Links" uses the model slugs from §5. **(v1)** marks the about 11 pages in the v1 freeze; every other row is planned.

### 6.1 Foundations of training

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Loss and gradient descent (v1, merged with backprop) | "Beat the error score" with manual weight and bias dials, then press the GD Step button (Alammar) | Single neuron in JS | Live | mlp-playground, tiny-cnn |
| Backprop / autograd (v1) | Edit any leaf value and the whole graph recomputes. Step the backward pass one chain-rule edge at a time, with grads shown as "?" until reached. Highlight `grad +=` when a node is used twice. Numerical gradient check button | In-house Value engine (port of micrograd MIT or micrograd-ts MIT); the same core powers live training | Live | mlp-playground, tiny-gpt (single-neuron drill-down) |
| Optimizers, LR schedules, warm-up, LR search (v1) | SGD / momentum / Adam / AdamW racing on one surface; optimizer state as a momentum arrow and squared-gradient boxes (lilipads idea); eigen view (Distill momentum). Schedule picker with an SVG preview; LR range test on TinyGPT; grid of precomputed runs (LR × warm-up × schedule × 3 seeds) with a step scrubber; deep links to Sequence Toy configs. Gradient clipping toggle on an exploding-gradient toy | JS on analytic surfaces + live 2D MLP + precomputed TinyGPT sweep | Both | tiny-gpt, mlp-playground |
| Batch size, regularization, overfitting (v1) | Train vs test curves diverging; L1/L2 and weight decay; dropout (units blinking per step); data augmentation on TinyCNN drawings; early-stopping marker; noise and train-ratio sliders (TF Playground); double descent (link to MLU-Explain); batch size vs LR (Gradient Lander idea) | 2D MLP + TinyCNN | Live | mlp-playground, classic-ml, tiny-cnn |
| Initialization and normalization | Activation and gradient histograms per layer under zero / too small / Xavier / He / too large init (ideas from deeplearning.ai AI Notes); BatchNorm vs LayerNorm on the same tensor (normalized over which axis) | Tiny MLP / TinyGPT tensors | Live | tiny-gpt, resnet-18 |
| Residual connections and vanishing gradients | Toggle skip connections in a 20-layer MLP; gradient-norm bars per layer | Own core | Live | resnet-18, tiny-gpt |
| Choice of loss | Cross-entropy vs MSE on the same classifier; label smoothing; focal loss on an imbalanced 2D set | Own core | Live | mlp-playground, tiny-cnn |
| Hyperparameter search and grokking | Grid vs random search on a 2D objective; a modular-arithmetic grokking replay (precomputed TinyGPT-style run) | JS + precomputed | Both | tiny-gpt |

### 6.2 Pre-training objectives and data

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Tokenization (v1, P0) | (1) Train BPE on your own text with a merge-by-merge stepper (Cornell idea), plus a vocabulary-size slider showing token count vs vocabulary size. (2) Real tokenizers side by side on the same text, with a byte view. E.g. for 我爱自然语言处理 unbelievable!: cl100k splits 爱 into 2 byte tokens, Llama 3 keeps 自然/语言/处理 whole, bert-base-cased yields 8 × [UNK] | Own BPE in JS + `@huggingface/tokenizers` | Live | tiny-gpt, tiny-bert, bert-base, whisper (decoder tokens), clip |
| Embeddings | wevi-style full-matrix training on your own context/target pairs; GloVe custom axes (word2viz); BERT contextual embeddings ("bank" in two sentences) | Own JS word2vec + precomputed BERT | Both | word2vec, bert-base, tiny-clip |
| Next-token prediction (causal LM) (v1, P0) | One sentence becomes many (context → next) examples via a sliding window; drag p(correct token) to watch −log p; color your own sentence by surprisal (GLTR rank buckets), with the average showing loss and perplexity; training replay from gibberish to text | TinyGPT live; distilgpt2 optional | Live + precomputed checkpoints | tiny-gpt, gpt-2, tiny-mamba |
| Decoding / sampling | Temperature, top-k and top-p applied to the same cached logits; generate token by token | Shared output-head component | Live | tiny-gpt, gpt-2, whisper, tiny-seq2seq |
| Self-supervised objectives overview (v1) | One page, one sentence/image: what is hidden and what is predicted under CLM, MLM (80/10/10), next-sentence prediction, prefix LM, span corruption (T5), denoising (BART), fill-in-the-middle, multi-token prediction, replaced-token detection (ELECTRA), contrastive, MAE, JEPA and denoising diffusion. Each objective is drawn as a mask pattern over the same tokens or patches | TinyGPT live + illustrative mask diagrams (labeled) | Both | tiny-gpt, tiny-bert, tiny-vit |
| Masked LM | Click any token to mask it; animate the 80/10/10 rule; TinyBERT vs TinyGPT trained on the same corpus; protein masking with ESM-2 | TinyBERT, ESM-2 | Live | tiny-bert, bert-base, esm-2 |
| Next-sentence prediction | Build sentence pairs (IsNext / NotNext) and see the [CLS] head's output; note RoBERTa-style removal | BERT-base precomputed | Precomputed | bert-base |
| Contrastive learning (CLIP / InfoNCE / SimCLR / SimCSE) (v1) | Add your own drawings and captions to a batch; see the N×N similarity matrix, drag the temperature, watch the loss; a training-progress slider morphs the matrix from noise to a clean diagonal. Augmentation-pair view (SimCLR) and dropout-pair view (SimCSE); BYOL/SimSiam/Barlow Twins as "no negatives" variants (precomputed collapse vs no-collapse curves) | TinyCLIP live; CLIP ViT-B/32 precomputed | Live | tiny-clip, clip |
| Masked autoencoding | Click patches to mask them in the TinyViT grid; masking-ratio slider with reconstructions | Own tiny MAE on TinyViT patches; ViT-MAE precomputed (model unverified) | Both | tiny-vit, dinov2 |
| Self-distillation (DINO) | Student/teacher views of the same image; attention and patch-PCA maps emerging | DINOv2-small live attention + precomputed | Both | dinov2 |
| Denoising objectives | Noise schedule slider; DDPM vs flow-matching targets on the same 2D data; "two equivalent views" | 2D playground | Live | flow-2d, ddpm-unet |
| Scaling laws | Chinchilla L(N, D) = E + A/N^α + B/D^β with a draggable compute budget (use the 2024 replication refit, as the understandingdata page notes); plus our own sweep of TinyGPT sizes | Analytic + precomputed sweep | Both | tiny-gpt |
| Data mixtures, curricula and curation | Choose a zh/en mix ratio and see per-domain loss curves from precomputed TinyGPT runs. Curation pipeline toy: near-duplicate detection (MinHash), quality filter threshold, synthetic data share. Link to the Smol Training Playbook | Precomputed runs + JS | Both | tiny-gpt |

### 6.3 Transfer learning and fine-tuning

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Transfer learning / feature extraction (v1, P0) | Record 8 clips per new word; the frozen KWS base (gray, locked) feeds a new 2000 → N head; PCA of your 2000-d embeddings before and after; Teachable-Machine-style webcam version on MobileNet | Speech-commands transfer API (TF.js port); MobileNet head | Live | kws, resnet-18 |
| Linear probe vs partial vs full fine-tuning | Click blocks of a 12-block GPT to freeze or unfreeze; see precomputed accuracy, trainable-parameter count and wall time for that configuration (3 seeds). Reference points, re-measured by us: rasbt ch06 last_block 95.00%, last_layer 72.00%, all layers 96.67%; the "which layer's embeddings work best" probe for BERT | Precomputed sweeps (LLMs-from-scratch code); live head-only training on KWS | Both | gpt-2, bert-base, kws |
| Vision fine-tuning | Fine-tune TinyCNN on your own 2 new drawing classes, with and without augmentation; ResNet-18 head-only vs full (precomputed) | Own core live + precomputed | Both | tiny-cnn, resnet-18 |
| LP-FT and feature distortion | Embedding scatter of in-distribution and out-of-distribution points drifting under full fine-tuning vs LP-then-FT | Precomputed (gap) | Precomputed | resnet-18, bert-base |
| Discriminative / layer-wise LR decay | Layer-wise gradient-norm and weight-change bars with an LR slider and a decay factor | Live small model + precomputed | Both | kws, bert-base |
| Catastrophic forgetting (v1) | Train a 2D MLP on task A, then task B, and watch A's decision boundary erode live. Toggles: replay buffer %, L2 pull toward old weights (EWC-lite), freeze the first layer, adapter-only training, lower LR. Real version: TinyGPT fine-tuned on Chinese poems, with English-names perplexity rising per step | 2D playground (live) + TinyGPT (precomputed) | Both | mlp-playground, tiny-gpt |
| Multi-task and intermediate-task training | Drag task blocks on a timeline (RTE alone, MNLI → RTE, multi-task mix) and see the matching precomputed curves; a "gradient conflict" toy with two loss bowls over 2 shared parameters | The builder's BERT MRPC/RTE pipeline on a GPU; JS toy | Both | bert-base |
| Continued / domain-adaptive pre-training and long-context extension | Continue TinyGPT on a new domain and watch in-domain vs general loss; extend context 64 → 256 with RoPE scaling vs naive extrapolation (precomputed) | Precomputed TinyGPT runs | Precomputed | tiny-gpt |
| Instruction tuning / SFT | Write a conversation and see it rendered through a real chat template (`@huggingface/jinja`, MIT), tokenized, with the loss mask: assistant tokens green (trained), system/user/template tokens gray (label −100). Toggles: assistant-only vs all tokens, last turn vs all turns, packing. Plus a template-mismatch demo and base vs instruct outputs on fixed prompts (nanochat base vs SFT checkpoints) | jinja + transformers.js tokenizer; mask spans found with private-use Unicode sentinels (jinja strips `{% generation %}`) | Live + precomputed outputs | tiny-gpt, gpt-2 |
| In-context learning vs fine-tuning vs RAG | The same few examples given in the prompt vs used for a few fine-tuning steps vs retrieved from a store; compare outputs, cost and freshness; a decision page "RAG or fine-tune?" | Precomputed (gap) + retrieval page | Precomputed | tiny-gpt, gpt-2, sentence-embeddings |
| Few-shot / meta-learning, semi-supervised pseudo-labeling, domain adaptation, embedding-model fine-tuning | 2D toys: prototypical networks on 5-shot clusters; pseudo-label confidence threshold spreading labels; source vs target domain shift with a feature-alignment slider; contrastive fine-tuning of the bi-encoder on your own pairs | Own core | Live | mlp-playground, sentence-embeddings |

### 6.4 Parameter-efficient tuning

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| LoRA / QLoRA / DoRA (v1) | Live parameter counter: r(d_in + d_out) vs d_in·d_out (e.g. a 10,000×20,000 W at r=8 needs 240,000 vs 200,000,000). Animate W₀ + (α/r)·BA, with B starting at zero so step 0 equals the base model. DoRA as magnitude/direction split. Live LoRA on TinyGPT; precomputed rank/alpha sweeps on the spam task (LoRA r=16 on GPT-2 small trains 2,666,528 of 124,441,346 params). Real adapter matrices as heatmaps. A LoRA-scale slider over precomputed diffusion image grids (Concept Sliders idea) | Analytic JS + own core + LLMs-from-scratch appendix-E code on a GPU | Both | gpt-2, tiny-gpt, flow-2d / sd |
| Adapters, BitFit, (IA)³, prefix/prompt tuning | The same block diagram with each method's trainable parameters lit up: adapter bottlenecks, bias-only, (IA)³ rescaling vectors, learned prefix tokens. Trainable-parameter footprint and precomputed accuracy side by side | Precomputed on BERT / TinyGPT (gap) | Precomputed | bert-base, tiny-gpt |
| Memory: what fits on your GPU | Stacked memory bars (weights, gradients, optimizer state, activations) for full fine-tuning vs LoRA vs QLoRA; VRAM is an input, so "can I fine-tune X on a 24 GB GPU?" is answerable (ideas from gpu_poor, the Ultra-Scale Playbook, Kingy's "parameters are not VRAM", MLSysBook's Tensor Tetris) | Analytic | Live | gpt-2, bert-base |

### 6.5 Alignment

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Reward modeling | "You are the labeller": choose between response pairs; a Bradley–Terry reward model fits to your choices | JS toy (idea from dev48v, young, MIT) | Live | tiny-gpt |
| Best-of-n / rejection sampling | Sample n completions, score with the reward model, keep the best; reward vs n and KL vs n curves | Precomputed TinyGPT samples + live scoring | Both | tiny-gpt |
| RLHF (PPO + KL penalty) | A β (KL-leash) slider: proxy reward keeps rising while true quality peaks then falls (Goodhart; reward hacking examples). A 1-D Gaussian policy (μ, σ) moving on one landscape under REINFORCE / PPO / GRPO (alignment-lab idea, reimplemented: no license) | JS toy | Live | cart-pole (policy gradient), tiny-gpt |
| DPO and variants (KTO, ORPO, SimPO) | Loss −log σ(β·Δ) with a β slider; it starts exactly at ln 2 ≈ 0.693 with zero margin. Variant switcher shows how each changes the loss curve (reference-free, unpaired). Precomputed reward-margin curves and before/after generations. GLTR-style token coloring shows which tokens became more or less likely | TinyGPT preference-tuned on our own polite vs rude pairs (rasbt's pair data has an unverified license) | Both | tiny-gpt |
| GRPO / RLVR | Group sampling, relative advantages, verifiable reward on a toy arithmetic task; nanochat's RL stage as the real-scale replay | Toy + precomputed (nanochat) | Both | tiny-gpt |
| Constitutional AI / RLAIF and reasoning distillation | Critique-and-revise loop on fixed prompts; a teacher's reasoning traces distilled into TinyGPT (precomputed before/after) | Precomputed | Precomputed | tiny-gpt |
| Knowledge editing and machine unlearning (P2) | Edit one fact in TinyGPT (rank-one update toy) and watch neighbors change; unlearn a subset and test leakage | Precomputed | Precomputed | tiny-gpt |

### 6.6 Compression and efficiency

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Number formats and mixed precision (v1, merged with quantization) | Click bits for FP32 / FP16 / BF16 / FP8 (e4m3, e5m2) / FP4 / INT4 / NF4 (Float Toy and ngrok ideas; reimplement, since neither has a license) | JS | Live | — |
| Quantization (v1) | Rounding ledger for 8 values (absmax vs affine, clipping range, block size); one outlier wrecking the scale (ngrok idea); quantize a TinyGPT layer in JS and watch logits, top-1, perplexity and KL change. Real A/B: Kokoro fp32 (325.5 MB) vs q8 (92.4 MB) vs q4 (305.2 MB, barely smaller because 4-bit only touches MatMul weights); SlimSAM masks fp32 vs fp16 vs q8; DINOv2 attention q4 vs fp32. Weight-outlier "wall of tensors" (Weight Atlas idea) | JS + real models | Live | tiny-gpt, kokoro, slimsam, dinov2 |
| PTQ vs QAT (GPTQ / AWQ) | Same TinyGPT layer: round-to-nearest vs error-compensating (GPTQ-style column updates) vs activation-aware scaling (AWQ-style), then QAT recovering accuracy (precomputed curve) | JS + precomputed | Both | tiny-gpt |
| Pruning | Sparsity slider on TinyCNN (magnitude vs structured), with the mask drawn; prune → accuracy drop → fine-tune to recover (precomputed curve) | TinyCNN live + precomputed | Both | tiny-cnn |
| Knowledge distillation | Teacher vs student soft labels; temperature T; hard-label vs soft-label student training race (dev48v idea). Real examples: distilgpt2 vs GPT-2 sizes; SlimSAM is pruned and distilled from SAM ViT-B | JS toy + real models | Both | gpt-2, slimsam, tiny-cnn |
| Model merging / weight averaging (P2) | Interpolate between two TinyGPT fine-tunes and see the loss along the path | Precomputed | Precomputed | tiny-gpt |

### 6.7 Interpretability and explainability (planned)

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Saliency and integrated gradients | On your TinyCNN drawing: vanilla gradient vs IG (baseline and step-count sliders) vs SmoothGrad; a sanity check that randomizes weights layer by layer | TinyCNN live | Live | tiny-cnn, resnet-18 |
| CAM family | Formula card CAM = ReLU(Σ w_k A_k); method switcher over precomputed maps | Precomputed (pytorch-grad-cam) + live CAM | Both | resnet-18 |
| SHAP / LIME for tabular models | SHAP waterfall for one row of the GBT model; LIME's local linear fit drawn on the 2D canvas | JS | Live | gbt, classic-ml |
| Probing classifiers | Train a linear probe on each TinyGPT/BERT layer for a property; accuracy-per-layer bars | Precomputed activations + live probe | Both | tiny-gpt, bert-base |
| Logit lens | Per-layer decoded predictions for your prompt | TinyGPT live; SmolLM2 precomputed | Both | tiny-gpt, logit-lens |
| Activation patching | Swap one activation from a clean to a corrupted prompt and watch the logit difference; heatmap over layer × position | TinyGPT live | Live | tiny-gpt |
| Sparse autoencoders and steering | Own SAE on TinyGPT's residual stream; top-activating tokens per feature; a steering slider adds a feature direction live; link to Neuronpedia for real-scale features | Own SAE (precomputed) + live steering | Both | tiny-gpt |
| "Attention is not explanation" | Same prediction with very different attention maps; attention vs gradient attribution | TinyGPT live | Live | tiny-gpt, bert-base |

### 6.8 Evaluation (planned)

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Splits and leakage | Drag points between train/val/test on the 2D canvas; leak a test point and watch the score inflate | 2D MLP | Live | mlp-playground |
| Metrics and calibration | Confusion matrix, precision/recall, ROC; reliability diagram for TinyCNN with a temperature-scaling slider | Own core | Live | tiny-cnn, classic-ml |
| Perplexity vs downstream accuracy | TinyGPT checkpoints: loss falls smoothly while a task metric jumps | Precomputed | Precomputed | tiny-gpt |
| Benchmark contamination | Fine-tune TinyGPT with test items leaked into training; compare clean vs contaminated scores | Precomputed | Precomputed | tiny-gpt |
| pass@k | Sample k answers from a toy arithmetic model; the unbiased pass@k estimator with live numbers | Analytic + precomputed samples | Both | tiny-gpt |
| LLM-as-judge (P2) | Position-bias and verbosity-bias toggles on precomputed judgments (judge model unverified) | Precomputed | Precomputed | — |

### 6.9 Inference and serving (planned)

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| KV cache and prompt caching | TinyGPT generation with and without the cache: recomputed cells grayed out; cache memory counter; shared-prefix reuse across two prompts (ngrok idea, reimplemented) | TinyGPT live | Live | tiny-gpt, gpt-2 |
| Attention variants (MHA / MQA / GQA / MLA) | Head-sharing diagram with a KV-cache size calculator; the same TinyGPT trained with MHA vs GQA (precomputed loss) | Analytic + precomputed | Both | tiny-gpt |
| Positional encodings and long context | Sinusoidal vs learned vs RoPE (rotate the 2D pair) vs ALiBi (bias ramp); extrapolation beyond context 64 on TinyGPT variants | Live + precomputed | Both | tiny-gpt |
| Speculative decoding | A tiny draft model proposes k tokens, TinyGPT verifies; accept/reject animation and speedup counter | Live | Live | tiny-gpt |
| Batching and FlashAttention tiling | Tile-by-tile softmax with a running max and sum; memory-read counter vs naive attention; static vs continuous batching timeline | JS | Live | tiny-gpt |
| Test-time compute and reasoning | Majority vote and best-of-n on toy arithmetic; accuracy vs samples; chain-of-thought length vs accuracy (precomputed) | Precomputed | Precomputed | tiny-gpt |

### 6.10 Robustness, privacy and scale (planned)

| Concept | Playable interaction | Engine | Live / precomputed | Links to |
|---|---|---|---|---|
| Adversarial examples and robustness | FGSM/PGD on your drawing (see the Vision page); adversarial training toggle on the 2D MLP showing the boundary margin grow | Own core | Live | tiny-cnn, mlp-playground |
| Federated learning | 4 simulated clients with non-IID 2D data; FedAvg rounds; client drift | Own core | Live | mlp-playground |
| Differential privacy (DP-SGD) | Per-example clipping + Gaussian noise on the 2D MLP; accuracy vs ε; one "canary" point's influence | Own core | Live | mlp-playground |
| Training at scale | Memory breakdown and data/tensor/pipeline parallelism diagrams; all-reduce animation; link to the Ultra-Scale Playbook and MLSysBook games | Analytic | Live | gpt-2 |

---

## 7. Technical architecture

**Framework and build**
- **App:** SvelteKit 2 + Svelte 5 with `adapter-static`, Vite and TypeScript. Svelte matches the MIT Polo Club code, but reuse means **porting**, not dropping in:
  - CNN Explainer components are Svelte 3 + Rollup 1 + TF.js 1.4 and must be rewritten for Svelte 5 runes.
  - Transformer Explainer components (Svelte 5.53 / Kit 2.53) pin `@xenova/transformers` ^2.17.1 and `onnxruntime-web` ^1.23.0, so they must be moved to Transformers.js 4.x / ORT 1.30.
  - Budget porting time in M0/M1.
- **Rendering:**
  - D3 v7 for SVG diagrams and annotations;
  - Canvas 2D for small heatmaps;
  - regl/WebGL or WebGPU for dense grids and more than ~10⁵ points (WizMap; Sequence Toy's WGSL visualizer);
  - GSAP or plain timelines for choreography.
- **Math:** KaTeX prerendered at build time, shipping only the CSS (labml does this).
- **Articles and language:**
  - Markdown-in-Svelte.
  - English is written first. The zh pass is driven by the glossary: all terminology lives in one `glossary.json` that both languages and `graph.json` reference.
  - UI strings and presets are bilingual from day one.
- **Offline figures:** reduced-motion stills and article figures rendered from the same spec (Playwright screenshots of `stateAt(t)`; ManimML (MIT) optional for video).
- **Deploy:** GitHub Actions builds once and deploys the same artifact automatically (see Hosting), so the live site never lags the repo.
- **Repo layout** (weights and traces are **not** in git history):

```
site/                 SvelteKit app (kit/, models/<slug>/page, train/<slug>, content/{en,zh}, glossary.json)
site/static/          code-side static files only
assets.lock.json      list of model/trace files: URL (release asset or HF), bytes, sha256, version
pipeline/             Python on a GPU: train/, export/, parity_tests/, precompute/, schema/, budgets/
schema/               pydantic → JSON Schema → generated TS types (Crucible pattern)
core/                 in-house tensor + autograd core (typed arrays, worker-safe)
```

**Model spec (drives layout, inspector, provenance and links):**

```json
{ "slug": "tiny-gpt", "area": "language", "family": "transformer-decoder",
  "params": 110000, "files": [{"path": "models/tiny-gpt/model.bin", "bytes": 440000, "sha256": "…"}],
  "runtime": ["js-reference", "ort-wasm"], "input": "text",
  "precision": "fp32", "parity": {"against": "pytorch-fp32", "max_abs_diff": 1e-5},
  "nodes": [{"id": "blk0.attn.q", "op": "linear", "shape": ["T", 64], "role": "activation",
             "concepts": ["self-attention"], "code": "pipeline/train/tiny_gpt.py#L42"}],
  "edges": [["blk0.ln1", "blk0.attn.q"]], "presets": ["traces/tiny-gpt/ex0.bin"] }
```

The `params`, `bytes` and `parity` values above are design targets.

**ModelRunner interface with three interchangeable backends:**
1. `precomputed`: loads trace files. Always available, and the default on phones.
2. `live-infer`: a Web Worker (Comlink-style messaging).
   - **Own tiny models (≤1 MB):** binary weights plus our own JS forward pass, which gives every tensor. ORT Web is used for parity tests.
   - **Hub models:** ONNX Runtime Web 1.30 with `executionProviders: ['webgpu','wasm']`, or Transformers.js 4.3 for tokenizers, processors and Hub ONNX files.
   - **Device and dtype** are chosen by feature detection (`navigator.gpu.requestAdapter`): fp32/fp16 on WebGPU, q8 on WASM.
3. `live-train`: the **in-house tensor/autograd core** (micrograd-ts style, typed arrays, always in a worker, posting snapshots every N steps), used for all own toy training.
   - TF.js is used only where an existing Apache-2.0 demo is ported as is (GAN Lab, cart-pole, speech-commands transfer, Anomagram, VAE Explainer).
   - jax-js (MIT) is the fallback for higher-order derivatives (PINN) if the core does not support them by V1.
   - TF.js has had no release since 4.22.0 (2024-10-21). Google launched LiteRT.js on 2026-07-09 as "a powerful evolution from TensorFlow.js" for running converted `.tflite` models. An official TF.js maintenance status is unverified, and LiteRT.js does not replace in-browser training.

**Extracting intermediate activations (choose per model class):**

| Model class | Method |
|---|---|
| Own tiny models | Wrap the PyTorch model so `forward` returns a dict of named intermediates, then `torch.onnx.export` with one named output per tensor (Transformer Explainer's `export_to_onnx.py`). Also export raw weights for the JS reference pass. Parity-test JS vs ORT vs PyTorch in CI and publish the measured tolerance on the badge |
| HF models | Stock onnx-community/Xenova exports generally expose final outputs only (attentions only in "-with-attentions" variants), and ORT Web returns only declared graph outputs. So every Hub model shown below model level must be **re-exported** with extra outputs (`output_attentions` / hidden states, or ONNX graph surgery appending tensors to `graph.output`: ishamf's `instrument_model.py`, onnx-modifier's "Add Output") **and self-hosted**. Default expectation: Hub pages are model-level live with precomputed internals |
| TF.js models | Build `tf.model({inputs, outputs: layers.map(l => l.output)})`, or chain `layer.apply` (CNN Explainer); `GraphModel.execute(inputs, [node names])` |
| Large models (precompute) | PyTorch forward hooks, TransformerLens `run_with_cache`, or nnsight on a GPU; pytorch-grad-cam for CAMs; Lucent for feature visualization; UMAP for trajectories; nanochat (MIT) at small `--depth` for lifecycle checkpoints (single GPU via gradient accumulation). Check first that the installed PyTorch wheel supports the GPU |
| Exactness wording | A JS float32/float64 pass differs from PyTorch fp32/fp16 by roughly 1e-6 to 1e-3 depending on depth and op order. Badges state the measured tolerance; q8/q4 runs are labeled "approximate"; never call quantized Hub runs "exact" |

**Trace format:**
- One JSON header (model version, input, and a list of tensors with name, shape, dtype, scale/zero-point, byte offset) plus one binary blob.
- Display tensors are stored as uint8 or fp16 with a per-tensor scale. Example: BERT-base attention for 29 tokens is 2.6 MB as float JSON and 57 KB as gzipped uint8.
- Embeddings use the Embedding Projector layout (float32 `.bytes` + TSV metadata + sprite).
- Never statically import traces into the JS bundle; Transformer Explainer's page chunk is 7.08 MB for this reason.

**Large single files and chunking.**
- Mid models with a single file above 20 MB need chunking: Whisper's merged decoder (30.7 MB ≈ 29.3 MiB) and DINOv2 q8 (25.1 MB).
- **Why:** to satisfy Cloudflare Pages' 25 MiB per-file limit, jsDelivr's 20 MB GitHub limit, and our own ≤20 MB chunk rule.
- **Options:** ONNX external-data format, or a custom loader. The loader fetches ≤20 MB chunks, verifies sha256, concatenates them into one ArrayBuffer and creates the ORT session from bytes (Transformer Explainer's `fetchChunks.js` does this for GPT-2).
- Whether Transformers.js 4.3 can take such a buffer or external data directly is unverified.
- Budget about 1–2 weeks (design target) for this loader in M2.

**Hosting and weight storage (limits from the evidence):**
- **GitHub Pages:** 1 GB per published site; 100 GB/month soft bandwidth; 10-minute deploy timeout; 10 builds/hour soft limit.
- **Git:** warns at 50 MiB per file and blocks at 100 MiB. Git LFS files reach Pages only through Actions with `lfs: true`.
- **jsDelivr:** serves GitHub files only up to 20 MB.
- **Cloudflare Pages (Free):** 25 MiB per file and 20,000 files. A `_headers` file sets custom response headers (up to 100 rules, 2,000 characters per line).
- **Dual deploy.** The same Actions build deploys to GitHub Pages (the builder's account) and Cloudflare Pages.
  - GitHub Pages offers no custom headers (widely reported, not verified from GitHub docs), so it can run only single-threaded WASM unless coi-serviceworker is used.
  - On Cloudflare Pages, COOP `same-origin` + COEP `require-corp` via `_headers` is a commonly used pattern (secondary sources; test on our deploy). It enables `crossOriginIsolated`, SharedArrayBuffer and multithreaded WASM.
  - COEP blocks cross-origin resources that lack CORS/CORP, so HF downloads must use CORS fetches (HF resolve URLs are CORS-enabled) and pages must not embed third-party content.
  - **Decision at M0:** make Cloudflare primary only if threaded WASM measurably helps the P1 mid models. Own tiny models do not need threads.
- **Repo bloat.** Weights and traces never enter git history. Transformer Explainer's repo is 2,118,108 KB (≈2.0 GiB, derived) because its model chunks live in git.
  - CI fetches the files listed in `assets.lock.json` at build time from versioned GitHub Release assets of a separate assets repo, or from HF.
  - It verifies sha256, copies them into the deploy artifact, and fails if the built site exceeds 900 MB (design target).

| Budget line (design targets unless noted) | Size |
|---|---|
| App code, fonts, self-hosted runtimes. onnxruntime-web 1.30.0 ships `ort-wasm-simd-threaded.wasm` 14.2 MB, `.jsep.wasm` 28.3 MB, `.asyncify.wasm` 26.8 MB, `.jspi.wasm` 16.8 MB (npm package unpacks to 144.6 MB). ORT docs say the WebGPU EP uses the `.jsep` file, while other sources name `.asyncify` for newer WebGPU builds. Check which artifact our build actually fetches and self-host only those | ≤ 60 MB |
| Own tiny models (all areas) | ≤ 30 MB |
| Self-hosted mid models (each file ≤ 20 MB after chunking). Verified sizes: KWS 5.9; ResNet-18 q8 12; DINOv2 q4 20.0; SlimSAM quantized 13.8; Whisper-tiny.en q8 40.9 (10.1 + 30.7, chunked); ESM-2 8M q8 8.06; YOLOS-tiny q8 9.66. Sum ≈ 110 MB (derived) | ≤ 150 MB |
| Precomputed traces and media | ≤ 400 MB |
| Headroom | ≥ 300 MB |

- **Bandwidth.** 100 GB/month is only about 4,000 downloads of a 25 MB model, or about 2,400 of Whisper's 41 MB (derived), so caching is essential.
  - Use the Cache API or OPFS with versioned names.
  - Show the size before any download over a few MB ("Load live model (41 MB)").
- **Larger models** go to the builder's own Hugging Face model repos: GPT-2 q8 128 MB, BERT-base, Kokoro 92 MB, CLIP ViT-B/32 q8 153.7 MB.
  - HF resolve URLs are CORS-enabled (verified).
  - The anonymous limit is 3,000 resolver requests per 5 minutes per IP (HF table dated September 2025, marked "subject to change"). A classroom behind one NAT IP shares this quota.
- **HF reachability risk.**
  - huggingface.co reset connections (ECONNRESET) from the builder's network during the research and again during this review.
  - The HF mirror redirects back to huggingface.co.
  - Learners in China may not reach HF at all. So:
    - (a) every page must work in precomputed mode without HF;
    - (b) MVP and V1 live models are self-hosted;
    - (c) HF-hosted models are an opt-in enhancement, and their pages are designed precomputed-first;
    - (d) mainland-China reachability of GitHub Pages and Cloudflare Pages is unverified, so test from target networks before launch.

**Performance budget (design targets):**

| Metric | Target |
|---|---|
| Page shell JS | ≤ 300 KB gzipped (lazy-load per model) |
| First example rendered | ≤ 2 s on a mid laptop, from ≤ 1 MB of precomputed data |
| Auto-loaded live model | ≤ 10 MB on desktop, ≤ 5 MB on phones; 10–100 MB opt-in on desktop; > 100 MB precomputed or explicit opt-in only |
| Input launcher | Loads no model until the learner picks one; previews come from precomputed thumbnails |
| Hover / slider response without a model re-run | One frame (≤ 16 ms); post-processing (softmax, temperature, thresholds, NMS) in JS on cached outputs |
| Tiny-model re-run | ≤ 100 ms |
| Training loops | Always in a worker, not tied to `requestAnimationFrame`; UI stays at 60 fps; warn when the tab is hidden |
| Tensor transfer | Typed arrays plus dims, never nested JS arrays; slice only what is visible. Large WebGPU outputs can stay on the GPU (`preferredOutputLocation: 'gpu-buffer'`) |
| Reference speeds | MiniLM-L6 at seq 512: WebGPU 32 ms vs WASM 378 ms; WebGPU shader compile adds a 1–5 s cold start (secondary source). Rule of thumb: WASM below ~100M params and for per-layer stepping; WebGPU for ViT/BERT/GPT-2 sizes |

**Mobile.**
- **WebGPU status** (gpuweb Implementation Status wiki):
  - Chrome Android 121+ on Android 12+ for ARM, Qualcomm and Intel GPUs only. Imagination GPUs need Chrome 139+ on Android 16+. Samsung Xclipse is still TBD.
  - Firefox ships WebGPU on Windows (141+) and Apple Silicon macOS (145+ on macOS 26+). It is Nightly-only on Intel macOS and Linux, and behind a flag on Android.
  - Safari 26 enables it by default on macOS, iOS, iPadOS and visionOS.
- **iOS WASM memory.** WebKit bug 269777 ("Only ~300 MB of RAM is reliably available to WebAssembly pages") is resolved as a duplicate of 269937, the WASM memory/Gigacage fragmentation "MegaBug". Reported availability: about 1 GB on first load, 300–500 MB after a reload, and as little as 28 MB on an iPhone Xs after a reload. iOS kills the tab when memory runs out.
- **Rules:**
  - precomputed is the default on phones;
  - live models auto-load on phones only for tiny own models (≤ 5 MB); mid models (≤ 25 MB, e.g. DINOv2 q4 20 MB) are opt-in;
  - pointer events for all drawing (GAN Lab's mouse-only pad fails on touch);
  - layout runs top to bottom;
  - tap replaces hover; long-press is the secondary action (e.g. a negative SAM point);
  - test on a real mid-range Android device in M0.

**Accessibility.**
- Colorblind-checked diverging and sequential palettes; no jet; WCAG AA contrast; light and dark themes from design tokens.
- Every hover interaction is also reachable by focus and keyboard: `[` and `]` for layers, arrow keys inside grids, space for play/pause.
- `prefers-reduced-motion` turns animations into step lists.
- Each visualization has a generated text summary ("Top prediction: 'yes' 87%; strongest conv3 channel: 12") and a data-table toggle, announced through ARIA live regions.
- Captions and transcripts for audio. A zh/en toggle on every string.
- Inputs never leave the device (images, audio and text are processed locally), and pages say so. No third-party analytics by default.

**Quality gates in CI:**
- numeric parity tests (JS vs PyTorch, with tolerance written into the spec);
- Playwright end-to-end tests per page;
- a size-budget check per page and for the whole built site;
- link checks;
- a license manifest check (`licenses.json`, one entry per model, dataset, preset, font and adapted figure);
- an "Illustrative" badge lint (no unlabeled fake numbers);
- an `assets.lock.json` sha256 check.

---

## 8. Reuse plan

**Reuse as code or assets, with the exact license and our obligations:**

| Source | License (as verified) | What we take | Obligations |
|---|---|---|---|
| [CNN Explainer](https://github.com/poloclub/cnn-explainer) | MIT | Detail-view Svelte components (KernelMathView, ConvolutionAnimator, PoolAnimator, Softmaxview, Hyperparameterview), `cnn.js` single-op math, the layer-by-layer `apply` idea, `tiny-vgg.py` recipe. Port from Svelte 3 / TF.js 1.4 to Svelte 5 runes | Keep the "Copyright (c) 2020 Polo Club of Data Science" notice. Do **not** reuse the Tiny ImageNet sample images, `data.zip` or the two bundled fonts (licenses unverified) |
| [Transformer Explainer](https://github.com/poloclub/transformer-explainer) | MIT | `export_to_onnx.py` intermediate-output wrapper, `fetchChunks.js` + `chunk.py` (basis for our chunk loader), sampling math, AttentionMatrix / ProbabilityBars / Sankey / Slider components, the textbook tour engine. Migrate from `@xenova/transformers` 2.17 / ORT 1.23 | Keep the "(c) 2022 Polo Club" notice. Regenerate example data ourselves |
| [Diffusion Explainer](https://github.com/poloclub/diffusion-explainer) | MIT | Precompute scripts and layout ideas for the SD page | Notice |
| [GAN Lab](https://github.com/poloclub/ganlab) | Apache-2.0 | `ganlab_models.ts`, input providers, drawing canvas (rewrite with pointer events), evaluators, manifold-cell and gradient-arrow code, the slow-motion pattern, pretrained preset format | Include the license text; keep the Google/Kahng headers; add a NOTICE marking modified files |
| [TensorFlow Playground](https://github.com/tensorflow/playground) | Apache-2.0 | `nn.ts`, `dataset.ts`, `state.ts` (URL-hash state + `*_hide`), `heatmap.ts`, `linechart.ts` | License, headers, change notices |
| [LIT](https://github.com/PAIR-code/lit) | Apache-2.0 | Datapoint/model comparison and counterfactual-editor patterns; salience ↔ embedding linking; possibly UI code | License + NOTICE |
| [VAE Explainer](https://github.com/xnought/vae-explainer), [Backprop Explainer](https://github.com/xnought/backprop-explainer) | MIT | VAE stages, code-to-graph sync; backprop UI ideas and code | Notice |
| [micrograd](https://github.com/karpathy/micrograd) / [micrograd-ts](https://github.com/trekhleb/micrograd-ts) | MIT / MIT | Value engine for the backprop page; the seed of the in-house tensor/autograd core | Notices for Karpathy and Trekhleb |
| [nanochat](https://github.com/karpathy/nanochat) | MIT | Precompute backbone for the lifecycle track: tokenizer → pre-train → SFT → RL checkpoints from one architecture, run at small `--depth` on a single GPU | Notice |
| [ConvNetJS](https://github.com/karpathy/convnetjs) | MIT | classify2d warp view, activation-strip layout, PNG sprite-sheet dataset trick | Notice (MNIST/CIFAR data are not covered) |
| [BertViz](https://github.com/jessevig/bertviz) | Apache-2.0 | Head / model / neuron view JS (port to ES modules) | License, tensor2tensor and HF notices, change notices |
| [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis), [AnimatedLLM](https://github.com/kasnerz/animated-llm), [llm-viz](https://github.com/bbycroft/llm-viz) | MIT (llm-viz excludes homepage and third-party assets) | Token/attention components; trace pipeline idea; walkthrough patterns | Notices |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | MIT | Precompute scripts: hooks, reshape transforms, targets, ROAD metrics, `get_2d_projection` (port to JS for EigenCAM) | Notice. Book prose has no license: paraphrase and link |
| [tfjs-models](https://github.com/tensorflow/tfjs-models) (speech-commands), [tfjs-examples](https://github.com/tensorflow/tfjs-examples) | Apache-2.0 | WebAudio extractor, dataset and serialization code, transfer-learning logic, training notebook; cart-pole, snake-DQN and transfer-recipe code; `visualize-convnet` cam.js pattern | License + NOTICE. Retrain weights ourselves (hosted weights' license unverified). Cite Warden 2018 for Speech Commands (dataset license unverified here) |
| [Transformers.js](https://github.com/huggingface/transformers.js), `@huggingface/tokenizers`, [transformers.js-examples](https://github.com/huggingface/transformers.js-examples) | Apache-2.0 | Runtime, tokenizers, processors; SAM and attention-visualization example code | License + NOTICE |
| [`@huggingface/jinja`](https://github.com/huggingface/huggingface.js) | MIT | Chat-template rendering for the SFT page | Notice. Rewrite the playground UI (its license is ambiguous) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), [onnx-modifier](https://github.com/ZhangGe6/onnx-modifier), [jax-js](https://github.com/ekzhang/jax-js), [TF.js](https://github.com/tensorflow/tfjs) | MIT / MIT / MIT / Apache-2.0 | Runtimes and tooling | Notices |
| [WizMap](https://github.com/poloclub/wizmap), [Netron](https://github.com/lutzroeder/netron), [Model Explorer](https://github.com/google-ai-edge/model-explorer), [TensorSpace](https://github.com/tensorspace-team/tensorspace) | MIT / MIT / Apache-2.0 / Apache-2.0 | Large-scatter rendering; "show the real graph" toggle; TensorSpace layer-glyph ideas | Notices |
| [ManimML](https://github.com/helblazer811/ManimML) | MIT | Optional offline rendering of article figures and reduced-motion stills | Notice |
| [web-rl-playground](https://github.com/awjuliani/web-rl-playground), [Anomagram](https://github.com/victordibia/anomagram), [wevi](https://github.com/ronxin/wevi), [word2viz](https://github.com/lamyiowce/word2viz), [Explained Visually](https://github.com/vicapow/explained-visually), [CREPE](https://github.com/marl/crepe), [harmonic-oscillator-pinn](https://github.com/benmoseley/harmonic-oscillator-pinn), [Jezzamon Fourier](https://github.com/Jezzamonn/fourier), [Weight Atlas](https://github.com/alesha-pro/atlas), [Sequence Toy / Piston](https://github.com/vinhowe/piston) | MIT | Code ports per page. Piston: the WGSL colormap/visualizer and toy-task generators (not the Berkeley Mono font or Typekit) | Notices |
| Young MIT repos: [PINNLab](https://github.com/middesurya/daily-webapp-2026-04-21-pinnlab), [dev48v lessons](https://github.com/dev48v/ai-from-zero), [NNvisual](https://github.com/JadenB9/NNvisual) | MIT (0 stars each) | Ideas first; code only after review | Notices if any code is used |
| [Seeing Theory](https://github.com/seeingtheory/Seeing-Theory), [Understanding UMAP](https://github.com/PAIR-code/understanding-umap) (+ umap-js), [PAIR explorables](https://github.com/PAIR-code/ai-explorables), [Chrome Music Lab](https://github.com/googlecreativelab/chrome-music-lab), [web-audio-recognition](https://github.com/google/web-audio-recognition), [GLTR](https://github.com/HendrikStrobelt/detecting-fake-text), [exBERT](https://github.com/bhoov/exbert), [Neuronpedia](https://github.com/hijohnnylin/neuronpedia), [TensorBoard projector](https://github.com/tensorflow/tensorboard), [Distill GNN intro code](https://github.com/distillpub/post--gnn-intro) | Apache-2.0 | Component ports (GLTR color modes and payload schema; mel extractor; UMAP figure modules) or link-outs | License + NOTICE (do not reuse GLTR's NYTimes text) |
| Distill articles (momentum, t-SNE, CTC, augmented RNNs, feature visualization, building blocks, GNN text and figures) | CC-BY-4.0 | Adapted diagrams | Attribution and a link to the license. Activation Atlas license is unverified, so do not reuse it |
| [labml annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | MIT | Pure-torch reference models (ViT, DDPM U-Net, LSTM, GAT, DCGAN, MHA); `interactive.js` highlighter | Notice. Do not redistribute the `/papers` PDFs |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Apache-2.0 (book and images excluded) | GPTModel, classification and instruction fine-tuning, LoRALayer, DPO functions for precompute | License; cite numbers but re-measure. Never use book or article figures |
| [rlhf-book](https://github.com/natolambert/rlhf-book) | Code and diagrams MIT; text CC-BY-NC-SA-4.0 | Diagram sources to restyle | MIT notice; do not copy the text |
| [Kingy.ai Atlas](https://kingy.ai/atlas/lora-rank/) | Three named diagrams CC BY 4.0 (v2026-09-07): "Follow the low-rank path", "Rank has a storage slope", "Parameters are not VRAM". Tool UI and text not covered | Restyled versions of those three diagrams only | Attribution with version; tool UI is ideas only |
| [d2l](https://github.com/d2l-ai/d2l-zh) figures and TERMINOLOGY.md | Conflicting: d2l-zh LICENSE is Apache-2.0, but config.ini and d2l-en say CC BY-SA 4.0 text + MIT-0 code | Adapted SVGs; glossary seed; deep links | Treat as CC BY-SA 4.0: keep in a separately licensed `/assets/d2l/` folder with attribution. d2l-en code (MIT-0) is free to use |
| [MLU-Explain](https://github.com/aws-samples/aws-mlu-explain) | CC BY-SA 4.0 (articles) + modified MIT (code); GitHub reports NOASSERTION; **archived** 2024-11-12 | Scrollytelling shell ideas, code | Share-alike for adapted text; notice for code; expect no upstream fixes |
| [Mol*](https://github.com/molstar/molstar); [ESM](https://github.com/facebookresearch/esm) | MIT; MIT (code) | Structure viewer; ESM code | Notices. AFDB data and ESM weights licenses unverified |
| Model weights: SlimSAM-77 (apache-2.0), Kokoro-82M (Apache-2.0), Whisper (MIT per openai/whisper) | As listed | Live models | Attribution. eSpeak NG is GPL-3.0 (verified on its repo); whether kokoro-js bundles it is unverified: check before redistributing |
| Other Hub weights (DINOv2-with-registers: ONNX card declares no license, upstream unverified; ResNet, CLIP, ESM-2, YOLOS, GPT-2, BERT) | unverified | Live or precomputed | Check each model card before shipping; record it in `licenses.json` |

**Ideas only (no license, unclear license, copyleft or non-commercial; reimplement from scratch, never copy):**
- **No license detected or stated:**
  - Diffusion Explorer (still unlicensed as of its 2026-09-05 push);
  - GNN 101, REINFORCEjs, Q-Learning Lab, intuitions-about-ddpm, mcts-viz, ML Playground;
  - arogozhnikov's boosting playground, Hyuto YOLOv8, homemade-gpt-js, Autograd UI, recurrentjs;
  - Crucible (young), alignment-lab, gpu_poor, Float Toy, the two-tower demo;
  - ezyang's Convolution Visualizer, nathanrs gpt2-webgl, microgpt-visualizer, Ellena Reid's Whisper interpretability repo;
  - the ngrok posts by Sam Rose (quantization; prompt caching).
- **NOASSERTION, "Other" or unclear:**
  - Neuromancer, LLM Transparency Tool, WebGPT;
  - BuildAnLLM (sources conflict: NOASSERTION vs MIT);
  - AMU applets, the Imperial LSTM demo;
  - LLM-RL-Visualized ("Other", terms unverified);
  - deep-learning-visualized (repo LICENSE is CC BY-NC-SA 4.0; site claims code MIT; ask the author before reusing code).
- **Copyleft or non-commercial:**
  - GPL-3.0: Neural-Visualizer, tfjs-gradcam, limchr YOLO.
  - CC BY-NC-SA 4.0: Jay Alammar posts, deep-learning-visualized content, MLSysBook (textbook and games).
  - AGPL-3.0 + CC BY-NC-SA: Virtual Labs.
  - CC BY-NC-ND: udlbook.
- **All rights reserved:** deeplearning.ai AI Notes (© 2021).
- **Unverified:** Maarten Grootendorst guides, The Illustrated AlphaFold diagrams, Google MLCC content (link only).
- **Attribution only, no formal license:** Asimov Zoo.

Polite permission requests are worth sending to:
- Diffusion Explorer's author (Polo Club co-author; repo actively maintained but unlicensed);
- the AMU author;
- the deep-learning-visualized author (to resolve the code license).

**What we must build ourselves:**
- **Site kit:** spec-driven layout engine; tensor view kit; layer inspector; step player and timeline; formula cards with symbol coloring; provenance badges (with precision and tolerance); input launcher; Playground2D engine; comparison and diff views; bilingual content and glossary; knowledge graph.
- **Runtime:** the in-house tensor/autograd core; the chunked-model loader.
- **Python pipeline:** train, export with named outputs, parity tests, quantize, chunk, `assets.lock.json` manifest, trace precompute, schema-to-TypeScript, GPU-hour logging.
- **Tiny models:**
  - V1: TinyCNN, TinyGPT, flow-2D, MLP, TinyForecaster, matrix factorization, GCN, TinyCLIP, PINN, gridworld/bandit agents.
  - Planned: the rest of §5.
- Retrained KWS (and later snake) weights.
- Concept simulations for the v1 concept pages (§9).
- Our own preset images, audio clips and corpora with known licenses.

---

## 9. MVP and roadmap

**MVP: 4 models from 4 areas + 3 concept pages.** Chosen so that together they exercise every input type and every tensor role in the grammar.

| Piece | Area | Why it is in the MVP | Proves |
|---|---|---|---|
| **TinyCNN** (draw / upload / webcam) | Vision | Most-loved template; tiny and exact within tolerance | Image and drawing input; feature-map tiles; weights slab; kernel arithmetic; color scope |
| **TinyGPT** (+ precomputed GPT-2 small for the same prompts) | Language | Every number computed live; scale jump to GPT-2 | Text input and token chips; vectors and attention matrices; causal mask; output bars and sampling |
| **KWS speech-commands** (retrained) | Speech/audio | A real 5.9 MB model on the learner's own voice; doubles as the fine-tuning demo | Audio input and spectrogram as step 0; parameter budget; frozen vs trainable |
| **2D flow matching** (draw a distribution, train live) | Generative | The engine for the whole generative chapter, and the first user of the in-house autograd core | 2D brush input; live training in a worker; time scrubber; gradients and loss; real vs generated |
| Concept: **Tokenization** | Pre-training | Feeds TinyGPT's input bar; Chinese example | Byte view; side-by-side comparison; BPE stepper |
| Concept: **Next-token prediction and the pre-training loss** | Pre-training | Surprisal coloring, sliding-window examples, training replay | Precomputed checkpoints + live model in one view |
| Concept: **Transfer learning and fine-tuning** | Fine-tuning | Fine-tune KWS on your own words; head-only, then optionally unfreeze dense_1 | Gray/locked vs trainable; layer-wise gradient bars; embedding PCA before and after |

Plus:
- the home page (input launcher + card gallery);
- the atlas skeleton listing all 11 areas (later models marked "Planned");
- glossary seed (~50 terms, zh/en);
- the /about honesty page;
- the license manifest.

**v1 freeze (the public 1.0):**
- **11 model pages:** the 4 MVP models, plus MLP playground, TinyForecaster, matrix factorization, gridworld + bandits, GCN, TinyCLIP and PINN.
- **~11 concept pages:** the 3 MVP concepts, plus backprop/gradient descent, optimizers and LR, overfitting and regularization, self-supervised objectives overview, contrastive learning, LoRA/PEFT, number formats and quantization, catastrophic forgetting.
- Everything else in §5–§6 appears as "Planned" cards only.

**Pedagogy plan:**
- Each page states 2–3 learning goals, a pre/post question pair and one misconception check (§4.2, §4.4).
- The M1 user test is a small structured study rather than a feedback round:
  - 5–10 volunteers (a small user study);
  - each does the pre-question, a 10-minute guided task and the post-question on 3 pages;
  - we log task completion and time, and ask 3 open questions.
- Transformer Explainer and NeuroViz both ran user studies; we follow that practice at small scale.

**Milestones** (durations are rough estimates for one part-time builder):

| Milestone | Content | Exit criteria |
|---|---|---|
| **M0 Foundation** (~3–4 weeks) | Design tokens and grammar; kit (TensorView, LayerCard, StepPlayer, FormulaCard, ModeBadge with tolerance, InputBar text/image); ModelRunner (precomputed + live-infer); in-house autograd core skeleton; Python export, parity and `assets.lock.json` pipeline; trace format; CI with size budget; dual deploy (GitHub Pages + Cloudflare Pages) and the COOP/COEP decision; GPU-hour logging; one vertical slice: TinyCNN end-to-end; real mid-range Android test | TinyCNN works on desktop and phone; parity tolerance published; size check in CI; no weights in git |
| **M1 MVP** (~6–8 weeks) | The remaining 3 models and 3 concept pages; port the needed CNN Explainer / Transformer Explainer components to Svelte 5 and Transformers.js 4.x; home; atlas skeleton; zh/en for UI, presets and glossary; accessibility pass | Structured user study done; fixes applied |
| **M2 v1 breadth** (~12–16 weeks) | 7 flagship model pages (MLP playground, TinyForecaster, matrix factorization, gridworld + bandits, GCN, TinyCLIP, PINN) and 8 concept pages (backprop, optimizers/LR, overfitting, self-supervised overview, contrastive, LoRA/PEFT, number formats + quantization, forgetting); zh article pass; chunk loader prototype | Every area has one live model; shared inspector used by all 11 model pages; **v1.0 release** |
| **Post-v1 backlog** (no dates; one area at a time; a page starts only when the kit covers ~90% of it) | B1: TinyBERT, BERT-base precomputed (reuse the builder's MRPC/RTE slides and checkpoints), TinyViT, ResNet-18 + CAM, DINOv2, word2vec, RNN/LSTM, classic-ML canvas, adversarial examples. B2: Whisper-tiny (chunked), ESM-2 + ESMFold/Mol*, 2D GAN, VAE, cart-pole, GAT/SAGE, sentence embeddings + RAG. B3: fine-tuning and alignment depth (LP vs FT, LLRD, multi-task, SFT mask, adapters, reward model, RLHF toy, DPO, GRPO with nanochat checkpoints, distillation, pruning, Kokoro A/B), interpretability (§6.7), evaluation (§6.8), inference (§6.9). B4: SAM, detector with NMS, seq2seq / encoder-decoder, Mamba-lite, MoE, DDPM U-Net / DiT, SD precomputed, VLM, logit lens, scaling laws, data mixtures, AlphaFold page, AlphaZero-lite, and the planned backlog rows in §5 | Each page passes parity, size, license and study checks before its card switches from "Planned" |

**GPU-hour budget for v1** (design targets, re-measured at M0; logged per page in `pipeline/budgets/`):

| Item | Budget |
|---|---|
| TinyCNN, TinyCLIP, GCN, PINN presets, flow-2D presets, MF, forecaster (training + exports) | ≤ 5 GPU-hours total |
| TinyGPT base training + checkpoints for replay | ≤ 5 GPU-hours |
| KWS retrain (tfjs-models notebook) | ≤ 3 GPU-hours |
| GPT-2 small traces for MVP prompts | ≤ 1 GPU-hour |
| Optimizer/LR sweep on TinyGPT (LR × warm-up × schedule × 3 seeds, small grid) | ≤ 10 GPU-hours |
| Forgetting, LoRA and quantization runs on TinyGPT | ≤ 6 GPU-hours |
| Contingency | ≤ 10 GPU-hours |
| Total v1 | ≤ ~40 GPU-hours |

nanochat lifecycle runs, BERT-base MNLI → RTE, the full PEFT matrix and DPO at GPT-2 scale belong to backlog B3 and get their own budget then.

**Main risks and mitigations:**

| Risk | Mitigation |
|---|---|
| Scope explosion (the catalog lists ~60 model and ~70 concept ideas; Distill paused partly because each article took heavy editorial and design effort) | v1 freeze (11 models, ~11 concepts); kit-first; the 90%-kit-coverage rule; strict tiering; one backlog area at a time |
| Hub models cannot show per-layer internals without re-export | Plan Hub pages as model-level live with precomputed internals; re-export and self-host only where depth is the lesson; keep per-neuron depth for own tiny models |
| Large single files (Whisper decoder 30.7 MB, DINOv2 q8 25.1 MB, CLIP 153.7 MB) | Chunk loader or ONNX external data (M2); CLIP precomputed-first; TinyCLIP as the live default |
| Model size and bandwidth (Transformer Explainer alone uses ~2/3 of the 1 GB Pages limit) | Tiny own models; per-page and site size budgets in CI; chunked mid models; large models only precomputed or on HF |
| Repo bloat (Transformer Explainer's repo ≈ 2.0 GiB) | Weights and traces out of git; `assets.lock.json` fetched at build time from a separate assets repo's releases or HF |
| Hosting headers and reachability | Dual deploy; the COOP/COEP decision at M0; single-threaded WASM as the baseline; test from target networks; HF pages precomputed-first |
| HF unreachable for the builder and for learners in China (resets observed again in this review) | Precomputed mode everywhere; MVP and V1 models self-hosted; HF only as opt-in |
| WebGPU fragmentation and iOS memory collapse | WASM fallback; phones default to precomputed with ≤ 5 MB tiny live models; feature detection with clear messages; a real Android device test in M0 |
| Live-training dependency rot (no TF.js release since 2024-10-21; LiteRT.js is inference-focused; jax-js is young) | In-house tensor/autograd core for own toys; TF.js only for direct ports; pin versions; self-host runtimes |
| Underestimated porting cost (Svelte 3 → 5; Transformers.js 2 → 4; ORT 1.23 → 1.30) | Port only the components the MVP needs; budget porting inside M1 |
| Fake or misleading visuals; overclaimed exactness | Provenance lint; parity tests with published tolerance; "approximate" on quantized runs; no `Math.random()` placeholders; "attention is not explanation" and CAM sanity-check panels |
| Weak evidence base (several references are ≤1-star 2026 repos) | Mark them "(young)"; use them only for ideas; rest credibility on mature references (Polo Club, TF Playground, Distill, BertViz, Transformers.js examples, LIT, nanochat) |
| License contamination (GPL, NC, no-license code) | `licenses.json` manifest checked in CI; ideas-only list (§8); a separate CC BY-SA folder for d2l art; retrain weights with unclear licenses |
| Dataset and preset licensing (Speech Commands citation; MNIST/CIFAR/MovieLens/GloVe terms unverified; eSpeak NG GPL-3.0) | One manifest entry per item; self-recorded or synthetic presets for the MVP so 1.0 ships with no unresolved license |
| Technical errors in explanations | Each page cites its source (paper, d2l, labml); peer review by reviewers; numeric checkpoints learners can verify |
| Learning not measured | Pre/post questions per page; the structured M1 study; repeat on v1 |
| Bilingual cost | English first; glossary-driven zh pass; zh UI, presets and glossary from day one |
| Input launcher as a performance trap | Previews from precomputed thumbnails; load only the chosen model |
| Third-party APIs disappearing (ESMFold, AFDB) | Cache sample results as static fallbacks; label them "third-party live" |
| Burnout and time | MVP is shippable on its own; v1 is a fixed set; the backlog is optional; GitHub Discussions for feedback instead of an ever-growing scope |

---

## 10. Sources

Polo Club and flagship explainers
- https://poloclub.github.io/cnn-explainer/ · https://github.com/poloclub/cnn-explainer · https://raw.githubusercontent.com/poloclub/cnn-explainer/master/package.json
- https://poloclub.github.io/transformer-explainer/ · https://github.com/poloclub/transformer-explainer · https://raw.githubusercontent.com/poloclub/transformer-explainer/main/package.json · https://api.github.com/repos/poloclub/transformer-explainer
- https://poloclub.github.io/diffusion-explainer/ · https://github.com/poloclub/diffusion-explainer
- https://poloclub.github.io/ganlab/ · https://github.com/poloclub/ganlab
- https://poloclub.github.io/ · https://github.com/poloclub
- https://poloclub.github.io/dodrio/ · https://github.com/poloclub/dodrio
- https://poloclub.github.io/wizmap/ · https://github.com/poloclub/wizmap
- https://poloclub.github.io/interactive-classification/ · https://github.com/poloclub/interactive-classification · https://github.com/jaxball/interactive-classification

Multi-model tools and curricula (added in this revision)
- https://pair-code.github.io/lit/ · https://pair-code.github.io/lit/demos/ · https://github.com/PAIR-code/lit · https://api.github.com/repos/PAIR-code/lit
- https://tensorspace.org/ · https://github.com/tensorspace-team/tensorspace
- https://j4den.com/nnvisual/ · https://github.com/JadenB9/NNvisual
- https://developers.google.com/machine-learning/crash-course/exercises · https://developers.google.cn/machine-learning/crash-course?hl=zh-cn
- https://mlsysbook.ai/games/
- https://www.deeplearning.ai/ai-notes/initialization/ · https://www.deeplearning.ai/ai-notes/optimization/
- https://ngrok.com/blog/quantization · https://ngrok.com/blog/prompt-caching/
- https://github.com/karpathy/nanochat
- https://github.com/helblazer811/ManimML
- https://github.com/changyeyu/LLM-RL-Visualized

Vision
- https://adamharley.com/nn_vis/ · https://github.com/aharley/nn_vis
- https://transcranial.github.io/keras-js/ · https://github.com/transcranial/keras-js
- https://cs.stanford.edu/people/karpathy/convnetjs/ · https://github.com/karpathy/convnetjs
- https://carlthome.github.io/tfjs-gradcam/ · https://github.com/carlthome/tfjs-gradcam
- https://jacobgil.github.io/pytorch-gradcam-book · https://github.com/jacobgil/pytorch-grad-cam
- https://vit-explainer.vercel.app/
- https://lyonsno.github.io/webgpu-vit-attention/ · https://github.com/lyonsno/webgpu-vit-attention
- https://github.com/jacobgil/vit-explain
- http://attentionviz.com/ · https://github.com/catherinesyeh/attention-viz
- https://distill.pub/2017/feature-visualization/ · https://github.com/distillpub/post--feature-visualization
- https://distill.pub/2019/activation-atlas/ · https://github.com/distillpub/post--activation-atlas
- https://distill.pub/2018/building-blocks/ · https://github.com/distillpub/post--building-blocks
- https://microscope.openai.com/
- https://fredhohman.com/summit/ · https://github.com/fredhohman/summit
- https://limchr.github.io/yolo_visualization/ · https://github.com/limchr/yolo_visualization
- https://huggingface.co/spaces/webml-community/segment-anything-webgpu · https://huggingface.co/spaces/Xenova/segment-anything-webgpu · https://github.com/huggingface/transformers.js-examples/tree/main/segment-anything-webgpu · https://github.com/facebookresearch/segment-anything/tree/main/demo
- https://ezyang.github.io/convolution-visualizer/ · https://github.com/ezyang/convolution-visualizer
- https://simonnchong.github.io/convolution_visualization_tool/ · https://github.com/simonnchong/convolution_visualization_tool
- https://neuromancer.rudro.me/ · https://github.com/rwdr0/neuromancer
- https://teachablemachine.withgoogle.com/ · https://github.com/googlecreativelab/teachablemachine-community
- http://yosinski.com/deepvis · https://github.com/yosinski/deep-visualization-toolbox
- https://github.com/Hyuto/yolov8-onnxruntime-web
- https://github.com/HendrikStrobelt/miniClip
- https://huggingface.co/api/models/Xenova/yolos-tiny/tree/main/onnx
- https://huggingface.co/api/models/onnx-community/dinov2-with-registers-small-with-attentions

Language and sequence
- https://bbycroft.net/llm · https://github.com/bbycroft/llm-viz
- https://animatedllm.github.io · https://github.com/kasnerz/animated-llm
- https://github.com/jessevig/bertviz
- https://www.neuronpedia.org/gemma-scope · https://github.com/hijohnnylin/neuronpedia
- https://huggingface.co/spaces/Xenova/the-tokenizer-playground · https://github.com/huggingface/transformers.js · https://registry.npmjs.org/@huggingface/tokenizers/latest
- https://bnaskrecki.faculty.wmi.amu.edu.pl/nnets/applets/ · https://bnaskrecki.faculty.wmi.amu.edu.pl/nnets/applets/index.html
- https://jalammar.github.io/ · https://jalammar.github.io/illustrated-transformer/ · https://jalammar.github.io/illustrated-bert/ · https://jalammar.github.io/illustrated-stable-diffusion/ · https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/ · https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- https://huggingface.co/spaces/exbert-project/exbert · https://github.com/bhoov/exbert
- https://github.com/nathanrs/gpt2-webgl
- https://millerharry.github.io/steerscope/ · https://github.com/millerharry/steerscope
- https://ecco.readthedocs.io · https://github.com/jalammar/ecco
- https://huggingface.co/spaces/facebook/llm-transparency-tool-demo · https://github.com/facebookresearch/llm-transparency-tool
- https://pypi.org/project/actlens/ · https://github.com/Evan704/ActLens
- https://alan-cooney.github.io/CircuitsVis/ · https://github.com/TransformerLensOrg/CircuitsVis
- https://microgpt.boratto.ca · https://microgpt.enescang.dev · https://github.com/enescang/microgpt-visualizer
- https://sampling.amanvir.com/
- https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state · https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
- https://martinalderson.com/posts/moe-expert-routing-visualization/
- https://sebastianraschka.com/llm-architecture-gallery/ · https://github.com/rasbt/llm-architecture-gallery
- https://tiktokenizer.vercel.app · https://github.com/dqbd/tiktokenizer
- https://mi-pages.informatik.uni-ulm.de/explornn/ · https://github.com/Sparkier/exploRNN
- https://lstmvis.vizhub.ai/ · https://github.com/HendrikStrobelt/LSTMVis
- https://distill.pub/2019/memorization-in-rnns/ · https://github.com/distillpub/post--memorization-in-rnns
- https://cs.stanford.edu/people/karpathy/recurrentjs/ · https://github.com/karpathy/recurrentjs
- https://arxiv.org/abs/1804.09299 · https://github.com/HendrikStrobelt/Seq2Seq-Vis
- https://distill.pub/2016/augmented-rnns/ · https://github.com/distillpub/post--augmented-rnns
- https://ronxin.github.io/wevi/ · https://github.com/ronxin/wevi
- https://lamyiowce.github.io/word2viz/ · https://github.com/lamyiowce/word2viz
- https://projector.tensorflow.org/ · https://github.com/tensorflow/embedding-projector-standalone · https://github.com/tensorflow/tensorboard
- https://www.cs.cornell.edu/courses/cs4782/2026sp/demos/bytepair
- http://gltr.io · https://github.com/HendrikStrobelt/detecting-fake-text

Speech, audio, time series, recommendation
- https://storage.googleapis.com/tfjs-speech-model-test/2019-01-03a/dist/index.html · https://github.com/tensorflow/tfjs-models
- https://www.tensorflow.org/datasets/catalog/speech_commands · https://arxiv.org/abs/1804.03209
- https://marl.github.io/crepe/ · https://github.com/marl/crepe
- https://huggingface.co/spaces/Xenova/whisper-web · https://github.com/xenova/whisper-web · https://github.com/openai/whisper
- https://huggingface.co/api/models/Xenova/whisper-tiny.en/tree/main/onnx
- https://huggingface.co/spaces/webml-community/kokoro-webgpu · https://github.com/hexgrad/kokoro · https://github.com/espeak-ng/espeak-ng
- https://musiclab.chromeexperiments.com/Spectrogram/ · https://github.com/googlecreativelab/chrome-music-lab
- https://smus.com/web-audio-ml-features · https://github.com/google/web-audio-recognition
- https://er537.github.io/blog/2023/09/05/whisper_interpretability.html · https://github.com/er537/whisper_interpretability
- https://distill.pub/2017/ctc/ · https://github.com/distillpub/post--ctc
- https://www.jezzamon.com/fourier/ · https://github.com/Jezzamonn/fourier
- https://anomagram.fastforwardlabs.com/ · https://github.com/victordibia/anomagram
- https://transport-systems.imperial.ac.uk/resources/cive70111/lec06c_lstm/content.html
- https://jinglescode.github.io/time-series-forecasting-tensorflowjs/ · https://github.com/jinglescode/time-series-forecasting-tensorflowjs
- https://thanushthrony.github.io/projects/recommender-demo.html · https://github.com/thanushthrony/movie-recommender
- https://developers.google.com/machine-learning/recommendation

Classic ML, RL, graphs
- https://playground.tensorflow.org · https://github.com/tensorflow/playground
- https://mlu-explain.github.io/ · https://github.com/aws-samples/aws-mlu-explain
- https://r2d3.us/visual-intro-to-machine-learning-part-1/
- https://seeing-theory.brown.edu/ · https://github.com/seeingtheory/Seeing-Theory
- https://setosa.io/ev/principal-component-analysis/ · https://github.com/vicapow/explained-visually
- https://distill.pub/2016/misread-tsne/ · https://github.com/distillpub/post--misread-tsne
- https://pair-code.github.io/understanding-umap/ · https://github.com/PAIR-code/understanding-umap
- https://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html · https://github.com/arogozhnikov/arogozhnikov.github.io
- https://ml-playground.com/ · https://github.com/josephch405/ML-Playground
- https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
- https://cs.stanford.edu/people/karpathy/reinforcejs/ · https://github.com/karpathy/reinforcejs
- https://awjuliani.github.io/web-rl-playground/ · https://github.com/awjuliani/web-rl-playground
- https://github.com/excel007/qlearning
- https://distill.pub/2019/paths-perspective-on-value-learning/ · https://github.com/distillpub/post--td-paths
- https://storage.googleapis.com/tfjs-examples/cart-pole/dist/index.html · https://github.com/tensorflow/tfjs-examples
- https://vgarciasc.github.io/mcts-viz/ · https://github.com/vgarciasc/mcts-viz
- https://visual-intelligence-umn.github.io/GNN-101/ · https://github.com/Visual-Intelligence-UMN/GNN-101
- https://distill.pub/2021/gnn-intro/ · https://github.com/distillpub/post--gnn-intro
- https://distill.pub/2021/understanding-gnns/ · https://github.com/distillpub/post--understanding-gnns

Generative and multimodal
- https://alechelbling.com/Diffusion-Explorer/ · https://github.com/helblazer811/Diffusion-Explorer
- https://xnought.github.io/vae-explainer/ · https://github.com/xnought/vae-explainer
- https://xnought.github.io/vq-vae-explainer/ · https://github.com/xnought/vq-vae-explainer
- https://iclr-blogposts.github.io/2025/blog/conditional-flow-matching/ · https://github.com/iclr-blogposts/2025
- https://diffusionflow.github.io/
- https://mariogemoll.com/flow-matching · https://github.com/mariogemoll/flow-matching-and-diffusion
- https://parkcheolhee-lab.github.io/intuitions-about-ddpm/ · https://github.com/PARKCHEOLHEE-lab/intuitions-about-ddpm
- https://magenta.tensorflow.org/sketch-rnn-demo · https://github.com/magenta/magenta-js
- https://reiinakano.com/gan-playground/ · https://github.com/reiinakano/gan-playground
- https://gandissect.csail.mit.edu/ · https://github.com/CSAILVision/gandissect
- https://websd.mlc.ai/ · https://github.com/mlc-ai/web-stable-diffusion
- https://huggingface.co/spaces/webml-community/attention-visualization · https://github.com/huggingface/transformers.js-examples
- https://huggingface.co/api/models/Xenova/clip-vit-base-patch32/tree/main/onnx
- https://sliders.baulab.info/ · https://github.com/rohitgandikota/sliders

Science
- https://alphafold.ebi.ac.uk/ · https://github.com/molstar/molstar
- https://esmatlas.com/ · https://github.com/facebookresearch/esm
- https://huggingface.co/api/models/Xenova/esm2_t6_8M_UR50D/tree/main/onnx
- https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/ · https://github.com/ElanaPearl/elanapearl.github.io
- https://pinn-lab.vercel.app · https://github.com/middesurya/daily-webapp-2026-04-21-pinnlab
- https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/ · https://github.com/benmoseley/harmonic-oscillator-pinn

Training concepts and methods
- https://github.com/karpathy/micrograd · https://autograd.hewliyang.com · https://github.com/hewliyang/autograd-ui · https://github.com/trekhleb/micrograd-ts · https://karpathy.github.io/2026/02/12/microgpt/
- https://xnought.github.io/backprop-explainer/ · https://github.com/xnought/backprop-explainer
- https://nn.ameo.dev · https://github.com/Ameobea/neural-network-from-scratch
- https://distill.pub/2017/momentum/ · https://github.com/distillpub/post--momentum
- https://github.com/lilipads/gradient_descent_viz
- https://github.com/tomgoldstein/loss-landscape
- https://understandingdata.com/ai-engineering-visualised/
- https://growingswe.com/blog/microgpt
- https://sequence.toys · https://github.com/vinhowe/piston
- https://trekhleb.dev/homemade-gpt-js/ · https://github.com/trekhleb/homemade-gpt-js
- https://www.buildanllm.com/ · https://github.com/jammastergirish/BuildAnLLM
- https://learn-ml-visualized-interactive.onrender.com/ · https://learn-ml-visualized-interactive.onrender.com/chapter/contrastive-learning
- https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- https://dp-dei.vlabs.ac.in/exp/transfer-learning/simulation/index.html
- https://kingy.ai/atlas/lora-rank/
- https://dev48v.infy.uk/aifromzero.php · https://github.com/dev48v/ai-from-zero · https://dev48v.infy.uk/deeplearningfromzero.php · https://github.com/dev48v/dl-from-zero
- https://crucible.carloperottino.com · https://github.com/The-CarL/llm-lifecycle
- https://github.com/landajuela/alignment-lab
- https://atlas.alesha.pro · https://github.com/alesha-pro/atlas
- https://rahulschand.github.io/gpu_poor/ · https://github.com/RahulSChand/gpu_poor
- https://huggingface.co/spaces/nanotron/ultrascale-playbook
- https://github.com/rasbt/LLMs-from-scratch · https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- https://rlhfbook.com/ · https://github.com/natolambert/rlhf-book
- https://huggingface.co/spaces/huggingfacejs/chat-template-playground · https://github.com/huggingface/huggingface.js
- http://evanw.github.io/float-toy/ · https://github.com/evanw/float-toy
- https://metricgate.com/calculator/knowledge-distillation
- https://pair.withgoogle.com/explorables/ · https://github.com/PAIR-code/ai-explorables

Platforms, catalogs, curricula
- https://distill.pub/ · https://distill.pub/2021/distill-hiatus/ · https://github.com/distillpub/template
- https://zh.d2l.ai · https://github.com/d2l-ai/d2l-zh · https://github.com/d2l-ai/d2l-en · https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html · https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html
- https://nn.labml.ai · https://github.com/labmlai/annotated_deep_learning_paper_implementations
- https://netron.app · https://github.com/lutzroeder/netron
- https://github.com/google-ai-edge/model-explorer
- https://github.com/zetane/viewer
- https://www.asimovinstitute.org/neural-network-zoo/
- https://github.com/Machine-Learning-Tokyo/Interactive_Tools
- https://deeplearning.wizardwu.top · https://github.com/FreeWizardWu/deep-learning-visualized · https://raw.githubusercontent.com/FreeWizardWu/deep-learning-visualized/main/LICENSE
- https://neural-visualizer-ruby.vercel.app/ · https://github.com/PeakScripter/Neural-Visualizer

Runtimes, hosting, browsers
- https://onnxruntime.ai/docs/tutorials/web/ · https://onnxruntime.ai/docs/tutorials/web/deploy.html · https://github.com/microsoft/onnxruntime · https://registry.npmjs.org/onnxruntime-web/1.30.0 · https://app.unpkg.com/onnxruntime-web@1.30.0/files/dist
- https://huggingface.co/docs/transformers.js
- https://ishamf.dev/p/llm-attention-visualizer/ · https://github.com/ishamf/llm-visualizer
- https://github.com/ZhangGe6/onnx-modifier
- https://www.tensorflow.org/js · https://github.com/tensorflow/tfjs · https://api.github.com/repos/tensorflow/tfjs/releases/latest
- https://developers.googleblog.com/en/litertjs-googles-high-performance-web-ai-inference/ · https://github.com/google-ai-edge/LiteRT
- https://webllm.mlc.ai · https://github.com/mlc-ai/web-llm
- https://neuralnetvis.app · https://github.com/daniel-marcus/neuralnetvis
- https://jax-js.com · https://github.com/ekzhang/jax-js
- https://kmeans.org · https://github.com/0hq/WebGPT
- https://huggingface.co/docs/hub/rate-limits
- https://developers.cloudflare.com/pages/configuration/headers/ · https://app.cinevva.com/tutorials/coop-coep-sharedarraybuffer (secondary source for COOP/COEP on Cloudflare Pages)
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
- https://bugs.webkit.org/show_bug.cgi?id=269777 · https://bugs.webkit.org/show_bug.cgi?id=269937

---

## 11. Revision notes

**Missing projects added (all verified 2026-10-05 via the GitHub API or the page itself):**
- **LIT** (#41): Apache-2.0, 3,669 stars, pushed 2026-10-05. Hosted demos: penguins tabular, GLUE BERT.
- **TensorSpace** (#42): Apache-2.0, 5,198 stars, last push 2022-12-05. Site online with LeNet, AlexNet, YOLOv2-tiny, ACGAN and ResNet-50 demos.
- **NNvisual** (#43): MIT, 0 stars, created 2026-08-07.
- **ngrok Quantization** (#44, 2026-03-25) and **ngrok Prompt caching** (#45, 2025-12-16): no license stated.
- **MLSysBook games** (#46): 14 games, CC BY-NC-SA 4.0.
- **deeplearning.ai AI Notes: Initialization** (#47): © 2021; TF.js + D3; 5-layer MNIST net.
- **Google MLCC** (#48): includes fine-tuning/distillation; Simplified-Chinese version confirmed.
- **nanochat** (#49): MIT, 58,431 stars, pushed 2026-09-07; single-GPU via gradient accumulation and `--depth`.
- **ManimML** (#50): MIT, 3,513 stars, last push 2024-06-22.
- **LLM-RL-Visualized** (#51): "Other"/NOASSERTION, 4,933 stars, pushed 2026-09-10.
- LIT, TensorSpace, NNvisual and MLCC are now in §3 "Close to our idea". The summary's novelty claim is narrowed to breadth + concepts track + provenance + bilingual + static.

**Corrections applied:**
- **TF.js:** "maintenance mode" replaced. Verified: latest release `tfjs-v4.22.0` (2024-10-21); README has no maintenance statement; LiteRT.js launched 2026-07-09 as "an evolution from TensorFlow.js" for `.tflite` inference.
- **deep-learning-visualized:** site says code MIT / content CC BY-NC-SA 4.0 and 9 stages · 77 chapters; repo LICENSE is CC BY-NC-SA 4.0 only, repo says 79 chapters; created 2026-09-20.
- **Distill hiatus** reworded (see "partly rejected" below).
- **Sizes and params now verified from HF trees:**
  - Whisper-tiny.en: 39M params, MIT weights; q8 encoder 10,124,913 B + merged decoder 30,727,382 B ≈ 40.9 MB; fp16 sizes.
  - CLIP ViT-B/32: 605.8 MB fp32 → ≈151M derived; q8 153.7 / vision 89.1 / text 64.5 MB. Demoted to precomputed-first, with TinyCLIP as the live default.
  - ESM-2 8M: 30.3 / 15.4 / 8.06 MB.
  - YOLOS-tiny: 26.2 / 13.3 / 9.66 / 7.81 MB.
- **Svelte reuse** reframed as porting; versions verified from both package.json files.
- **Mobile WebGPU:** per-GPU Android status and Firefox desktop status added from the gpuweb wiki.
- **iOS memory:** bug 269777 is a duplicate of 269937; 300–500 MB after reload, down to 28 MB.
- **ORT 1.30.0 runtime files:** all four WASM sizes listed; npm unpacked size 144,618,540 B; which file the WebGPU EP fetches is flagged as needing a check.
- **`@huggingface/tokenizers`:** 360,962 B unpacked; 36.6 KB marked unverified.
- **DINOv2:** the ONNX card declares no license, and its base model is the `imagenet1k-1-layer` variant.
- **Kingy:** CC BY 4.0 limited to three named diagrams (v2026-09-07); "type weight values" removed (inputs are k, d, r, alpha, precision, identical targets).
- **Young markers** on Crucible (0 stars, no license, created 2026-07-08), ai-from-zero (0 stars, created 2026-06-23), PINNLab (0 stars, single day 2026-04-21) and webgpu-vit-attention (1 star, 2026-06-25, no homepage in its metadata).
- **MLU-Explain** marked archived (2024-11-12, NOASSERTION).
- **HF rate limit:** "September '25, subject to change, per IP" added.
- **eSpeak NG:** GPL-3.0 now verified on its repo.
- **Speech Commands:** citation added (Warden 2018).
- **Transformer Explainer repo size:** 2,118,108 KB.

**Design concerns accepted:**
- Scope freeze: v1 = 11 model pages + ~11 concept pages; the rest "Planned"; the 90% kit rule; a backlog without dates.
- Hub per-layer depth requires re-export and self-hosting; the default is model-level live with precomputed internals.
- Size budget fixed with verified files, plus a chunk-loader work item.
- Dual deploy with a COOP/COEP decision at M0.
- Weights and traces out of git via `assets.lock.json`.
- In-house autograd core for live training; TF.js only for direct ports.
- Porting cost budgeted.
- Evidence quality: young markers; mature references carry the plan.
- GPU-hour budget table and a shared backbone (TinyGPT; nanochat for lifecycle).
- Input launcher loads only the chosen model.
- English first with a glossary-driven zh pass.
- Exactness wording with published tolerance; quantized runs labeled approximate.
- Phones limited to ≤ 5 MB auto-loaded live models, plus a real Android test.
- Dataset/preset manifest; self-recorded and synthetic MVP presets.
- Pedagogy: learning goals, pre/post questions, misconception checks, a structured M1 study.

**Missing families and concepts:** accepted into §5–§6 as planned rows. This covers:
- 3D vision, video, robotics/embodied;
- segmentation, depth, OCR, style transfer, adversarial examples;
- T5-style encoder-decoder, retrieval/RAG, CRF, reasoning/agents;
- KV cache, attention variants, positional encodings, speculative decoding, FlashAttention;
- audio codecs/RVQ, YAMNet/AST, VAD, speaker embeddings;
- Chronos/TimesFM, PatchTST, Kalman;
- CTR, SASRec, bandits;
- PPO continuous, world models, imitation/offline RL;
- node2vec, graph transformers, equivariant GNNs, KG embeddings;
- PixelCNN/MaskGIT, DiT, consistency, score matching, EBM/Hopfield/RBM;
- LLaVA connector, cross-attention conditioning, CLAP;
- FNO, weather GNNs, molecules, DNA LMs, protein design, medical segmentation;
- GP, kernel SVM, GMM/EM, HMM, isolation forest, metric learning;
- dropout, residuals, losses, AdamW, clipping, augmentation, early stopping, HP search, grokking;
- ELECTRA/T5/BART/prefix-LM/FIM/MTP/JEPA/BYOL/SimSiam/Barlow/SimCSE and data curation;
- continued pre-training, long context, DoRA, (IA)³, QAT vs PTQ, few-shot/meta, pseudo-labeling, domain adaptation, embedding fine-tuning, RAG vs FT, vision fine-tuning;
- best-of-n, KTO/ORPO/SimPO, CAI/RLAIF, reasoning distillation, reward hacking, editing, unlearning;
- new §6.7 Interpretability, §6.8 Evaluation, §6.9 Inference, §6.10 Robustness/privacy/scale (federated, DP-SGD).

**Critique items rejected or only partly accepted:**
- **Distill: partly rejected.** The editorial does say the bottleneck is "the amount of effort it takes to produce these articles and the unusual combination of scientific and design expertise required". So production cost is part of the reason, not only editor burnout. The new wording includes both.
- **"LiteRT.js is inference only": nuanced.** The launch post also covers conversion (LiteRT Torch) and quantization tooling. The runtime is for inference of converted models, so the practical conclusion (no training replacement) stands.
- **"Make Cloudflare Pages primary": partly accepted.** We dual-deploy and decide at M0. The builder's GitHub Pages stays usable because own tiny models do not need threads, and COEP `require-corp` constrains cross-origin resources. COOP/COEP via `_headers` is supported by Cloudflare's custom-headers feature (up to 100 rules) but confirmed only by secondary sources, so it must be tested.
- **New application areas: not added as separate areas.** Robotics was folded into "RL and control", and 3D and video into "Vision", to keep the 11-area atlas stable under the scope freeze.
- **NNvisual "direct analogue": partly accepted.** The site claims four families with real models, but the repo description mentions only an MLP builder, and its depth and custom input are unverified. It is listed (young) and flagged for re-checking, not treated as a proven competitor.
- **deeplearning.ai note "3-layer and 9-layer nets": not confirmed.** The page shows an adjustable three-layer network and a five-layer MNIST network. The brief describes what was verified.
- **"Transformers.js expects one file per graph": not verified.** The brief says it is unverified whether Transformers.js 4.3 can load chunked buffers or external data, and plans a custom loader as the fallback.
- **Diffusion Explorer "15-month lag": not re-verified.** The figure was removed. Verified instead: 1,190 stars, pushed 2026-09-05, still no license, and the live page shows flow matching only.
- **DINOv2 upstream license:** still unverified (the HF model page returned ECONNRESET during this review).
- **GitHub "1 GB recommended repository size":** not re-verified, so no number is cited. The brief relies on the verified 1 GB published-site limit and Transformer Explainer's measured repo size.