# Third-party licence texts

Model Atlas redistributes a few third-party files with the site: tokenizer vocabularies on the tokenization page, the
self-hosted fonts and KaTeX. Their licences ask that the licence text travels with them, so the full texts are here.
`site/static/licenses.json` lists every third-party source (also the ones that only gave an idea) and names the file
below in its `licenseText` field; the licence table on the site's About page links to these files.

| File | Licence | Covers | Copyright |
| --- | --- | --- | --- |
| [Apache-2.0.txt](Apache-2.0.txt) | Apache License 2.0 | the bert-base-uncased WordPiece vocabulary (`site/static/pages/tokenization/tok/bert-base-uncased.txt`) | Google Research, BERT (Devlin et al., 2019) |
| [Apache-2.0-Qwen2.5.txt](Apache-2.0-Qwen2.5.txt) | Apache License 2.0 | the Qwen2.5 tokenizer (`site/static/pages/tokenization/tok/qwen2.5.tokb`) | Copyright 2024 Alibaba Cloud |
| [MIT-GPT-2.txt](MIT-GPT-2.txt) | MIT | the GPT-2 byte-level BPE vocabulary (`site/static/pages/tokenization/tok/gpt2.tokb`) | Copyright (c) 2019 OpenAI |
| [MIT-tiktoken.txt](MIT-tiktoken.txt) | MIT | the tiktoken cl100k_base encoding (`site/static/pages/tokenization/tok/cl100k_base.tokb`) and tiktoken's byte-pair merge, re-implemented in TypeScript | Copyright (c) 2022 OpenAI, Shantanu Jain |
| [MIT-KaTeX.txt](MIT-KaTeX.txt) | MIT | KaTeX: its JavaScript, CSS and fonts, bundled with the site | Copyright (c) 2013-2020 Khan Academy and other contributors |
| [OFL-1.1-Atkinson-Hyperlegible-Next.txt](OFL-1.1-Atkinson-Hyperlegible-Next.txt) | SIL Open Font License 1.1 | Atkinson Hyperlegible Next (body text), self-hosted woff2 files | Copyright 2020-2024 The Atkinson Hyperlegible Next Project Authors |
| [OFL-1.1-JetBrains-Mono.txt](OFL-1.1-JetBrains-Mono.txt) | SIL Open Font License 1.1 | JetBrains Mono (numbers and code), self-hosted woff2 files | Copyright 2020 The JetBrains Mono Project Authors |

Where the texts come from:

- `Apache-2.0.txt`: the Apache License 2.0 as published (the copy in the `transformers` package's LICENSE, without that
  package's own copyright line). `Apache-2.0-Qwen2.5.txt`: the LICENSE file that comes with the Qwen2.5 tokenizer files,
  unchanged.
- `MIT-tiktoken.txt`: the LICENSE of the `tiktoken` package (0.14.0), unchanged.
- `MIT-GPT-2.txt`: OpenAI's copyright line for GPT-2 with the MIT permission notice (the vocabulary's source,
  https://github.com/openai/gpt-2; its copy at https://huggingface.co/openai-community/gpt2 declares the MIT licence).
- `MIT-KaTeX.txt`: the LICENSE of the `katex` package (0.19.0), unchanged.
- `OFL-1.1-*.txt`: the LICENSE files of the `@fontsource-variable/atkinson-hyperlegible-next` and
  `@fontsource-variable/jetbrains-mono` packages (5.3.0), unchanged.

Model Atlas's own work is under the MIT licence (code) and CC BY 4.0 (lesson text, articles and figures); see
`LICENSING.md` at the repository root.
