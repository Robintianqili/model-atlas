// decoding: the manifest of a CONCEPT package (/train/decoding/; concept.ts holds the page around the lesson).
// Decoding and sampling on ONE row of live numbers: TinyGPT's next-token logits for the learner's prompt. Temperature,
// top-k and top-p reshape that row (decode.ts, the kit's sampling arithmetic); then TinyGPT writes token by token.
// Sources:
//   gpt     TinyGPT (static/models/tinygpt, read only) on the prompt: the cached logits      (trace, inputs: input)
//   mine    the learner's settings and seed, 48 new tokens                                    (process 'decode')
//   greedy  temperature 0: always the top token, 96 new tokens                                (process 'decode')
//   sample  temperature 1, no cutoffs, seed 8, 96 new tokens                                  (process 'decode')
//   hot     temperature 2, no cutoffs, seed 8, 48 new tokens                                  (process 'decode')
// The decode runs receive TinyGPT's spec and weights (gptInput) and run in the package worker. No Python reference: the
// sampling process is the kit's (its own tests check it against Python), so their badge says "Live".
import type { AtlasSpec, ExplainerManifest, ExplainerRuntime, I18n, Tensor } from '../../atlas/types/index.ts';
import { PROMPTS } from './adapters/prompt/adapter.ts';
import { DEFAULTS, HIGH_T, LONG, SHORT, settingsOf } from './decode.ts';
import type { DecodeInput } from './process.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it),
 *  and the concept catalog fields promote prints (interaction, engine, mode, links). */
type ConceptManifest = Omit<ExplainerManifest, 'type' | 'catalog'> & {
	type: 'concept';
	catalog?: ExplainerManifest['catalog'] & { interaction?: I18n; engine?: I18n; mode?: string; links?: string[] };
};

/** TinyGPT's spec and weights for the worker, copied once per loaded model (each tensor its own buffer; the same object
 *  every time, so a new learner input that leaves a run's prompt and settings alone keeps its run). Copied from the
 *  quantization package (explainer.ts gptInput). */
const copies = new WeakMap<object, { spec: AtlasSpec; weights: Record<string, Tensor> }>();
function model(rt: ExplainerRuntime) {
	const m = rt.sources.gpt?.model;
	if (!m) return null;
	let c = copies.get(m);
	if (!c) {
		const weights: Record<string, Tensor> = {};
		for (const [k, t] of Object.entries(m.weights)) weights[k] = { shape: t.shape.slice(), data: (t.data as Float32Array).slice() };
		c = { spec: JSON.parse(JSON.stringify(m.spec)), weights };
		copies.set(m, c);
	}
	return c;
}
/** The prompt's ids as TinyGPT reads them (the adapter keeps the last 64). */
const idsOf = (rt: ExplainerRuntime): number[] => {
	const t = rt.pre?.tensors?.input;
	return t ? Array.from(t.data as Int32Array) : [];
};
const run = (extra: (rt: ExplainerRuntime) => Partial<NonNullable<DecodeInput>>) => (rt: ExplainerRuntime): DecodeInput => {
	const m = model(rt);
	return m ? { ...m, ids: idsOf(rt), ...extra(rt) } : null;
};
// the learner's settings and seed (settingsOf returns exactly temperature, topK, topP and seed, clamped)
const learner = run((rt) => settingsOf(rt.input));
const plain = run(() => ({}));
const seeded = run(() => ({ seed: DEFAULTS.seed }));

const manifest: ConceptManifest = {
	slug: 'decoding',
	type: 'concept',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	// a copy of TinyGPT's tokenizer.json (claims.test.ts checks it is identical), for the prompt widget and the token text
	assets: { dir: 'models/decoding', auto: ['tokenizer.json'] },
	input: { adapter: 'prompt', presetsFromAdapter: true, start: PROMPTS[0].raw },
	sources: {
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['input'] },
		mine: { kind: 'process', process: 'decode', input: learner, opt: { maxNew: SHORT }, reference: false },
		greedy: { kind: 'process', process: 'decode', input: plain, opt: { maxNew: LONG, temperature: 0 }, reference: false },
		sample: { kind: 'process', process: 'decode', input: seeded, opt: { maxNew: LONG, temperature: 1 }, reference: false },
		hot: { kind: 'process', process: 'decode', input: seeded, opt: { maxNew: SHORT, temperature: HIGH_T }, reference: false }
	},
	page: {
		goals: [
			{ en: 'Turn a language model’s scores (logits) into chances with softmax, and say what dividing them by a temperature does, with real numbers.', zh: '用 softmax 把语言模型的分数（logits）变成概率，并用真实数字说出把它们除以温度会怎样。' },
			{ en: 'Apply top-k and top-p to the same chances: find the cutoff, drop the tokens behind it and renormalize the rest.', zh: '在同一组概率上使用 top-k 和 top-p：找到截断线，去掉线后的词元，再把剩下的重新归一化。' },
			{ en: 'Explain the difference between greedy decoding and sampling with a seed, and why greedy text can get stuck repeating itself.', zh: '解释贪心解码和带种子的采样有什么不同，以及为什么贪心生成的文字会陷入重复。' }
		],
		prequestion: {
			id: 'decoding-pre',
			question: { en: 'A language model has read your prompt. What does it actually give back?', zh: '语言模型读完了你的提示。它实际给出的是什么？' },
			options: [
				{ en: 'The next word', zh: '下一个词' },
				{ en: 'A chance for every token it knows; a separate rule picks one', zh: '它认识的每个词元各一个概率；再由另一条规则选出一个' },
				{ en: 'A whole finished sentence', zh: '一整句写好的话' }
			],
			answer: 1,
			explain: { en: 'The network’s output is one score per token of its vocabulary, turned into chances. Which token gets written is decided afterwards by a decoding rule: always the top one (greedy), or a random draw that temperature, top-k and top-p reshape. The same model writes different text with different rules.', zh: '网络的输出是词表里每个词元一个分数，再变成概率。写出哪个词元，是之后由解码规则决定的：总选第一名（贪心），或者按被温度、top-k、top-p 调整过的概率随机抽一个。同一个模型，换一条规则就会写出不同的文字。' }
		},
		quiz: [
			{
				q: { en: 'Top-k = 3 keeps three tokens whose chances add up to 60%. What chance does the top one, at 30%, get?', zh: 'top-k = 3 保留了三个词元，它们的概率加起来是 60%。排第一、原来是 30% 的那个，现在概率是多少？' },
				options: [{ en: '50%', zh: '50%' }, { en: '30%', zh: '30%' }, { en: '90%', zh: '90%' }], answer: 0,
				explain: { en: 'The kept chances are divided by their total so they add up to 1 again: 30% ÷ 60% = 50%.', zh: '保留下来的概率要除以它们的总和，才能重新加起来等于 1：30% ÷ 60% = 50%。' }
			},
			{
				q: { en: 'Which setting always writes the same text for the same prompt, whatever the seed?', zh: '哪种设置对同一个提示总写出同样的文字，不管种子是多少？' },
				options: [{ en: 'Temperature 0 (greedy)', zh: '温度 0（贪心）' }, { en: 'Temperature 1', zh: '温度 1' }, { en: 'Top-p 0.9', zh: 'top-p 0.9' }], answer: 0,
				explain: { en: 'Greedy takes the top token every time and draws no random number, so the seed plays no part. Every sampling setting draws, so another seed gives another text.', zh: '贪心每次都取第一名，不抽随机数，所以种子不起作用。所有采样设置都要抽随机数，换一个种子就会得到另一段文字。' }
			}
		],
		misconception: {
			claim: { en: 'Greedy decoding, always taking the most likely token, gives the best text.', zh: '贪心解码总选最可能的词元，所以写出的文字最好。' },
			truth: { en: 'Each choice is the most likely one, but nothing stops a loop: once a phrase comes back, the model can come to expect it again, and greedy follows. Sampling can step out of a loop, because now and then it takes a less likely token.', zh: '每一步都选了最可能的词元，但没有任何机制阻止循环：一个短语一旦重新出现，模型可能更期待它再出现，贪心也就跟着走。采样能走出循环，因为它不时会选一个不太可能的词元。' },
			step: 'loops'
		},
		lessonNote: { en: 'Ten steps on one row of TinyGPT’s scores for your prompt: chances, a draw, temperature, top-k, top-p, then whole texts written token by token, greedy against sampling, a loop and a temperature that is too high.', zh: '在 TinyGPT 对你的提示给出的一行分数上走十步：概率、一次抽取、温度、top-k、top-p，然后逐个词元写出整段文字，比较贪心和采样，看一个循环和一个过高的温度。' }
	},
	credits: [
		{ name: 'TinyStories (TinyStoriesV2-GPT4)', url: 'https://huggingface.co/datasets/roneneldan/TinyStories', license: 'CDLA-Sharing-1.0', usedFor: 'the training text of TinyGPT, the model whose logits this page reshapes', attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)' },
		{ name: 'The Curious Case of Neural Text Degeneration (ideas)', url: 'https://arxiv.org/abs/1904.09751', license: 'ideas only (paper; no code or text used)', usedFor: 'top-p (nucleus) sampling, and repetition under greedy decoding', attribution: 'Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes and Yejin Choi (2020)' },
		{ name: 'Hierarchical Neural Story Generation (ideas)', url: 'https://arxiv.org/abs/1805.04833', license: 'ideas only (paper; no code or text used)', usedFor: 'top-k sampling', attribution: 'Angela Fan, Mike Lewis and Yann Dauphin (2018)' },
		{ name: 'Transformer Explainer (ideas)', url: 'https://github.com/poloclub/transformer-explainer', license: 'MIT (ideas only; no code used)', usedFor: 'probability bars with the temperature, top-k and top-p cutoffs drawn on them', attribution: 'Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Zijie J. Wang, Seongmin Lee, Benjamin Hoover and Duen Horng (Polo) Chau, Georgia Tech (2024)' }
	],
	catalog: {
		interaction: { en: 'Temperature, top-k and top-p applied to the same cached TinyGPT logits for your prompt, with the cutoffs drawn and the chances renormalized; greedy against sampling with a fixed seed, token by token; why greedy loops.', zh: '在 TinyGPT 对你的提示算出的同一组缓存 logits 上调节温度、top-k、top-p，画出截断线并重新归一化概率；逐个词元比较贪心和固定种子的采样；看贪心为什么会循环。' },
		engine: { en: 'TinyGPT live', zh: 'TinyGPT 实时' },
		mode: 'live',
		links: ['tinygpt', 'gpt-2', 'whisper-tiny', 'tinyseq2seq']
	},
	glossary: [
		{ id: 'random-seed', term: { en: 'random seed', zh: '随机种子' }, def: { en: 'The starting number of a pseudo-random number generator: the same seed gives the same sequence of “random” numbers, so a sampled text can be reproduced.', zh: '伪随机数生成器的起始数：同一个种子给出同一串“随机”数，所以采样出的文字可以重现。' }, related: ['sampling'] }
	]
};

export default manifest;
