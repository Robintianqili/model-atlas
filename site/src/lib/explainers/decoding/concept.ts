// decoding: the concept page around the lesson (atlas/ui/concept/panels.ts ConceptDef). The toy, real and compare panels
// are StepDefs on the lesson's runtime and sources: the learner's prompt and settings (the side widget) drive all three.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { label, logitsOf, settingsOf, shapeOf, tokOf, V } from './decode.ts';
import { plural } from '../../atlas/lesson/say.ts';
import { settingsText } from './views/dist/meta.ts';
import { written } from './lesson.ts';

export const hook: I18n = { en: 'A language model never writes a word by itself. It gives a chance to every token it knows, and a separate rule picks one: always the most likely (greedy), or a random draw that temperature, top-k and top-p reshape. Here every rule works on the same live numbers, TinyGPT’s chances for your prompt, so you can see exactly what each one changes.', zh: '语言模型从不自己写出一个词。它给认识的每个词元一个概率，再由另一条规则选出一个：总选最可能的（贪心），或者按温度、top-k、top-p 调整过的概率随机抽一个。这里每条规则都作用在同一组实时数字上，也就是 TinyGPT 对你的提示给出的概率，所以你能清楚看到每条规则改变了什么。' };

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your settings on the same chances', zh: '你的设置作用在同一组概率上' },
	source: 'gpt',
	at: { layer: 'logits' },
	view: { key: 'decoding/dist', props: () => ({ settings: 'input', ghost: true }) },
	provenance: 'live',
	narration: {
		en: 'With your settings ({set}), {kept} of the {V} tokens can be picked. “{t1}” has {p1|pct} at temperature 1 and {p|pct} after your settings. Change the prompt or the three settings in this section’s input panel and the bars follow.',
		zh: '按你的设置（{set}），{V} 个词元中有 {kept} 个可以被选中。“{t1}”在温度 1 时是 {p1|pct}，按你的设置调整后是 {p|pct}。在本节的输入面板里改动提示或三个设置，条形会跟着变。',
		vars: (rt) => {
			const l = logitsOf(rt);
			if (!l) return { set: '', kept: NaN, V, t1: '', p1: NaN, p: NaN };
			const s = settingsOf(rt.input);
			const x = shapeOf(l, s);
			const top = x.order[0];
			return { set: settingsText(s), kept: x.kept, V, t1: label(tokOf(rt), top), p1: x.p1[top], p: x.dist[top] };
		}
	}
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'TinyGPT writes with your settings', zh: 'TinyGPT 按你的设置写作' },
	source: 'mine',
	at: 'last',
	view: 'decoding/text',
	provenance: 'live',
	narration: {
		en: 'TinyGPT wrote {n} tokens with your settings and seed. Their mean chance under the model was {mean|pct}; the least likely pick was “{low}” at {lowP|pct}.',
		zh: 'TinyGPT 按你的设置和种子写了 {n} 个词元。它们在模型眼里的平均概率是 {mean|pct}；最不可能的选择是“{low}”，只有 {lowP|pct}。',
		vars: (rt, at) => {
			const w = written(at);
			return { n: w.n || NaN, mean: w.mean, low: w.low ? label(tokOf(rt), w.low.id) : '', lowP: w.low?.pModel ?? NaN };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Greedy against your settings', zh: '贪心对比你的设置' },
	source: 'greedy',
	at: 'last',
	compare: { a: 'greedy', b: 'mine', label: [{ en: 'Greedy', zh: '贪心' }, { en: 'Your settings', zh: '你的设置' }] },
	view: 'decoding/text',
	provenance: 'live',
	narration: {
		en: 'From the same prompt, greedy’s picks had a mean chance of {g|pct}, and {gRep} appeared earlier. With your settings: {m|pct}, and {mRep}.',
		zh: '从同一个提示出发，贪心选中的词元平均概率是 {g|pct}，它的文字里有 {gRep} 段四个词元的片段在前面出现过。按你的设置：平均 {m|pct}，这样的片段有 {mRep} 段。',
		vars: (rt, at) => {
			const g = written(at);
			const t = rt.sources.mine?.tape;
			const m = t ? written({ index: t.length - 1, tape: t }) : null;
			return { g: g.mean, gRep: g.n ? plural(g.rep, { one: '{n} stretch of four tokens in its text had', many: '{n} stretches of four tokens in its text had', zh: '{n}' }, 'int') : NaN, m: m?.mean ?? NaN, mRep: m && m.n ? plural(m.rep, { one: '{n} such stretch', many: '{n} such stretches', zh: '{n}' }, 'int') : NaN };
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'Temperature 0 is the most random setting.', zh: '温度 0 是最随机的设置。' },
		truth: { en: 'It is the least random. As the temperature falls toward 0 the top token takes all of the chance, and at 0 decoding is greedy: no random number is drawn at all.', zh: '恰恰相反，它最不随机。温度越接近 0，第一名拿走的概率越多；到了 0 就是贪心解码，根本不抽随机数。' },
		step: 'temp-numbers'
	},
	{
		claim: { en: 'Top-p always keeps the same number of tokens.', zh: 'top-p 总是保留同样多的词元。' },
		truth: { en: 'It keeps as many as it takes to reach p: few where the model is sure (try the prompt “A big dog”) and many where it is not (“The cat sat”). It is top-k that always keeps k.', zh: '它保留的数量是凑够 p 所需的数量：模型有把握时少（试试提示“一只大狗”），没把握时多（“猫坐在”）。总是保留 k 个的是 top-k。' },
		step: 'top-p'
	},
	{
		claim: { en: 'A higher temperature makes the model more creative.', zh: '温度越高，模型越有创造力。' },
		truth: { en: 'It only flattens the same chances; the model knows nothing new. The extra variety comes from picking tokens the model itself ranked low, and too many of those turn the text into nonsense.', zh: '它只是把同一组概率压平了；模型并没有多知道什么。多出来的变化来自选中模型自己排在后面的词元，这样的词元一多，文字就变成胡话。' },
		step: 'too-hot'
	}
];

export const further = [
	{ label: { en: 'TinyGPT, the model on this page, layer by layer', zh: 'TinyGPT：本页用到的模型，逐层讲解' }, href: '/m/tinygpt/' },
	{ label: { en: 'Next-token prediction: how the chances are learned', zh: '下一个词元预测：这些概率是怎么学出来的' }, href: '/train/next-token-prediction/' },
	{ label: { en: 'How to generate text: decoding methods with Transformers (Hugging Face blog)', zh: '如何生成文本：Transformers 中的解码方法（Hugging Face 博客）' }, href: 'https://huggingface.co/blog/how-to-generate' },
	{ label: { en: 'The Curious Case of Neural Text Degeneration (Holtzman et al., 2020): top-p sampling', zh: '《神经文本退化的奇特案例》（Holtzman 等，2020）：top-p 采样' }, href: 'https://arxiv.org/abs/1904.09751' }
];
