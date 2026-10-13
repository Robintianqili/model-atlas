// decoding: the guided lesson (ten steps; the concept page's primary path). Every number is live: the trace source 'gpt'
// gives TinyGPT's next-token logits for the learner's prompt, decode.ts reshapes them with the kit's sampling arithmetic,
// and the decode sources write text token by token in the package worker.
//   scores -> pick -> temperature (predict) -> temp-numbers -> top-k -> top-p (task) -> generate (play)
//   -> greedy-vs-sample (compare) -> loops (limits) -> too-hot (breakIt)
import type { At, ExplainerRuntime, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtPct, fmtSetting } from '../../atlas/core/format.ts';
import { PROMPTS } from './adapters/prompt/adapter.ts';
import { DEFAULTS, HIGH_T, K_DEMO, LOW_T, P_DEMO, ROWS, V, label, logitsOf, longestLoop, repeats4, restMass, settingsOf, shapeOf, textOf, tokOf, type Settings } from './decode.ts';
import { runOf } from './views/text/meta.ts';
import { settingsText } from './views/dist/meta.ts';

type S = Pick<Settings, 'temperature' | 'topK' | 'topP'>;
const PLAIN: S = { temperature: 1, topK: 0, topP: 1 };
/** a pick the model itself found unlikely: below 1% at temperature 1 */
export const LOW_P = 0.01;

const shaped = (rt: ExplainerRuntime, s: S) => {
	const l = logitsOf(rt);
	return l ? shapeOf(l, s) : null;
};
/** The two most likely next tokens at temperature 1 and their chances. */
function leaders(rt: ExplainerRuntime) {
	const x = shaped(rt, PLAIN);
	const tok = tokOf(rt);
	if (!x) return { t1: '', p1: NaN, t2: '', p2: NaN, top: -1 };
	return { t1: label(tok, x.order[0]), p1: x.p1[x.order[0]], t2: label(tok, x.order[1]), p2: x.p1[x.order[1]], top: x.order[0] };
}
/** The tokens a decode run wrote up to the step's frame, with their chances under the model. */
export function written(at: At) {
	const r = runOf(at.tape, at.index);
	const made = r?.made ?? [];
	const ids = made.map((m) => m.id);
	let low = -1;
	made.forEach((m, i) => {
		if (low < 0 || m.pModel < made[low].pModel) low = i;
	});
	const mean = made.length ? made.reduce((a, m) => a + m.pModel, 0) / made.length : NaN;
	return { made, ids, n: made.length, mean, low: low >= 0 ? made[low] : null, rep: repeats4(ids), rare: made.filter((m) => m.pModel < LOW_P).length };
}
/** The first `w` words of a phrase. */
const words = (s: string, w = 8) => s.split(' ').slice(0, w).join(' ');
/** Whether a new seed can change the text: not when nothing is drawn (temperature 0) or only one token is ever kept. */
const seedNote = (s: Settings): I18n =>
	pick(
		s.temperature <= 0 || s.topK === 1,
		{ en: 'With your settings only one token is ever left, so a new seed changes nothing.', zh: '按你的设置每次只剩一个词元，所以换种子也不会改变什么。' },
		{ en: 'A new seed in the side panel draws another text from the same chances.', zh: '在侧栏里换一个种子，就会从同样的概率里抽出另一段文字。' }
	);
const promptLen = (rt: ExplainerRuntime) => rt.pre?.tensors?.input?.data.length ?? NaN;

const lesson = defineLesson({
	title: { en: 'Decoding and sampling, step by step', zh: '解码与采样，分步课程' },
	summary: { en: 'One row of TinyGPT’s scores for your prompt becomes chances; temperature, top-k and top-p reshape them with the cutoffs drawn; then TinyGPT writes token by token, greedy against sampling with a fixed seed, and you see why greedy loops.', zh: 'TinyGPT 对你的提示算出的一行分数变成概率；温度、top-k、top-p 重新调整它们，并画出截断线；然后 TinyGPT 逐个词元写作，比较贪心和固定种子的采样，并看清贪心为什么会循环。' },
	fixtures: [
		{ id: 'greedy-settings', label: { en: 'temperature 0 (greedy)', zh: '温度 0（贪心）' }, raw: { ...PROMPTS[0].raw, temperature: 0 } },
		{ id: 'hot-settings', label: { en: 'temperature 2, no cutoffs', zh: '温度 2，不截断' }, raw: { ...PROMPTS[0].raw, temperature: 2, topK: 0, topP: 1 } },
		{ id: 'k1', label: { en: 'top-k 1, top-p 0.5, seed 3', zh: 'top-k 1、top-p 0.5、种子 3' }, raw: { ...PROMPTS[3].raw, topK: 1, topP: 0.5, seed: 3 } }
	],
	steps: [
		{
			id: 'scores',
			title: { en: 'Scores become chances', zh: '分数变成概率' },
			source: 'gpt',
			at: { layer: 'logits' },
			view: { key: 'decoding/dist', props: () => ({}) },
			provenance: 'live',
			terms: ['logit', 'softmax', 'vocabulary'],
			narration: {
				en: 'TinyGPT read your {n} and gave each of the {V} tokens it knows a score, a logit. Softmax turns the scores into chances that add up to 1: “{t1}” gets {p1|pct} and “{t2}” {p2|pct}. These {V} chances are all the model says; every rule on this page only reshapes them.',
				zh: 'TinyGPT 读了你的 {n}，给它认识的 {V} 个词元各打一个分数，也就是 logit。softmax 把这些分数变成加起来等于 1 的概率：“{t1}”得到 {p1|pct}，“{t2}”得到 {p2|pct}。模型说出的就只有这 {V} 个概率；本页的每一条规则都只是在重新调整它们。',
				vars: (rt) => {
					const l = leaders(rt);
					return { n: plural(promptLen(rt), { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }), V, t1: l.t1, p1: l.p1, t2: l.t2, p2: l.p2 };
				}
			},
			look: { en: 'The ten most likely next tokens, in order; the last row adds up all the others.', zh: '最可能的十个下一个词元，按顺序排列；最后一行把其余的全部加在一起。' }
		},
		{
			id: 'pick',
			title: { en: 'Greedy, or a draw with a seed', zh: '贪心，或者用种子抽一次' },
			source: 'sample',
			at: { phase: 'sample', step: 0 },
			view: { key: 'decoding/dist', props: () => ({ draw: true }) },
			provenance: 'live',
			terms: ['greedy-decoding', 'sampling', 'random-seed'],
			narration: {
				en: 'Now one token must be written. Greedy decoding always takes the top one, “{t1}”. Sampling lays all the chances end to end from 0 to 1, in vocabulary order rather than by size, each piece as long as its chance, and draws a random number u: here u = {u|n3} lands in the piece of “{picked}”, whose chance is {p|pct}. The random numbers come from a generator started with seed {seed}, so the same seed gives the same draw.',
				zh: '现在要写出一个词元。贪心解码总是取第一名“{t1}”。采样则把所有概率按词表顺序（不是按大小）首尾相接，排成从 0 到 1 的一条带，每一段的长度就是它的概率，再抽一个随机数 u：这里 u = {u|n3}，落在“{picked}”那一段里，它的概率是 {p|pct}。随机数来自一个用种子 {seed} 启动的生成器，所以同一个种子总给出同一次抽取。',
				vars: (rt, at) => {
					const m = at.frame?.metrics ?? {};
					const st = (at.state ?? {}) as { token?: number };
					return { t1: leaders(rt).t1, u: m.u ?? NaN, picked: typeof st.token === 'number' ? label(tokOf(rt), st.token) : '', p: m.p ?? NaN, seed: DEFAULTS.seed };
				}
			},
			look: { en: 'Under the list: the strip of chances; the dark mark is u, and the outlined piece and row are the token it picked.', zh: '列表下方是概率带；深色的标记是 u，加框的那一段和那一行就是它选中的词元。' }
		},
		{
			id: 'temperature',
			title: { en: 'Temperature: guess first', zh: '温度：先猜一猜' },
			source: 'gpt',
			at: { layer: 'logits' },
			view: { key: 'decoding/dist', props: () => ({ settings: { temperature: LOW_T }, ghost: true }) },
			provenance: 'live',
			terms: ['temperature'],
			predict: {
				id: 'decoding-temperature',
				question: { en: 'At T = 0.5 every score is doubled before the softmax. What happens to the top token’s chance?', zh: 'T = 0.5 时，每个分数在 softmax 之前都被放大一倍。排第一的词元的概率会怎样？' },
				options: [
					{ en: 'It grows: the leader pulls further ahead', zh: '变大：领先者被拉得更远' },
					{ en: 'It stays the same: the order of the scores does not change', zh: '不变：分数的先后顺序没有变' },
					{ en: 'It shrinks: the chances even out', zh: '变小：各个概率变得更平均' }
				],
				answer: 0,
				explain: { en: 'Softmax depends on the gaps between the scores, and dividing by 0.5 doubles every gap. So the top token’s share grows and the long tail shrinks. Below 1 the distribution gets sharper; above 1 it gets flatter.', zh: 'softmax 取决于分数之间的差距，除以 0.5 会把每个差距都放大一倍。所以排第一的词元份额变大，长长的尾巴变小。温度低于 1，分布变尖；高于 1，分布变平。' }
			},
			narration: {
				en: 'Before anything is picked, a temperature T can change the chances: every score is divided by T, then softmax runs as before. At T = {one} nothing changes, and “{t1}” keeps {p1|pct}. Answer the question, then compare the bars with their outlines.',
				zh: '在选词之前，可以用温度 T 改变这些概率：每个分数先除以 T，再照常做 softmax。T = {one} 时什么都不变，“{t1}”仍是 {p1|pct}。先回答问题，再把条形和它们的轮廓比一比。',
				vars: (rt) => ({ one: 1, t1: leaders(rt).t1, p1: leaders(rt).p1 })
			},
			look: { en: 'Outlines: the chances at temperature 1. Bars: the same scores divided by the new temperature.', zh: '轮廓：温度为 1 时的概率。条形：同样的分数除以新温度后的概率。' }
		},
		{
			id: 'temp-numbers',
			title: { en: 'Your temperature', zh: '你的温度' },
			source: 'gpt',
			at: { layer: 'logits' },
			view: { key: 'decoding/dist', props: () => ({ settings: 'input', ghost: true }) },
			provenance: 'live',
			terms: ['temperature', 'greedy-decoding'],
			narration: {
				en: '“{t1}” has {p1|pct} at T = {one}, {pLow|pct} at T = {lo} and {pHigh|pct} at T = {hi}, where the {rest} tokens outside the top ten share {restHigh|pct}. As T falls toward 0 the top token takes everything: that is greedy. Your own temperature, {T}, gives it {pMine|pct}; move it in the side panel and watch the bars.',
				zh: '“{t1}”在 T = {one} 时是 {p1|pct}，T = {lo} 时是 {pLow|pct}，T = {hi} 时是 {pHigh|pct}，这时前十名以外的 {rest} 个词元一共分到 {restHigh|pct}。T 越接近 0，第一名拿走的就越多，到 0 就是贪心。你自己的温度 {T} 让它得到 {pMine|pct}；在侧栏里移动它，看条形怎么变。',
				vars: (rt) => {
					const l = leaders(rt);
					const s = settingsOf(rt.input);
					const lo = shaped(rt, { ...PLAIN, temperature: LOW_T });
					const hi = shaped(rt, { ...PLAIN, temperature: HIGH_T });
					const mine = shaped(rt, { ...PLAIN, temperature: s.temperature });
					return {
						t1: l.t1, one: 1, p1: l.p1, lo: fmtSetting(LOW_T), hi: fmtSetting(HIGH_T), pLow: lo ? lo.pT[l.top] : NaN, pHigh: hi ? hi.pT[l.top] : NaN,
						rest: V - ROWS, restHigh: hi ? restMass(hi.pT, hi.order, ROWS) : NaN, T: fmtSetting(s.temperature), pMine: mine ? mine.pT[l.top] : NaN
					};
				}
			},
			look: { en: 'The outlines stay at temperature 1; the bars follow the temperature you set in the side panel.', zh: '轮廓始终是温度 1 时的概率；条形跟着你在侧栏里设的温度变化。' }
		},
		{
			id: 'top-k',
			title: { en: 'Top-k: keep the k most likely', zh: 'top-k：只留最可能的 k 个' },
			source: 'gpt',
			at: { layer: 'logits' },
			view: { key: 'decoding/dist', props: () => ({ settings: { temperature: 1, topK: K_DEMO } }) },
			provenance: 'live',
			terms: ['sampling'],
			narration: {
				en: 'Top-k sampling keeps only the k most likely tokens. With k = {k}, the {k} kept tokens had {mass|pct} of the chance together. Each is divided by that total, so they add up to 1 again: “{t1}” goes from {p1|pct} to {p1k|pct}. Every other token, however sensible, now has no chance at all.',
				zh: 'top-k 采样只保留最可能的 k 个词元。k = {k} 时，保留下来的 {k} 个词元原本一共占 {mass|pct}。每个都除以这个总和，让它们重新加起来等于 1：“{t1}”从 {p1|pct} 变成 {p1k|pct}。其他所有词元，哪怕也说得通，现在都完全没有机会了。',
				vars: (rt) => {
					const l = leaders(rt);
					const x = shaped(rt, { ...PLAIN, topK: K_DEMO });
					return { k: K_DEMO, mass: x ? x.keptMass : NaN, t1: l.t1, p1: l.p1, p1k: x && l.top >= 0 ? x.dist[l.top] : NaN };
				}
			},
			look: { en: 'The dashed line is the cutoff; the right column is each chance after renormalizing.', zh: '虚线就是截断线；右边一列是重新归一化之后的概率。' }
		},
		{
			id: 'top-p',
			title: { en: 'Top-p: keep enough to reach p', zh: 'top-p：留到总和够 p 为止' },
			source: 'gpt',
			at: { layer: 'logits' },
			view: { key: 'decoding/dist', props: () => ({ settings: 'input' }) },
			provenance: 'live',
			terms: ['sampling'],
			task: {
				prompt: { en: 'Set top-k to 0 (off), then move top-p and the temperature until 2 to 5 tokens can still be picked.', zh: '把 top-k 设为 0（关闭），然后调节 top-p 和温度，直到还能被选中的词元有 2 到 5 个。' },
				done: (rt) => {
					const s = settingsOf(rt.input);
					const x = shaped(rt, s);
					return s.topK === 0 && !!x && x.kept >= 2 && x.kept <= 5;
				},
				hint: { en: 'A smaller top-p keeps fewer tokens. A lower temperature puts more of the chance on the top tokens, so top-p reaches its total sooner.', zh: 'top-p 越小，留下的词元越少。温度越低，概率越集中在前几名，top-p 也就越快凑够总和。' },
				waiting: { en: 'Not yet: top-k must be off, with 2 to 5 tokens left.', zh: '还没有：top-k 要关闭，并且只剩 2 到 5 个词元。' },
				doneText: { en: 'Done: top-p alone cut the list to a handful.', zh: '完成：只靠 top-p 就把名单缩到了几个。' }
			},
			narration: {
				en: 'Top-p, or nucleus, sampling keeps the most likely tokens until their chances add up to p. At T = {one} and p = {pd}, your prompt keeps {nP} tokens. The number adapts: few when the model is sure, many when it is not, which a fixed k ignores. Your settings in the side panel ({set}) leave {kept}.',
				zh: 'top-p 采样（又叫核采样）按可能性从高到低保留词元，直到它们的概率加起来达到 p。在 T = {one}、p = {pd} 时，你的提示保留 {nP} 个词元。这个数会自己调整：模型有把握时少，没把握时多，而固定的 k 做不到这一点。你在侧栏里的设置（{set}）留下 {kept}。',
				vars: (rt) => {
					const s = settingsOf(rt.input);
					const d = shaped(rt, { ...PLAIN, topP: P_DEMO });
					const mine = shaped(rt, s);
					return {
						one: 1, pd: fmtSetting(P_DEMO), nP: d ? d.kept : NaN, set: settingsText(s),
						kept: mine ? plural(mine.kept, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }, 'int') : NaN
					};
				}
			},
			look: { en: 'With your settings: the dashed lines mark where top-k and top-p cut.', zh: '按你的设置：虚线标出 top-k 和 top-p 截断的位置。' }
		},
		{
			id: 'generate',
			title: { en: 'Write token by token', zh: '逐个词元地写' },
			source: 'mine',
			at: 'last',
			view: 'decoding/text',
			play: { from: { phase: 'sample', step: 0 }, to: 'last', phases: ['sample'], interval: 160, unit: { en: 'token', zh: '词元' } },
			provenance: 'live',
			terms: ['autoregressive'],
			narration: {
				en: 'Writing is a loop: shape the chances with your settings ({set}), draw one token, append it, run TinyGPT again on the longer text. After {n} tokens, the picks had a mean chance of {mean|pct} under the model; the least likely was “{low}” at {lowP|pct}. {seedNote}',
				zh: '写作就是一个循环：按你的设置（{set}）调整概率，抽出一个词元，接到文字后面，再让 TinyGPT 读一遍更长的文字。写了 {n} 个词元之后，被选中的词元在模型眼里的平均概率是 {mean|pct}；最不可能的是“{low}”，只有 {lowP|pct}。{seedNote}',
				vars: (rt, at) => {
					const w = written(at);
					const s = settingsOf(rt.input);
					return { set: settingsText(s), n: w.n || NaN, mean: w.mean, low: w.low ? label(tokOf(rt), w.low.id) : '', lowP: w.low?.pModel ?? NaN, seedNote: seedNote(s) };
				}
			},
			look: { en: 'Each new token is shaded by how much the model expected it (the scale is in the legend); the outlined one is the newest.', zh: '每个新词元的颜色表示模型有多期待它（色标见图例）；加框的是最新的一个。' }
		},
		{
			id: 'greedy-vs-sample',
			title: { en: 'Greedy against sampling', zh: '贪心对比采样' },
			source: 'greedy',
			at: 'last',
			compare: { a: 'greedy', b: 'sample', label: [{ en: 'Greedy: always the top token', zh: '贪心：总取第一名' }, { en: `Sampling at temperature 1, seed ${DEFAULTS.seed}`, zh: `温度 1 的采样，种子 ${DEFAULTS.seed}` }] },
			view: 'decoding/text',
			provenance: 'live',
			terms: ['greedy-decoding', 'sampling'],
			narration: {
				en: 'The same prompt and {n} tokens each. Greedy took the top token every time: its picks had a mean chance of {g|pct}. Sampling at T = {one} took less likely tokens now and then, a mean of {s|pct}. In greedy’s text {gRep} already appeared earlier; in sampling’s, {sRep}.',
				zh: '同一个提示，各写 {n} 个词元。贪心每次都取第一名：它选中的词元平均概率是 {g|pct}。T = {one} 的采样不时会选不太可能的词元，平均 {s|pct}。贪心的文字里有 {gRep} 段四个词元的片段在前面出现过；采样的文字里有 {sRep} 段。',
				vars: (rt, at) => {
					const g = written(at);
					const sm = rt.sources.sample?.tape;
					const s = sm ? written({ index: sm.length - 1, tape: sm }) : null;
					return { n: g.n || NaN, g: g.mean, one: 1, s: s?.mean ?? NaN, gRep: g.n ? plural(g.rep, { one: '{n} stretch of four tokens had', many: '{n} stretches of four tokens had', zh: '{n}' }, 'int') : NaN, sRep: s && s.n ? s.rep : NaN };
				}
			},
			look: { en: 'One panel per rule; under each text, the candidates its newest token was drawn from.', zh: '每条规则一个面板；每段文字下方列出它最新一个词元的候选。' }
		},
		{
			id: 'loops',
			title: { en: 'Why greedy loops', zh: '贪心为什么会循环' },
			source: 'greedy',
			at: 'last',
			view: { key: 'decoding/text', props: () => ({ loop: true }) },
			provenance: 'live',
			kindOf: 'limits',
			narration: {
				en: 'Greedy is a fixed rule: the same text so far always gives the same next token, whatever the seed. Once a phrase comes back, the text looks more like itself, the model may expect the repeat, and greedy never takes a less likely token to break out. {found}',
				zh: '贪心是一条固定的规则：已有的文字相同，下一个词元就一定相同，与种子无关。一个短语一旦重新出现，文字就更像它自己，模型可能更期待它再重复，而贪心从不选不太可能的词元来打破循环。{found}',
				vars: (rt, at) => {
					const w = written(at);
					if (!w.n) return { found: '' };
					const L = longestLoop(w.ids);
					if (!L) return { found: { en: `For this prompt no phrase of four or more tokens came back within ${w.n} tokens; try the prompt “At the park”.`, zh: `对这个提示，写到 ${w.n} 个词元时还没有四个词元以上的短语重复出现；试试提示“在公园”。` } };
					const mean = (a: number) => w.made.slice(a, a + L.len).reduce((x, m) => x + m.pModel, 0) / L.len;
					const first = mean(L.at[0]);
					const last = mean(L.at[L.at.length - 1]);
					const times = L.at.length === 2 ? 'twice' : `${L.at.length} times`;
					const phrase = words(textOf(tokOf(rt), w.ids.slice(L.start, L.start + L.len)));
					return {
						found: pick(
							last > first,
							{ en: `Here the phrase that starts “${phrase}” (${L.len} tokens) appears ${times}, and the model grows surer of it: its tokens had a mean chance of ${fmtPct(first)} the first time and ${fmtPct(last)} the last time.`, zh: `这里以“${phrase}”开头的短语（${L.len} 个词元）出现了 ${L.at.length} 次，而且模型越来越笃定：它的词元第一次的平均概率是 ${fmtPct(first)}，最后一次是 ${fmtPct(last)}。` },
							{ en: `Here the phrase that starts “${phrase}” (${L.len} tokens) appears ${times}; its tokens had a mean chance of ${fmtPct(first)} the first time and ${fmtPct(last)} the last time.`, zh: `这里以“${phrase}”开头的短语（${L.len} 个词元）出现了 ${L.at.length} 次；它的词元第一次的平均概率是 ${fmtPct(first)}，最后一次是 ${fmtPct(last)}。` }
						)
					};
				}
			},
			look: { en: 'Underlined: every copy of the phrase that comes back.', zh: '下划线：重复出现的短语的每一份。' }
		},
		{
			id: 'too-hot',
			title: { en: 'A temperature that is too high', zh: '温度太高' },
			source: 'hot',
			at: 'last',
			view: { key: 'decoding/text', props: () => ({ rare: LOW_P }) },
			provenance: 'live',
			kindOf: 'breakIt',
			narration: {
				en: 'Raise the temperature too far and the text falls apart. At T = {hi}, the {rest} tokens outside the top ten hold {restHot|pct} of the chance for the first token, against {rest1|pct} at T = {one}. Over {n} tokens, {rare} the model itself gave less than {thr}.',
				zh: '温度升得太高，文字就散架了。T = {hi} 时，第一个词元的概率里，前十名以外的 {rest} 个词元一共占 {restHot|pct}，而 T = {one} 时只占 {rest1|pct}。写了 {n} 个词元，其中有 {rare} 次选中的是模型自己只给了不到 {thr} 的词元。',
				vars: (rt, at) => {
					const w = written(at);
					const hot = shaped(rt, { ...PLAIN, temperature: HIGH_T });
					const one = shaped(rt, PLAIN);
					return {
						hi: HIGH_T, rest: V - ROWS, restHot: hot ? restMass(hot.pT, hot.order, ROWS) : NaN, rest1: one ? restMass(one.pT, one.order, ROWS) : NaN,
						one: 1, n: w.n || NaN, rare: w.n ? plural(w.rare, { one: '{n} pick was a token', many: '{n} picks were tokens', zh: '{n}' }, 'int') : NaN, thr: fmtPct(LOW_P, 0)
					};
				}
			},
			look: { en: 'Dashed boxes: the picks the model itself found unlikely.', zh: '虚线框：模型自己都觉得不太可能的那些选择。' }
		}
	]
});

export default lesson;
