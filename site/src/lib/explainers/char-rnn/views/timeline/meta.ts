// View 'char-rnn/timeline': the LSTM's memory under the learner's characters (modes in data.ts).
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { show, timelineOf } from './data.ts';

type P = Omit<StageProps, 'ctx'>;

export function describeTimeline(p: P) {
	const g = timelineOf(p);
	if (!g || !g.chars.length) return { en: 'The memory of the LSTM: no text yet.', zh: 'LSTM 的记忆：还没有文字。' };
	const n = g.chars.length;
	if ('fade' in g && g.fade) {
		const f = g.fade;
		return {
			en: `Share of a memory left after d characters, for d from 0 to ${f.D}: the median over all 64 units falls to ${fmtPct(f.keep[f.D])}; unit ${f.patient} keeps the most, ${fmtPct(f.patientCurve[f.D])}.`,
			zh: `记忆在 d 个字符之后还剩的份额（d 从 0 到 ${f.D}）：全部 64 个单元的中位数降到 ${fmtPct(f.keep[f.D])}；单元 ${f.patient} 保留得最多，${fmtPct(f.patientCurve[f.D])}。`
		};
	}
	if ('probs' in g && g.probs) {
		const row = Array.from(g.probs.data.subarray((n - 1) * g.probs.shape[1], n * g.probs.shape[1]));
		const top = row.map((_, k) => k).sort((x, y) => row[y] - row[x]).slice(0, 3).map((k) => `“${show(g.alphabet[k])}” ${fmtPct(row[k])}`);
		return { en: `Chances of the next character after your last character: ${top.join(', ')}.`, zh: `你最后一个字符之后，下一个字符的概率：${top.join('，')}。` };
	}
	if ('cu' in g && g.cu) {
		const w = g.word;
		const v = w ? g.cu[w.to] : NaN;
		return {
			en: `Unit ${g.unit}'s cell value under each of your ${n} characters${w ? `; at the end of “${w.word}” it is ${fmt3(v)}` : ''}. ${g.top.length} top contexts from held-out stories are listed.`,
			zh: `单元 ${g.unit} 的细胞值，写在你的 ${n} 个字符下面${w ? `；在“${w.word}”的末尾是 ${fmt3(v)}` : ''}。下面列出它在没见过的故事里最强的 ${g.top.length} 处。`
		};
	}
	if ('a' in g) {
		const txt = g.a ? g.a.ids.slice(g.a.prompt).map((i) => show(g.alphabet[i])).join('') : '';
		return { en: `After your text it wrote “${txt}”.`, zh: `在你的文字后面，它写出了“${txt}”。` };
	}
	if ('cols' in g) {
		const k = g.guesses.filter((x, s) => x.next !== undefined && x.top === x.next && s <= g.t).length;
		return {
			en: `The LSTM has read ${g.t + 1} of your ${n} characters, one at a time; its top guess for the next character was right ${k} times so far.`,
			zh: `LSTM 已经一个一个地读了你的 ${n} 个字符中的 ${g.t + 1} 个；到目前为止，它对下一个字符的第一猜测对了 ${k} 次。`
		};
	}
	return {
		en: `The hidden state h: 64 numbers after each of your ${n} characters (rows are units, columns are characters). Unit ${g.unit} is marked.`,
		zh: `隐藏状态 h：你的 ${n} 个字符中每读一个后的 64 个数（行是单元，列是字符）。单元 ${g.unit} 已标出。`
	};
}

const meta = {
	key: 'char-rnn/timeline',
	title: { en: 'Memory under the characters', zh: '字符下面的记忆' },
	kinds: ['trace', 'process'],
	describe: describeTimeline,
	async demo() {
		// made-up data: 4 characters, 64 units whose values rise with the unit index
		const T = 4;
		const H = 64;
		const h = Float32Array.from({ length: T * H }, (_, k) => Math.sin(k / 7));
		const rt = fakeRuntime({
			slug: 'char-rnn',
			pre: { tensors: {}, steps: [], empty: false, tokens: ['O', 'n', 'c', 'e'] },
			facts: { 'fact:counterUnit': 51 },
			trace: { order: ['lstm', 'lstm#c'], t: { lstm: { shape: [T, H], data: h }, 'lstm#c': { shape: [T, H], data: h } } }
		});
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'memory' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
