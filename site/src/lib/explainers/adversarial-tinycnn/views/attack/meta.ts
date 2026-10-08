// View 'adversarial-tinycnn/attack': the attack's frames (process.ts) as pictures and numbers. props.mode:
//   grad    your drawing next to the gradient g (diverging; hover or focus a pixel for its number)
//   sign    sign(g): +1, −1 or 0 per pixel
//   step    x + ε·sign(g) = x′, with the ε slider
//   flip    x and x′ side by side, the change x′ − x amplified, and TinyCNN's class shares on each
//   pgd     x_k after PGD step k, the change x_k − x, and p(y) against the best other class over the steps
//   budget  the ε slider, FGSM's x′ and its shares; once the task is done, p(y) for every ε of the scan
//   defend  the training pair adversarial training would add: x′ with your drawing's label
//   limits  the measured table (facts.json): FGSM and PGD on TinyCNN's test drawings, and transfer to TinyViT
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { TABLE, TABLE_EPS, cls, epsText, stateOf, type St } from '../../read.ts';

type P = Omit<StageProps, 'ctx'>;
export const MODES = ['grad', 'sign', 'step', 'flip', 'pgd', 'budget', 'defend', 'limits'] as const;

export function attackOf(p: P) {
	const st = p.at.state as St | undefined, m = p.at.frame?.metrics;
	if (!st?.x || !m) return null;
	const mode = MODES.find((k) => k === p.props.mode) ?? 'grad';
	// the FGSM step at the learner's ε (the budget step's cursor is the scan frame, where adv holds PGD's last drawing)
	const fg = stateOf(p.at.tape, 'fgsm') ?? st;
	const fm = p.at.tape ? (Array.from({ length: p.at.tape.length }, (_, i) => p.at.tape!.frame(i)).find((f) => f.phase === 'fgsm')?.metrics ?? m) : m;
	return { mode, st, m, fg, fm, y: m.y ?? 0 };
}

const name = (p: P, i: number) => cls(p.rt, i);
const fact = (p: P, k: string) => p.rt.num(`fact:${k}`);

export function describeAttack(p: P): I18n {
	const a = attackOf(p);
	if (!a) return { en: 'The attack: nothing to show yet.', zh: '攻击：还没有可显示的内容。' };
	const { m, fm } = a, y = name(p, a.y);
	switch (a.mode) {
		case 'grad':
			return { en: `Your drawing next to the gradient of the loss for every pixel. The largest |g| is ${fmt3(m.gmax)} at row ${m.row}, column ${m.col}; ${fmtPct(m.onInk)} of the gradient’s total size sits on inked pixels.`, zh: `你的画旁边是损失对每个像素的梯度。最大的 |g| 是 ${fmt3(m.gmax)}，在第 ${m.row} 行第 ${m.col} 列；梯度总大小的 ${fmtPct(m.onInk)} 落在有墨的像素上。` };
		case 'sign':
			return { en: `The sign of the gradient: ${m.nPos} pixels +1, ${m.nNeg} pixels −1, ${m.nZero} pixels 0.`, zh: `梯度的符号：${m.nPos} 个像素为 +1，${m.nNeg} 个为 −1，${m.nZero} 个为 0。` };
		case 'pgd': {
			const top = name(p, m.top);
			return { en: `PGD step ${m.k} of ${m.steps}: TinyCNN says ${top.en} (${fmtPct(m.ptop)}); p(${y.en}) = ${fmtPct(m.py)}.`, zh: `PGD 第 ${m.k} 步（共 ${m.steps} 步）：TinyCNN 说是“${top.zh}”（${fmtPct(m.ptop)}）；p(“${y.zh}”) = ${fmtPct(m.py)}。` };
		}
		case 'limits':
			return {
				en: `On TinyCNN’s test drawings, ${TABLE.map((r) => `at ε = ${r.eps} FGSM flips ${fmtPct(fact(p, `fgsm${r.key}`))} and PGD ${fmtPct(fact(p, `pgd${r.key}`))}`).join('; ')}. PGD drawings made on TinyCNN at ε = ${TABLE_EPS} fool TinyViT ${fmtPct(fact(p, 'transferVit'))} of the time.`,
				zh: `在 TinyCNN 的测试画上，${TABLE.map((r) => `ε = ${r.eps} 时 FGSM 翻转 ${fmtPct(fact(p, `fgsm${r.key}`))}、PGD 翻转 ${fmtPct(fact(p, `pgd${r.key}`))}`).join('；')}。ε = ${TABLE_EPS} 时对 TinyCNN 做出的 PGD 画骗过 TinyViT 的比例是 ${fmtPct(fact(p, 'transferVit'))}。`
			};
		default: {
			const top = name(p, fm.top ?? a.y);
			return { en: `Your drawing and the changed drawing x′ (ε = ${epsText(fm.eps)}): TinyCNN says ${top.en} (${fmtPct(fm.ptop)}) on x′; p(${y.en}) went from ${fmtPct(m.py ?? NaN)} to ${fmtPct(fm.py)}.`, zh: `你的画和改过的画 x′（ε = ${epsText(fm.eps)}）：TinyCNN 对 x′ 说是“${top.zh}”（${fmtPct(fm.ptop)}）；p(“${y.zh}”) 从 ${fmtPct(m.py ?? NaN)} 变成 ${fmtPct(fm.py)}。` };
		}
	}
}

const meta = {
	key: 'adversarial-tinycnn/attack',
	title: { en: 'An attack on your drawing', zh: '对你的画的攻击' },
	kinds: ['process'],
	describe: describeAttack,
	async demo() {
		// made-up data: a ring, a made-up gradient around it and its sign (illustrative)
		const x = new Float64Array(784), g = new Float64Array(784);
		for (let i = 0; i < 784; i++) {
			const d = Math.hypot((i % 28) - 13.5, Math.floor(i / 28) - 13.5);
			x[i] = Math.abs(d - 8) < 1.2 ? 1 : 0;
			g[i] = 1e-3 * Math.sin(d) * Math.cos((i % 28) / 3);
		}
		const T = (d: Float64Array) => ({ shape: [1, 28, 28], data: d });
		const state = { x: T(x), g: T(g), s: T(g.map(Math.sign)) };
		const frame = { phase: 'sign', state, metrics: { y: 0, nPos: 380, nNeg: 400, nZero: 4 } };
		return { rt: fakeRuntime({ slug: 'adversarial-tinycnn' }), source: 'attack', at: { index: 0, frame, state }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'sign' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
