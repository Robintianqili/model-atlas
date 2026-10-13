// View 'tinymoe/route': routing in a mixture-of-experts layer (modes in data.ts).
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { denseOf, gridOf, loadOf, modeOf, tokenOf } from './data.ts';

type P = Omit<StageProps, 'ctx'>;

export function describeRoute(p: P) {
	const mode = modeOf(p);
	if (mode === 'token') {
		const r = tokenOf(p);
		if (!r.p.length) return { en: 'One character’s route: no text yet.', zh: '一个字符的分派：还没有文字。' };
		const ch = r.p.map((v, e) => `${e}: ${fmtPct(v)}`).join(', ');
		return {
			en: `Block ${r.block}, the character “${r.c}”: router chances for experts ${ch}. Kept: experts ${r.a} (gate ${fmt3(r.ga)}) and ${r.b} (gate ${fmt3(r.gb)}); the other two do not run.`,
			zh: `第 ${r.block} 块，字符“${r.c}”：各专家的路由概率 ${ch}。保留：专家 ${r.a}（门控 ${fmt3(r.ga)}）和专家 ${r.b}（门控 ${fmt3(r.gb)}）；另外两个不运行。`
		};
	}
	if (mode === 'load') {
		const l = loadOf(p);
		const per = l.load.map((v, e) => `${e}: ${v}`).join(', ');
		if (p.props?.capacity === false) {
			const sent = l.sent.map((v, e) => `${e}: ${v}`).join(', ');
			const aux = l.balance ? Number(l.balance.aux).toFixed(4) : '';
			return {
				en: `Block ${l.block}: ${l.chars.length} characters routed; slots sent per expert ${sent}.${aux ? ` Balancing loss ${aux}.` : ''}`,
				zh: `第 ${l.block} 块：已分派 ${l.chars.length} 个字符；各专家被送来的位置 ${sent}。${aux ? `负载均衡损失 ${aux}。` : ''}`
			};
		}
		return {
			en: `Block ${l.block}, capacity ${l.cap} slots per expert. ${l.chars.length} characters routed; slots taken per expert ${per}; ${l.dropped} slots dropped.${l.balance ? ` Balancing loss ${fmt3(l.balance.aux)}.` : ''}`,
			zh: `第 ${l.block} 块，每个专家容量 ${l.cap} 个位置。已分派 ${l.chars.length} 个字符；各专家接下的位置 ${per}；丢掉 ${l.dropped} 个位置。${l.balance ? `负载均衡损失 ${fmt3(l.balance.aux)}。` : ''}`
		};
	}
	if (mode === 'dense') {
		const rows = denseOf(p);
		return {
			en: rows.map((r) => `${r.name.en}: ${r.total} parameters, ${r.active} used per character, loss ${r.loss}`).join('; ') + '.',
			zh: rows.map((r) => `${r.name.zh}：${r.total} 个参数，每个字符用到 ${r.active} 个，损失 ${r.loss}`).join('；') + '。'
		};
	}
	const g = gridOf(p);
	if (!g.chars.length) return { en: 'Routing of every character: no text yet.', zh: '每个字符的分派：还没有文字。' };
	const pick = (b: number) => g.chars.map((c) => `${c.c} ${c.g[b].flatMap((v, e) => (v > 0 ? [e] : [])).join('+')}`).join(', ');
	return {
		en: `The experts each block chose for each of your ${g.chars.length} characters. Block 1: ${pick(0)}. Block 2: ${pick(1)}.`,
		zh: `每一块为你的 ${g.chars.length} 个字符分别选中的专家。第 1 块：${pick(0)}。第 2 块：${pick(1)}。`
	};
}

const meta = {
	key: 'tinymoe/route',
	title: { en: 'Routing to experts', zh: '分派给专家' },
	kinds: ['trace', 'process'],
	describe: describeRoute,
	async demo() {
		// made-up gates: 3 characters, 2 blocks, 4 experts (each character keeps 2)
		const gate = (a: number[]) => ({ shape: [3, 4], data: Float32Array.from(a) });
		const rt = fakeRuntime({
			slug: 'tinymoe',
			spec: { slug: 'tinymoe', version: 'demo', layers: [], config: { d: 32, heads: 2, layers: 2, experts: 4, k: 2, hidden: 64, denseHidden: 128, vocab: 4 }, alphabet: ['?', ' ', 'a', 'b'] } as never,
			pre: { tensors: { input: { shape: [3], data: Int32Array.from([2, 1, 3]), dtype: 'i32' as const } }, steps: [], empty: false },
			trace: { order: ['B1.moe#gate', 'B2.moe#gate'], t: { 'B1.moe#gate': gate([0.7, 0.3, 0, 0, 0, 0.6, 0.4, 0, 0.5, 0, 0, 0.5]), 'B2.moe#gate': gate([0, 0.8, 0, 0.2, 0.55, 0.45, 0, 0, 0, 0, 0.9, 0.1]) } }
		});
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'grid' } };
	},
	minHeight: 220
} satisfies ViewMeta;

export default meta;
