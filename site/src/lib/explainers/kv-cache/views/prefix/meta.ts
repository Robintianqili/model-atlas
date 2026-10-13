// View 'kv-cache/prefix': prompts A and B (process 'prefix'), token by token: B's tokens before the first difference reuse
// A's cached keys and values, the rest are computed; bars of B's prefill work from scratch and with the reuse. Props:
//   layers?: boolean   also, layer by layer, the largest difference between A's and B's keys and values for the tokens
//                      both prompts have in the same place after the first difference, and what a reuse of those rows
//                      would do to B's next-token logits
import type { I18n } from '../../../../atlas/types/core.ts';
import type { Frame } from '../../../../atlas/types/process.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';

export function describePrefix(f: Frame | undefined, layers: boolean): I18n {
	const m = f?.metrics;
	if (!m) return { en: 'The two prompts have not been read yet.', zh: '两个提示词还没有读入。' };
	const base: I18n = {
		en: `Prompt B (${m.PB} tokens) shares its first ${m.s} tokens with prompt A (${m.PA}). With A's cache, B's prefill computes ${m.computed} tokens: ${fmtInt(m.macsReuse)} multiply-adds instead of ${fmtInt(m.macsB)}.`,
		zh: `提示词 B（${m.PB} 个词元）的前 ${m.s} 个词元与提示词 A（${m.PA} 个）相同。借助 A 的缓存，B 的预填充只计算 ${m.computed} 个词元：${fmtInt(m.macsReuse)} 次乘加，而不是 ${fmtInt(m.macsB)} 次。`
	};
	if (!layers) return base;
	return {
		en: `${base.en} ${m.nSame} later tokens sit in the same place in both prompts; their keys differ by up to ${fmt3(m.dK0)} in layer 0 and ${fmt3(m.dKdeep)} in the later layers.`,
		zh: `${base.zh}后面有 ${m.nSame} 个词元在两个提示词里位置相同；它们的键在第 0 层最多相差 ${fmt3(m.dK0)}，在后面几层最多相差 ${fmt3(m.dKdeep)}。`
	};
}

const meta = {
	key: 'kv-cache/prefix',
	title: { en: 'Two prompts sharing a prefix', zh: '共享前缀的两个提示词' },
	kinds: ['process'],
	describe: (p: StageProps) => describePrefix(p.at?.frame, !!p.props?.layers),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const { TapeStore } = await import('../../../../atlas/process/tape.ts');
		const { prefix } = await import('../../process.ts');
		const tape = new TapeStore({ process: 'prefix', schema: prefix.schema, phases: prefix.phases, provenance: 'illustrative' });
		const v = (a: number[], n = 40) => ({ shape: [n], data: Float64Array.from({ length: n }, (_, i) => a[i] ?? -1) });
		tape.push({ phase: 'share', step: 0, state: { a: v([735, 363, 258, 345, 315, 528, 265]), b: v([735, 363, 258, 345, 804, 463]), dK: v([0, 1, 1.1, 1.2], 4), dV: v([0, 0.4, 0.9, 0.9], 4) }, metrics: { PA: 7, PB: 6, s: 4, reused: 4, computed: 2, macsA: 2e6, macsB: 1.7e6, macsReuse: 6e5, saved: 1.1e6, nSame: 0, dK0: 0, dKdeep: 0, dWrong: 0, L: 4 } } as never);
		const rt = fakeRuntime({ slug: 'kv-cache', input: {}, tapes: { prefix: tape } });
		return { rt, source: 'prefix', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
