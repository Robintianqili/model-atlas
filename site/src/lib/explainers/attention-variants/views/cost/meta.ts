// View 'attention-variants/cost': what sharing keys and values costs TinyGPT. Props:
//   mode?: 'convert' (default)  the 'eval' frames of process 'share' (every block converted by mean pooling to 4, 2 and
//                               1 key/value heads, no training): perplexity bars on one linear axis from 0, with the KL
//                               divergence from the trained model, the top-1 agreement and the cache per token
//          'uptrain'            facts.json (pipeline/attention-variants/prepare.py, PyTorch, 64,000 validation tokens):
//                               perplexity after 0, 50, 150, 500 and 1,500 uptraining steps for 4, 2 and 1 key/value
//                               heads, as lines on a log axis (evaluation points evenly spaced)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';

export const KV = [4, 2, 1];
export const NAME: Record<number, string> = { 4: 'Mha', 2: 'Gqa', 1: 'Mqa' };
export const LABEL: Record<number, I18n> = { 4: { en: 'MHA, 4 KV heads', zh: 'MHA，4 个 KV 头' }, 2: { en: 'GQA, 2 KV heads', zh: 'GQA，2 个 KV 头' }, 1: { en: 'MQA, 1 KV head', zh: 'MQA，1 个 KV 头' } };
export const STEPS = [0, 50, 150, 500, 1500];

/** The three 'eval' frames' metrics (kv 4, 2, 1), when the tape has reached them. */
export function evalsOf(p: Pick<StageProps, 'at'>) {
	const t = p.at?.tape;
	return KV.map((_, k) => {
		const i = t?.find({ phase: 'eval', step: k }) ?? -1;
		return i >= 0 && i <= (p.at?.index ?? -1) ? (t!.frame(i).metrics ?? null) : null;
	});
}
/** The uptraining curves from facts.json: ppl[kv][step index]. */
export const curvesOf = (p: Pick<StageProps, 'rt'>) => KV.map((kv) => STEPS.map((s) => p.rt.num(`fact:ppl${NAME[kv]}${s}`)));

const meta = {
	key: 'attention-variants/cost',
	title: { en: 'What sharing keys and values costs TinyGPT', zh: '共用键和值让 TinyGPT 付出的代价' },
	kinds: ['process'],
	describe: (p: StageProps) => {
		if (p.props?.mode === 'uptrain') {
			const c = curvesOf(p);
			const en = KV.map((kv, k) => `${LABEL[kv].en}: ` + STEPS.map((s, i) => `${s} steps ${fmt3(c[k][i])}`).join(', ')).join('; ');
			const zh = KV.map((kv, k) => `${LABEL[kv].zh}：` + STEPS.map((s, i) => `${s} 步 ${fmt3(c[k][i])}`).join('，')).join('；');
			return { en: `Validation perplexity while uptraining. ${en}.`, zh: `继续训练过程中的验证集困惑度。${zh}。` };
		}
		const e = evalsOf(p);
		const en = KV.map((kv, k) => (e[k] ? `${LABEL[kv].en}: perplexity ${fmt3(e[k]!.ppl)}, KL ${fmt3(e[k]!.kl)}, top-1 agreement ${fmtPct(e[k]!.agree)}` : '')).filter(Boolean).join('; ');
		const zh = KV.map((kv, k) => (e[k] ? `${LABEL[kv].zh}：困惑度 ${fmt3(e[k]!.ppl)}，KL ${fmt3(e[k]!.kl)}，top-1 一致率 ${fmtPct(e[k]!.agree)}` : '')).filter(Boolean).join('；');
		return { en: `TinyGPT with every block converted, no training: ${en}.`, zh: `所有块都转换、不再训练的 TinyGPT：${zh}。` };
	},
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const { TapeStore } = await import('../../../../atlas/process/tape.ts');
		const { share } = await import('../../process.ts');
		const tape = new TapeStore({ process: 'share', schema: share.schema, phases: share.phases, provenance: 'illustrative' });
		const z = { shape: [4, 9], data: new Float64Array(36) };
		[5, 25, 47].forEach((ppl, k) => tape.push({ phase: 'eval', step: k, state: { w: z, s: z }, metrics: { kv: KV[k], ppl, kl: k, agree: 1 - k / 3, cacheBytes: 2432 / KV[k] } } as never));
		const rt = fakeRuntime({ slug: 'attention-variants', input: {}, tapes: { share: tape } });
		return { rt, source: 'share', at: { index: 2, frame: tape.frame(2), state: tape.state(2), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
