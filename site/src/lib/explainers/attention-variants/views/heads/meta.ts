// View 'attention-variants/heads': the head-sharing diagram (illustrative) of the learner's configuration, one row per
// variant: the query heads on top, the key/value heads they read below (one box per KV head spanning its group; for MLA one
// latent box), and each row's cache per token and for the sequence (data.ts, analytic). Props:
//   rows?: ('mine' | 'mha' | 'gqa' | 'mqa' | 'mla')[]   the rows to draw (default ['mine']); 'gqa' keeps the learner's K
//          when it is GQA, else picks a group size of about 4 (no row when H has no divisor between 1 and H), 'mla' DeepSeek-V2's latent (data.ts MLA)
//   preset?: string   draw this published configuration (data.ts PRESETS) instead of the learner's
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { bytesText, configOf, divisors, KIND_SHORT, kindOf, MLA, perToken, PRESETS, presetConfig, total, withKv, type Config } from '../../data.ts';

export type RowId = 'mine' | 'mha' | 'gqa' | 'mqa' | 'mla';

/** The learner's configuration: the cache source's frame when the step has one, else the raw input. */
export const configAt = (p: Pick<StageProps, 'rt' | 'at'> & { props?: Record<string, unknown> }): Config => {
	const pre = PRESETS.find((x) => x.id === p.props?.preset);
	if (pre) return presetConfig(pre);
	const m = p.at?.frame?.metrics;
	return m && Number.isFinite(m.heads) ? configOf({ ...m, preset: 'custom' }) : configOf(p.rt.input);
};

/** A GQA count for H query heads: the divisor of H nearest H / 4 strictly between 1 and H (null when there is none). */
export function gqaKv(h: number): number | null {
	const mid = divisors(h).filter((k) => k > 1 && k < h);
	return mid.length ? mid.reduce((a, b) => (Math.abs(b - h / 4) < Math.abs(a - h / 4) ? b : a)) : null;
}

export function rowsOf(c: Config, ids: RowId[]): { id: RowId; c: Config }[] {
	const out: { id: RowId; c: Config }[] = [];
	for (const id of ids) {
		if (id === 'mine') out.push({ id, c });
		else if (id === 'mha') out.push({ id, c: withKv(c, c.heads) });
		else if (id === 'mqa') out.push({ id, c: withKv(c, 1) });
		else if (id === 'mla') out.push({ id, c: { ...c, latent: MLA.dc + MLA.dr } });
		else {
			const k = kindOf(c) === 1 ? c.kvHeads : gqaKv(c.heads);
			if (k) out.push({ id, c: withKv(c, k) });
		}
	}
	return out;
}

export const rowText = (c: Config): I18n => {
	const kind = KIND_SHORT[kindOf(c)];
	const per = bytesText(perToken(c));
	const all = bytesText(total(c));
	const T = c.tokens.toLocaleString('en-US');
	return kindOf(c) === 3
		? { en: `${kind}: ${c.heads} query heads read one cached latent of ${c.latent} numbers per layer; ${per} per token, ${all} for ${T} tokens`, zh: `${kind}：${c.heads} 个查询头读同一个缓存的潜向量（每层 ${c.latent} 个数）；每个词元 ${per}，${T} 个词元共 ${all}` }
		: { en: `${kind}: ${c.heads} query heads share ${c.kvHeads} key/value head${c.kvHeads === 1 ? '' : 's'} (${c.heads / c.kvHeads} per group); ${per} per token, ${all} for ${T} tokens`, zh: `${kind}：${c.heads} 个查询头共用 ${c.kvHeads} 个键值头（每组 ${c.heads / c.kvHeads} 个）；每个词元 ${per}，${T} 个词元共 ${all}` };
};

const ids = (p: StageProps) => (Array.isArray(p.props?.rows) ? (p.props.rows as RowId[]) : (['mine'] as RowId[]));

const meta = {
	key: 'attention-variants/heads',
	title: { en: 'Which heads share keys and values', zh: '哪些头共用键和值' },
	kinds: ['process'],
	describe: (p: StageProps) => {
		const rows = rowsOf(configAt(p), ids(p)).map((r) => rowText(r.c));
		return { en: rows.map((r) => r.en).join('. ') + '.', zh: rows.map((r) => r.zh).join('。') + '。' };
	},
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const rt = fakeRuntime({ slug: 'attention-variants', input: { preset: 'llama3-8b', layers: 32, heads: 32, kvHeads: 8, dHead: 128, tokens: 4096, bytes: 2, latent: 0 }, tapes: {} });
		return { rt, source: 'cache', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { rows: ['mha', 'gqa', 'mqa'] } };
	},
	minHeight: 240
} satisfies ViewMeta;

export default meta;
