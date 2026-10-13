// View 'decoding/dist': TinyGPT's next-token chances for the prompt (the last row of the trace source 'gpt', whatever the
// step's own source is), reshaped by temperature, top-k and top-p. One row per token, the ten most likely first: its chance
// at temperature 1 (an outline, when `ghost`), after the temperature (the bar), and after the cutoffs, renormalized (the
// last column; "cut" for a dropped token). The top-k and top-p cutoffs are dashed lines between the rows. With `draw`, the
// step's frame (a 'sample' frame of a decode source) adds the random number u on a strip of the kept chances laid end to
// end in vocabulary order (as the kit walks them), and the token it picked. Props:
//   settings?: 'input' | { temperature, topK, topP }   'input' = the learner's settings (default: temperature 1, no cutoffs)
//   ghost?: boolean    outline the chances at temperature 1 behind the bars
//   draw?: boolean     the frame's u and token
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtPct, fmtSetting } from '../../../../atlas/core/format.ts';
import { label, logitsOf, restMass, ROWS, settingsOf, shapeOf, tokOf, V, type Settings } from '../../decode.ts';

type P = Pick<StageProps, 'rt' | 'at' | 'props'>;
export type Shown = Pick<Settings, 'temperature' | 'topK' | 'topP'>;

export interface DistRow {
	id: number; text: string; p1: number; pT: number; final: number; cum: number;
}
// cutoff lines: after row n (counting rows from 1); null when off or below the list
export interface DistPic {
	s: Shown; rows: DistRow[]; kept: number; keptMass: number; kLine: number | null; pLine: number | null; nK: number; nP: number;
	rest: { pT: number; p1: number; kept: number }; greedy: boolean; ghost: boolean;
	draw: { u: number; token: number; text: string; strip: { id: number; text: string; from: number; to: number }[] } | null; max: number;
}

export function settingsFor(p: P): Shown {
	const s = p.props?.settings;
	if (s === 'input') return settingsOf(p.rt?.input);
	if (s && typeof s === 'object') return { temperature: 1, topK: 0, topP: 1, ...(s as Partial<Shown>) };
	return { temperature: 1, topK: 0, topP: 1 };
}

/** What the view draws (pure; also its text form). Null before there is a prompt. */
export function distOf(p: P): DistPic | null {
	const logits = p.rt ? logitsOf(p.rt) : null;
	if (!logits) return null;
	const s = settingsFor(p);
	const sh = shapeOf(logits, s);
	const tok = tokOf(p.rt);
	let cum = 0;
	const rows = sh.order.slice(0, ROWS).map((id) => {
		cum += sh.pT[id];
		return { id, text: label(tok, id), p1: sh.p1[id], pT: sh.pT[id], final: sh.dist[id], cum };
	});
	let restKept = 0;
	for (let r = ROWS; r < sh.order.length; r++) if (sh.dist[sh.order[r]] > 0) restKept++;
	const ghost = p.props?.ghost === true;
	let draw: DistPic['draw'] = null;
	const m = p.at?.frame?.metrics;
	const st = p.at?.state as { token?: number } | undefined;
	if (p.props?.draw && m && typeof st?.token === 'number') {
		let from = 0;
		const strip: { id: number; text: string; from: number; to: number }[] = [];
		// in vocabulary order, as the kit's pickToken walks them
		// (runs of pieces under 0.2% other than the picked one are merged into one span: they look the same, with less DOM)
		for (let id = 0; id < sh.dist.length; id++) {
			const w = sh.dist[id];
			if (w <= 0) continue;
			const prev = strip[strip.length - 1];
			if (w < 0.002 && id !== st.token && prev && prev.id < 0) prev.to += w;
			else strip.push({ id: w < 0.002 && id !== st.token ? -1 : id, text: label(tok, id), from, to: from + w });
			from += w;
		}
		draw = { u: m.u, token: st.token, text: label(tok, st.token), strip };
	}
	const max = Math.max(...rows.map((r) => Math.max(r.pT, ghost ? r.p1 : 0)), 1e-9);
	return {
		s, rows, kept: sh.kept, keptMass: sh.keptMass, kLine: !sh.greedy && sh.nK < V && sh.nK <= ROWS ? sh.nK : null,
		pLine: !sh.greedy && s.topP < 1 && sh.nP < sh.nK && sh.nP <= ROWS ? sh.nP : null, nK: sh.nK, nP: sh.nP,
		rest: { pT: restMass(sh.pT, sh.order, ROWS), p1: restMass(sh.p1, sh.order, ROWS), kept: restKept }, greedy: sh.greedy, ghost, draw, max
	};
}

export function settingsText(s: Shown): I18n {
	const k = s.topK > 0 && s.topK < V ? `top-k ${s.topK}` : null;
	const p = s.topP < 1 ? `top-p ${fmtSetting(s.topP)}` : null;
	const cut = [k, p].filter(Boolean);
	return {
		en: s.temperature <= 0 ? 'temperature 0: greedy' : `temperature ${fmtSetting(s.temperature)}${cut.length ? `, ${cut.join(', ')}` : ', no cutoffs'}`,
		zh: s.temperature <= 0 ? '温度 0：贪心' : `温度 ${fmtSetting(s.temperature)}${cut.length ? `，${cut.join('，')}` : '，不截断'}`
	};
}

export function describeDist(pic: DistPic | null): I18n {
	if (!pic) return { en: 'TinyGPT’s next-token chances: there is no prompt yet.', zh: 'TinyGPT 对下一个词元的概率：还没有提示。' };
	const s = settingsText(pic.s);
	const list = pic.rows.map((r) => `“${r.text}” ${fmtPct(r.pT)}${r.final > 0 ? ` → ${fmtPct(r.final)}` : ' cut'}`).join('; ');
	const listZh = pic.rows.map((r) => `“${r.text}” ${fmtPct(r.pT)}${r.final > 0 ? ` → ${fmtPct(r.final)}` : ' 截掉'}`).join('；');
	const d = pic.draw;
	return {
		en: `Next-token chances with ${s.en}; ${pic.kept} of ${V} tokens can be picked. Most likely first, chance → after the cutoffs: ${list}. The other ${V - pic.rows.length} tokens hold ${fmtPct(pic.rest.pT)}.${d ? ` The random number u = ${Number.isFinite(d.u) ? d.u.toFixed(3) : '—'} picked “${d.text}”.` : ''}`,
		zh: `下一个词元的概率，设置为${s.zh}；${V} 个词元中有 ${pic.kept} 个可以被选中。按可能性从高到低，概率 → 截断后：${listZh}。其余 ${V - pic.rows.length} 个词元共占 ${fmtPct(pic.rest.pT)}。${d ? `随机数 u = ${Number.isFinite(d.u) ? d.u.toFixed(3) : '—'} 选中了“${d.text}”。` : ''}`
	};
}

const meta = {
	key: 'decoding/dist',
	title: { en: 'Next-token chances, reshaped', zh: '重新调整的下一个词元概率' },
	kinds: ['trace', 'process'],
	describe: (p: StageProps) => describeDist(distOf(p)),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		// made-up logits: token i scores 6 − i/2 for the first 20 ids, −4 for the rest (no tokenizer: ids show as #i)
		const data = Float32Array.from({ length: V }, (_, i) => (i < 20 ? 6 - i / 2 : -4));
		const rt = fakeRuntime({ slug: 'decoding', input: { temperature: 0.8, topK: 0, topP: 0.9 }, traceSource: 'gpt', trace: { order: ['logits'], t: { logits: { shape: [1, V], data } } } });
		return { rt, source: 'gpt', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { settings: 'input', ghost: true } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
