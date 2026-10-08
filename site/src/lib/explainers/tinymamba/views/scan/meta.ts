// tinymamba view 'scan': the learner's row as columns (one per token) with rows of numbers under it, all from the trace:
//   want   the letter each '?' asks for          got   the model's answer at each '?' and its chance (right / wrong)
//   delta  Δ of every channel of block b          dch   Δ of the followed channel only
//   B, C   the write and read strengths (one row per slot)
//   h      the state of the followed channel after each token (one row per slot)
// props: { rows: [...], block (default 1), full (also draw the whole state at the cursor), marks ([{ key, row, col }]:
// outlined cells), note (a caption), tries ([{ label, text }]: buttons that put a row in), bars (the chances at the first ?) }. A step on the 'scan'
// process puts a cursor on its token (frame.step) and follows the process's channel; otherwise lesson.ts's pickCh.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { answers, dec, dims, idsOf, pickCh, T, tok } from '../../tm.ts';
import { isLetter } from '../../task.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export type Grid = { key: string; label: I18n; rows: number; cells: number[][]; kind: 'sequential' | 'diverging' | 'text'; text?: string[][]; tone?: string[]; rowNames: string[] };

export function scanOf(p: P) {
	const rt = p.rt;
	const b = Number(p.props.block ?? 1);
	const ids = idsOf(rt);
	const { D, N } = dims(rt);
	const f0 = p.at.tape && p.at.tape.length ? p.at.tape.frame(0).metrics : undefined;
	const ch = typeof f0?.ch === 'number' ? f0.ch : typeof p.props.ch === 'number' ? p.props.ch : pickCh(rt, b);
	const cur = p.at.frame?.phase === 'token' ? (p.at.frame.step ?? null) : null;
	const n = ids.length;
	const col = <V>(f: (t: number) => V): V[] => Array.from({ length: n }, (_, t) => f(t));
	const dl = T(rt, `B${b}.scan#delta`);
	const hT = T(rt, `B${b}.scan#h`);
	const ans = answers(rt);
	const at = (t: number) => ans.find((a) => a.at === t);
	const slots = Array.from({ length: N }, (_, k) => `n${k}`);
	const grids: Grid[] = [];
	for (const r of (p.props.rows as string[] | undefined) ?? ['delta']) {
		if (r === 'want') grids.push({ key: r, label: { en: 'Should give back', zh: '应当找回' }, rows: 1, kind: 'text', cells: [col(() => 0)], text: [col((t) => (at(t) ? (at(t)!.want >= 0 ? tok(at(t)!.want) : '–') : ''))], rowNames: [''] });
		if (r === 'got') grids.push({ key: r, label: { en: 'The model says', zh: '模型的回答' }, rows: 2, kind: 'text', cells: [col(() => 0), col(() => 0)], text: [col((t) => (at(t) && at(t)!.top >= 0 ? tok(at(t)!.top) : '')), col((t) => (at(t) && at(t)!.top >= 0 ? `${Math.round(at(t)!.p * 100)}%${at(t)!.want < 0 ? '' : at(t)!.ok ? '✓' : '✗'}` : ''))], tone: col((t) => (!at(t) || at(t)!.want < 0 ? '' : at(t)!.ok ? 'ok' : 'bad')), rowNames: ['', ''] });
		if ((r === 'delta' || r === 'dch') && dl) {
			const cs = r === 'delta' ? Array.from({ length: D }, (_, c) => c) : [ch];
			grids.push({ key: r, label: r === 'delta' ? { en: `Δ, block ${b}: one row per channel`, zh: `Δ，第 ${b} 块：每个通道一行` } : { en: `Δ of channel ${ch}`, zh: `通道 ${ch} 的 Δ` }, rows: cs.length, kind: 'sequential', cells: cs.map((c) => col((t) => dl.data[t * D + c])), rowNames: cs.map((c) => (c % 4 === 0 ? `c${c}` : '')) });
		}
		for (const [k, name] of [['B', { en: 'B (write)', zh: 'B（写入）' }], ['C', { en: 'C (read)', zh: 'C（读出）' }]] as const) {
			const x = T(rt, `B${b}.${k}`);
			if (r === k && x) grids.push({ key: r, label: { en: `${name.en}, block ${b}: one row per slot`, zh: `${name.zh}，第 ${b} 块：每个槽一行` }, rows: N, kind: 'diverging', cells: slots.map((_, s) => col((t) => x.data[t * N + s])), rowNames: slots });
		}
		if (r === 'h' && hT) grids.push({ key: r, label: { en: `State h of channel ${ch} after each token`, zh: `每个词元之后通道 ${ch} 的状态 h` }, rows: N, kind: 'diverging', cells: slots.map((_, s) => col((t) => hT.data[(t * D + ch) * N + s])), rowNames: slots });
	}
	const tt = cur ?? (p.at.frame?.phase === 'start' ? -1 : n - 1);
	// on the scan process the state rows fill in up to the token in the frame (the tokens after it are not read yet)
	const upto = p.at.frame?.phase === 'token' || p.at.frame?.phase === 'start' ? tt : n - 1;
	const full = p.props.full && hT && n ? Array.from({ length: D }, (_, c) => slots.map((_, s) => (tt < 0 ? 0 : hT.data[(tt * D + c) * N + s]))) : null;
	const fm = p.at.frame?.metrics;
	const caption: I18n | null =
		cur !== null && fm
			? { en: `Token ${cur} “${tok(ids[cur])}”: in channel ${ch}, Δ = ${dec(fm.delta, 3)}, so the old state is kept × ${dec(fm.decay, 3)}; slot ${f0?.slot} goes from ${dec(fm.hOld, 3)} to ${dec(fm.hNew, 3)}.`, zh: `第 ${cur} 个词元“${tok(ids[cur])}”：通道 ${ch} 的 Δ = ${dec(fm.delta, 3)}，所以旧状态乘以 ${dec(fm.decay, 3)} 保留下来；槽 ${f0?.slot} 从 ${dec(fm.hOld, 3)} 变成 ${dec(fm.hNew, 3)}。` }
			: null;
	const links = ans.filter((a) => a.from >= 0).map((a) => ({ from: a.from, to: a.at }));
	const hint: I18n = grids.every((g) => g.kind === 'text') ? { en: links.length ? 'Each arc joins a letter to the ? that asks for it.' : 'No ? has a letter before it yet.', zh: links.length ? '每条弧线把一个字母连到要它的那个 ?。' : '还没有前面带字母的 ?。' } : { en: 'Hover a cell to read its number.', zh: '把指针移到格子上可以读出数值。' };
	const ask = ans.find((a) => a.want >= 0) ?? ans[0];
	const bars = p.props.bars && ask?.row.length ? { at: ask.at, want: ask.want, items: ask.row.map((v, i) => ({ label: tok(i), p: v, id: i, color: i === ask.want ? 'var(--accent)' : undefined })) } : null;
	const marks = (p.props.marks as { key: string; row: number; col: number }[] | undefined) ?? [];
	const tries = (p.props.tries as { label: I18n; text: string }[] | undefined) ?? [];
	const wantIdx = ids.map((_, t) => links.findIndex((l) => l.to === t));
	return { wantIdx, b, ch, cur, toks: ids.map(tok), letters: ids.map((i) => isLetter(i)), grids, full, fullAt: tt, upto, D, N, caption: caption ?? ((p.props.note as I18n | undefined) || null), links, hint, bars, marks, tries };
}

const meta = {
	key: 'tinymamba/scan',
	title: { en: 'The row, token by token', zh: '这一行，一个词元接一个词元' },
	kinds: ['trace', 'process'],
	describe: (p: P): I18n => {
		const s = scanOf(p);
		const g = s.grids.map((x) => `${x.label.en}: ${x.cells.map((r, i) => (x.kind === 'text' ? x.text![i].filter(Boolean).join(' ') : r.map((v) => fmt3(v)).join(' '))).join(' / ')}`).join('. ');
		const gz = s.grids.map((x) => `${x.label.zh}：${x.cells.map((r, i) => (x.kind === 'text' ? x.text![i].filter(Boolean).join(' ') : r.map((v) => fmt3(v)).join(' '))).join(' / ')}`).join('。');
		return { en: `${s.toks.length} tokens: ${s.toks.join(' ') || 'none'}. ${g}. ${s.caption?.en ?? ''}`, zh: `${s.toks.length} 个词元：${s.toks.join(' ') || '无'}。${gz}。${s.caption?.zh ?? ''}` };
	},
	demo: async () => demoProps({ rows: ['want', 'got', 'dch', 'h'], full: true }),
	minHeight: 160
} satisfies ViewMeta;

export default meta;
