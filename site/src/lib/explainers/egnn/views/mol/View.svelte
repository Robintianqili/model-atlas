<!-- egnn 'mol' view: the molecule on the plane in six modes (meta.ts). Atoms are coloured by kind and numbered from 0;
     the network's arrows are solid, the toy formula's dashed. In 'turn' the slider changes raw.turn (setInput). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n, StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { focus, mol, pushes, STRETCH, variant, wrap } from '../../look.ts';
	import { angleOf, centre, energy, K, KIND_LABEL, type MolRaw } from '../../mol.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'atoms'));
	const m = $derived(mol(P.rt));
	const at = $derived(focus(P.rt));
	const L = (en: string, zh: string) => t({ en, zh });
	const A = 1.5; // molecule units of arrow per unit of force (an arrow is at most CAP long)
	const CAP = 1.6;
	const fm = $derived(mode === 'spin' ? P.at.frame?.metrics : undefined);
	const fs = $derived(mode === 'spin' ? (P.at.state as { pos?: number[]; f?: number[] } | undefined) : undefined);
	const mir = $derived(mode === 'mirror' || mode === 'stretch' ? variant(P.rt, mode) : null);
	const pu = $derived(mode === 'push' && m ? pushes(P.rt, at) : null);
	type Pane = { pos: ArrayLike<number>; f?: ArrayLike<number>; tf?: ArrayLike<number>; title: I18n; E: number; trueE?: number };
	const panes = $derived.by((): Pane[] => {
		if (!m) return [];
		if (mode === 'spin') return fs?.pos && fs.f && fm ? [{ pos: fs.pos, f: fs.f, title: { en: `Turned by ${fm.deg}°`, zh: `转了 ${fm.deg}°` }, E: fm.E }] : [];
		const own: Pane = { pos: m.pos, f: mode === 'atoms' || mode === 'push' || mode === 'stretch' ? undefined : m.f, tf: mode === 'forces' ? m.trueF : undefined, title: { en: 'Your molecule', zh: '你的分子' }, E: m.E, trueE: mode === 'forces' || mode === 'stretch' ? m.trueE : undefined };
		if (mode === 'mirror' && mir) return [own, { pos: mir.pos, f: mir.f, title: { en: 'Its mirror image', zh: '它的镜像' }, E: mir.E }];
		if (mode === 'stretch' && mir) return [own, { pos: mir.pos, title: { en: `Every distance × ${STRETCH}`, zh: `每个距离 × ${STRETCH}` }, E: mir.E, trueE: energy(mir.pos, m.kind).E }];
		return [own];
	});
	// one frame for every pane: the centre and the farthest atom do not change when the molecule turns or is mirrored
	const box = $derived.by(() => {
		if (!m) return '-3 -3 6 6';
		const g = mode === 'stretch' ? STRETCH : 1;
		const { cx, cy } = centre(m.pos);
		let r = 0;
		for (let i = 0; i < m.n; i++) r = Math.max(r, Math.hypot(m.pos[2 * i] - cx, m.pos[2 * i + 1] - cy));
		const R = g * r + 1.3;
		return `${cx - R} ${-cy - R} ${2 * R} ${2 * R}`;
	});
	/** an arrow from (x, y) along (dx, dy) in molecule units, y up: a line and a V head */
	function arrow(x: number, y: number, dx: number, dy: number) {
		const raw = Math.hypot(dx, dy);
		if (raw < 1e-6) return '';
		const k = Math.min(1, CAP / raw), len = raw * k;
		(dx *= k), (dy *= k);
		const x1 = x + dx, y1 = y + dy, h = Math.min(0.2, 0.45 * len), ux = dx / len, uy = dy / len;
		const hx = (s: number) => x1 - h * ux + s * 0.6 * h * -uy, hy = (s: number) => y1 - h * uy + s * 0.6 * h * ux;
		return `M${x},${-y}L${x1},${-y1}M${hx(1)},${-hy(1)}L${x1},${-y1}L${hx(-1)},${-hy(-1)}`;
	}
	const raw = $derived((P.rt.input ?? {}) as MolRaw);
	const deg = $derived(Number(raw.turn ?? 0));
	const set = (r: MolRaw) => (P.setInput ? P.setInput(r) : P.rt.setInput(r));
	const ang = $derived(m ? angleOf(m.f[2 * at], m.f[2 * at + 1]) : NaN);
	const spinRows = $derived(mode === 'spin' && P.at.tape ? Array.from({ length: Math.min(P.at.tape.length, P.at.index + 1) }, (_, k) => P.at.tape!.frame(k).metrics ?? {}) : []);
	const rows = $derived(m ? Array.from({ length: Math.min(3, m.n) }, (_, i) => `${i}: ${t(KIND_LABEL(m.kind[i]))}, (${fmt3(m.pos[2 * i])}, ${fmt3(m.pos[2 * i + 1])})`).join('; ') + (m.n > 3 ? '; …' : '') : '');
</script>

{#if !m}
	<p class="muted">{L('Place some atoms, or pick a molecule, to see what the network does with them.', '放几个原子，或者选一个分子，看看网络怎么处理它们。')}</p>
{:else}
	<div class="mol">
		<div class="panes" class:two={panes.length > 1}>
			{#each panes as p, k (k)}
				<figure>
					<figcaption>{t(p.title)}</figcaption>
					<svg viewBox={box} role="img" aria-label="{t(p.title)}: {m.n} {L('atoms', '个原子')}. {L('Energy', '能量')} {fmt3(p.E)}">
						{#if mode === 'atoms'}
							{#each { length: m.n } as _, i (i)}
								{#each { length: i } as _, j (j)}
									<line class="pair" x1={p.pos[2 * i]} y1={-p.pos[2 * i + 1]} x2={p.pos[2 * j]} y2={-p.pos[2 * j + 1]} />
								{/each}
							{/each}
						{/if}
						{#if pu}
							{#each pu.terms as q (q.j)}
								<line class="pair" x1={p.pos[2 * at]} y1={-p.pos[2 * at + 1]} x2={p.pos[2 * q.j]} y2={-p.pos[2 * q.j + 1]} />
								<path class="term" d={arrow(p.pos[2 * at], p.pos[2 * at + 1], A * q.x, A * q.y)} />
							{/each}
						{/if}
						{#each { length: m.n } as _, i (i)}
							<circle class="atom" class:at={i === at && mode !== 'atoms'} cx={p.pos[2 * i]} cy={-p.pos[2 * i + 1]} r="0.24" style:fill="var(--cat-{m.kind[i]})" />
							<text x={p.pos[2 * i] + 0.27} y={-p.pos[2 * i + 1] - 0.22}>{i}</text>
						{/each}
						{#if p.f}
							{#each { length: m.n } as _, i (i)}<path class="net" class:thin={mode !== 'forces' && i !== at} d={arrow(p.pos[2 * i], p.pos[2 * i + 1], A * p.f[2 * i], A * p.f[2 * i + 1])} />{/each}
						{/if}
						<!-- the formula's dashed arrows on top, so they still show where they lie along a longer network arrow -->
						{#if p.tf}
							{#each { length: m.n } as _, i (i)}<path class="true" d={arrow(p.pos[2 * i], p.pos[2 * i + 1], A * p.tf[2 * i], A * p.tf[2 * i + 1])} />{/each}
						{/if}
						{#if pu}<path class="net" d={arrow(p.pos[2 * at], p.pos[2 * at + 1], A * pu.x, A * pu.y)} />{/if}
						{#if mode === 'turn'}<path class="goal" d={arrow(p.pos[2 * at], p.pos[2 * at + 1], 0, 1.1)} />{/if}
					</svg>
					<p class="small">{L('Energy', '能量')}: <b>{fmt3(p.E)}</b>{#if p.trueE !== undefined}; {L('formula', '公式')}: <b>{fmt3(p.trueE)}</b>{/if}</p>
				</figure>
			{/each}
		</div>
		<p class="small keys">
			{#each { length: K } as _, k (k)}<span><i style:background="var(--cat-{k})"></i>{t(KIND_LABEL(k))}</span>{/each}
			{#if mode === 'forces'}<span class="net-t">{L('solid: network', '实线：网络')}</span><span class="true-t">{L('dashed: formula', '虚线：公式')}</span>{/if}
		</p>
		{#if mode === 'atoms'}
			<p class="small">{L(`Every pair is joined: ${m.n * (m.n - 1)} messages, one each way. First rows (counting from 0):`, `每一对都相连：共 ${m.n * (m.n - 1)} 条消息，每对两个方向各一条。前几行（从 0 数起）：`)} <code>{rows}</code></p>
		{:else if pu}
			<p class="small">{L(`Atom ${at}, layer 1: ${pu.terms.length} thin arrows, one per other atom; the thick arrow is their sum (${fmt3(pu.x)}, ${fmt3(pu.y)}).`, `原子 ${at}，第 1 层：${pu.terms.length} 个细箭头，每个其他原子一个；粗箭头是它们的和 (${fmt3(pu.x)}, ${fmt3(pu.y)})。`)}</p>
		{:else if mode === 'turn'}
			<label class="row">
				<span>{L('Turn', '转角')}: <b>{deg}°</b></span>
				<input type="range" min="0" max="355" step="5" value={deg} oninput={(e) => set({ ...raw, turn: Number((e.currentTarget as HTMLInputElement).value) })} aria-label={L('Turn the molecule (degrees)', '转动分子（度）')} />
			</label>
			<p class="small" aria-live="polite">{L(`Force on atom ${at}: ${fmt3(ang)}° (straight up is 90°, ${fmt3(Math.abs(wrap(ang - 90)))}° away). Energy ${fmt3(m.E)}.`, `原子 ${at} 受的力：${fmt3(ang)}°（正上方是 90°，还差 ${fmt3(Math.abs(wrap(ang - 90)))}°）。能量 ${fmt3(m.E)}。`)}</p>
		{:else if mode === 'spin' && spinRows.length}
			<div class="tbl" role="table" aria-label={L('Every turn so far', '目前为止的每一次转动')}>
				<div role="row"><span role="columnheader">{L('turn', '转角')}</span><span role="columnheader">{L('energy', '能量')}</span><span role="columnheader">{L(`force on atom ${at}`, `原子 ${at} 受的力`)}</span></div>
				{#each spinRows as r, k (k)}
					<div role="row" class:cur={k === spinRows.length - 1}><span role="cell">{r.deg}°</span><span role="cell">{fmt3(r.E)}</span><span role="cell">{fmt3(r.ang)}°</span></div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<style>
	.mol { display: grid; gap: var(--s2); min-width: 0; }
	.panes { display: grid; gap: var(--s2); grid-template-columns: minmax(0, 340px); }
	.panes.two { grid-template-columns: repeat(2, minmax(0, 300px)); }
	figure { margin: 0; min-width: 0; }
	figcaption { font-size: var(--fs-sm); color: var(--ink-2); }
	svg { display: block; width: 100%; height: auto; border: 1px solid var(--rule); border-radius: var(--r1); background: var(--surface); }
	.pair { stroke: var(--rule); stroke-width: 0.02; }
	.atom { stroke: var(--ink-2); stroke-width: 0.03; }
	.atom.at { stroke: var(--highlight); stroke-width: 0.08; }
	text { font-size: 0.32px; fill: var(--ink); }
	path { fill: none; stroke-linecap: round; stroke-linejoin: round; }
	.net { stroke: var(--generated); stroke-width: 0.07; }
	.net.thin { stroke-width: 0.04; }
	.term { stroke: var(--generated); stroke-width: 0.03; stroke-dasharray: 0.08 0.05; }
	.true { stroke: var(--real); stroke-width: 0.05; stroke-dasharray: 0.12 0.08; }
	.goal { stroke: var(--muted); stroke-width: 0.03; stroke-dasharray: 0.06 0.06; }
	.keys { display: flex; flex-wrap: wrap; gap: var(--s2); align-items: center; }
	.keys i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
	.net-t { color: var(--generated-text); }
	.true-t { color: var(--real-text); }
	.row { display: flex; flex-wrap: wrap; gap: var(--s2); align-items: center; }
	.row input { flex: 1 1 160px; min-width: 0; }
	.tbl { display: grid; grid-template-columns: repeat(3, auto); justify-content: start; column-gap: var(--s3); font-size: var(--fs-sm); font-variant-numeric: tabular-nums; }
	.tbl > div { display: grid; grid-column: 1 / -1; grid-template-columns: subgrid; }
	.tbl .cur span { font-weight: 600; }
	p { margin: 0; }
	code { font-size: var(--fs-sm); }
</style>
