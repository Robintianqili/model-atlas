<!--
  kalman-filter Capture (local adapter 'track'): pick a true track or draw one (press and drag across the plot: the
  positions under the stroke follow it; arrow keys pick a second and move it by 1 m), and draw new sensor noise. The
  orange dots are the readings the filter gets: the true track plus seeded noise. The filter's q and r are kept.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { dataOf, HI, LO, N, PRESETS, rawOf, seedOf, truthOf, type TrackRaw } from '../../track.ts';

	interface Props { raw: unknown; onchange: (raw: TrackRaw) => void; compact?: boolean }
	let { raw, onchange, compact = false }: Props = $props();
	const r = $derived(rawOf(raw));
	const keep = $derived({ seed: seedOf(raw), ...(r.q !== undefined ? { q: r.q } : {}), ...(r.r !== undefined ? { r: r.r } : {}) });
	let draft = $state.raw<number[] | null>(null);
	const truth = $derived(draft ?? truthOf(raw));
	const z = $derived(draft ? [] : dataOf(raw).z);
	let W = $state(480);
	const H = $derived(compact ? 120 : 170);
	const X = (s: number) => 6 + (s / (N - 1)) * (W - 12);
	const Y = (v: number) => 6 + ((HI - v) / (HI - LO)) * (H - 12);
	const path = $derived(truth.map((v, s) => `${s ? 'L' : 'M'}${X(s).toFixed(1)},${Y(v).toFixed(1)}`).join(''));
	let sel = $state(0);
	let last = -1;

	function at(e: PointerEvent): [number, number] {
		const b = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const s = Math.round((((e.clientX - b.left) / b.width) * W - 6) / ((W - 12) / (N - 1)));
		const v = HI - ((((e.clientY - b.top) / b.height) * H - 6) / (H - 12)) * (HI - LO);
		return [Math.max(0, Math.min(N - 1, s)), Math.round(Math.max(LO, Math.min(HI, v)) * 10) / 10];
	}
	function paint(e: PointerEvent) {
		const [s, v] = at(e), y = (draft ?? truthOf(raw)).slice();
		if (y.length < N) return;
		const a = last < 0 ? s : last;
		for (let j = Math.min(a, s); j <= Math.max(a, s); j++) y[j] = a === s ? v : y[a] + ((v - y[a]) * (j - a)) / (s - a);
		(last = s), (sel = s), (draft = y);
	}
	function down(e: PointerEvent) {
		if (!e.isPrimary || !truthOf(raw).length) return;
		(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
		last = -1;
		paint(e);
	}
	function up() {
		if (draft) onchange({ y: draft, ...keep });
		(draft = null), (last = -1);
	}
	function key(e: KeyboardEvent) {
		const y = truthOf(raw);
		if (!y.length) return;
		if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') sel = Math.max(0, Math.min(N - 1, sel + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 5 : 1)));
		else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
			const next = y.slice();
			next[sel] = Math.max(LO, Math.min(HI, next[sel] + (e.key === 'ArrowUp' ? 1 : -1)));
			onchange({ y: next, ...keep });
		} else return;
		e.preventDefault();
	}
	/** the lowest position with a typographic minus (−5) */
	const lo = String(LO).replace('-', '−');
	const name = $derived(r.y ? null : (r.preset ?? 'stopgo'));
</script>

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'True tracks', zh: '真实轨迹' })}>
		{#each PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={name === p.id} onclick={() => onchange({ preset: p.id, ...keep })}>{t(p.label)}</button>{/each}
		<button type="button" class="btn" onclick={() => onchange({ ...r, seed: seedOf(raw) + 1 })}>{t({ en: 'New sensor noise', zh: '换一组传感器噪声' })}</button>
	</div>
	<div bind:clientWidth={W}>
		<svg width={W} height={H} viewBox="0 0 {W} {H}" tabindex="0" role="slider" aria-valuemin={0} aria-valuemax={N - 1} aria-valuenow={sel}
			aria-label={t({ en: 'The true track: press and drag to redraw it; arrow keys move one second', zh: '真实轨迹：按住拖动来重画；方向键调整某一秒' })}
			aria-valuetext={truth.length ? t({ en: `second ${sel}: ${truth[sel].toFixed(1)} m`, zh: `第 ${sel} 秒：${truth[sel].toFixed(1)} 米` }) : ''}
			onpointerdown={down} onpointermove={(e) => draft && paint(e)} onpointerup={up} onpointercancel={up} onkeydown={key}>
			<rect x="0" y="0" width={W} height={H} rx="4" class="bg" />
			{#each z as v, s (s)}<circle cx={X(s)} cy={Y(v)} r="2" class="z" />{/each}
			<path d={path} class="line" />
			{#if truth.length}<circle cx={X(sel)} cy={Y(truth[sel])} r="3.5" class="sel" />{/if}
		</svg>
	</div>
	<p class="note">
		{#if !compact}{t({ en: `Dashed: the true track of the cart (${N} seconds, ${lo} to ${HI} m), which you can draw. Dots: what the sensor reads, the truth plus random noise.`, zh: `虚线：小车的真实轨迹（${N} 秒，${lo} 到 ${HI} 米），可以自己画。圆点：传感器的读数，也就是真实位置加上随机噪声。` })}{:else}{t({ en: 'Pick a track or draw one; the dots are the readings.', zh: '选一条轨迹或自己画；圆点是读数。' })}{/if}
	</p>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	svg { display: block; max-width: 100%; touch-action: none; cursor: crosshair; }
	svg:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.line { fill: none; stroke: var(--ink-2); stroke-width: 2; stroke-dasharray: 6 4; }
	.z { fill: var(--cat-1); opacity: 0.8; }
	.sel { fill: var(--accent); }
	p { margin: 0; }
	.note { font-size: var(--fs-xs); color: var(--muted); }
</style>
