<!-- vae-2d/latent (see meta.ts). The map is an SVG in latent units (y up); the "what each place decodes to" layer is a
     canvas filled row by row (9 decodes per task, a few ms each). Dragging the ring decodes on every move and commits the
     ring's z to the learner's input on release (or on each arrow key), so the lesson's task can read it. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Pic from '../../parts/Pic.svelte';
	import { K, RANGE, atBoundary, className, decode, near, withExtra, zhAfter, type XY } from '../../vae.ts';
	import { latentPic } from './meta.ts';

	let P: StageProps = $props();
	const m = $derived(latentPic(P));
	const model = $derived(P.rt.model);
	const S = 360;
	let live = $state<XY | null>(null);
	let dragging = false;
	let showGrid = $state(false);
	const probe = $derived<XY>(live ?? m?.probe ?? [0, 0]);
	// the window in latent units: the whole map, or a close-up around μ for the cloud
	const win = $derived.by(() => {
		if (!m || m.mode !== 'cloud') return { x0: -RANGE, y0: -RANGE, w: 2 * RANGE };
		const dz = Math.max(Math.abs(m.z[0] - m.mu[0]), Math.abs(m.z[1] - m.mu[1]));
		const h = Math.max(4 * Math.max(...m.sigma), m.sample ? 1.4 * dz : 0, 0.1);
		return { x0: m.mu[0] - h, y0: m.mu[1] - h, w: 2 * h };
	});
	const dp = $derived(m?.mode === 'cloud' ? 2 : 0);
	const tk = (v: number) => v.toFixed(dp).replace('-', '−');
	const X = (v: number) => ((v - win.x0) / win.w) * S;
	const Y = (v: number) => S - ((v - win.y0) / win.w) * S;
	const name = (c: number) => t(className(P.rt, c));
	const nearText = (n: ReturnType<typeof near>) => n.order.filter((c) => n.counts[c] > 0).map((c) => t({ en: `${n.counts[c]} ${name(c)}`, zh: `${n.counts[c]} 幅${zhAfter(name(c))}` })).join(t({ en: ', ', zh: '、' }));
	const probeImg = $derived(m && model && (m.mode === 'drag' || m.mode === 'map') ? decode(model, m.mode === 'map' ? m.mu : probe) : null);
	const nearP = $derived(m ? near(m.lat, probe) : null);
	const walk = $derived.by(() => {
		if (!m?.target) return [];
		const a = m.mu, b = m.target.mu;
		return Array.from({ length: 7 }, (_, k) => {
			const z: XY = [a[0] + ((b[0] - a[0]) * k) / 6, a[1] + ((b[1] - a[1]) * k) / 6];
			return { z, img: model ? decode(model, z) : null };
		});
	});

	// StageProps.setInput (the page store; it re-runs the sources and re-resolves the lesson; ViewSlot passes it since K1),
	// else rt.setInput (a view outside a stage: the gallery demo)
	const set = (raw: unknown) => (P.setInput ?? ((r: unknown) => P.rt.setInput(r)))(raw);
	function commit(z: XY) {
		live = null;
		set(withExtra(P.rt.input, { probe: [Math.round(z[0] * 1000) / 1000, Math.round(z[1] * 1000) / 1000] }));
	}
	function toLatent(e: PointerEvent): XY {
		const r = (e.currentTarget as Element).getBoundingClientRect();
		const c = (v: number) => Math.max(-RANGE, Math.min(RANGE, v));
		return [c(win.x0 + ((e.clientX - r.left) / r.width) * win.w), c(win.y0 + (1 - (e.clientY - r.top) / r.height) * win.w)];
	}
	const down = (e: PointerEvent) => {
		if (m?.mode !== 'drag') return;
		dragging = true;
		(e.currentTarget as Element).setPointerCapture(e.pointerId);
		live = toLatent(e);
	};
	const move = (e: PointerEvent) => void (dragging && (live = toLatent(e)));
	const up = () => {
		if (dragging && live) commit(live);
		dragging = false;
	};
	function key(e: KeyboardEvent) {
		if (m?.mode !== 'drag') return;
		const d: Record<string, XY> = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] };
		if (!d[e.key]) return;
		e.preventDefault();
		commit([Math.max(-RANGE, Math.min(RANGE, probe[0] + d[e.key][0])), Math.max(-RANGE, Math.min(RANGE, probe[1] + d[e.key][1]))]);
	}

	// the canvas under the SVG: the test drawings as dots (2,000 SVG circles made a long task), or "what each place
	// decodes to": N × N decodes, one row per task
	const N = 9;
	let cv = $state<HTMLCanvasElement>();
	const grid = $derived(showGrid && m?.mode !== 'cloud');
	$effect(() => {
		const theme = themeState.resolved, mm = m, w = win;
		if (!cv || !mm) return;
		const g = cv.getContext('2d')!, k = cv.width / S;
		g.clearRect(0, 0, cv.width, cv.height);
		if (!grid || !model) {
			const st = getComputedStyle(cv), col = mm.lat.classes.map((_, c) => st.getPropertyValue(`--cat-${c}`).trim() || '#888');
			g.globalAlpha = theme === 'dark' ? 0.7 : 0.55;
			mm.lat.x.forEach((x, i) => {
				const y = mm.lat.y[i];
				if (x < w.x0 || x > w.x0 + w.w || y < w.y0 || y > w.y0 + w.w) return;
				g.fillStyle = col[mm.lat.c[i]];
				g.beginPath();
				g.arc(X(x) * k, Y(y) * k, 2.2 * k, 0, 2 * Math.PI);
				g.fill();
			});
			g.globalAlpha = 1;
			return;
		}
		const tmp = document.createElement('canvas');
		tmp.width = tmp.height = 28;
		const tg = tmp.getContext('2d')!, cell = cv.width / N;
		const ink = theme === 'dark' ? [235, 235, 240] : [20, 20, 30];
		g.imageSmoothingEnabled = false;
		let row = 0, alive = true;
		const step = () => {
			if (!alive || row >= N) return;
			for (let col = 0; col < N; col++) {
				const z: XY = [-RANGE + (2 * RANGE * (col + 0.5)) / N, RANGE - (2 * RANGE * (row + 0.5)) / N];
				const a = decode(model, z), img = tg.createImageData(28, 28);
				for (let i = 0; i < 784; i++) img.data.set([ink[0], ink[1], ink[2], Math.round(255 * a[i])], i * 4);
				tg.putImageData(img, 0, 0);
				g.drawImage(tmp, col * cell + 1, row * cell + 1, cell - 2, cell - 2);
			}
			row++;
			setTimeout(step, 0);
		};
		step();
		return () => void (alive = false);
	});
	$effect(() => {
		if (P.autoplay && m) P.onend();
	});
</script>

{#if !m}
	<Notice text={{ en: 'Draw something first: the map shows where your drawing lands.', zh: '先画点什么：地图显示你的画落在哪里。' }} />
{:else}
	<div class="lat">
		<div class="mapbox">
			<canvas bind:this={cv} width="540" height="540" aria-hidden="true"></canvas>
			<svg viewBox="0 0 {S} {S}" class:drag={m.mode === 'drag'} role={m.mode === 'drag' ? 'group' : 'img'} aria-label={t({ en: 'The latent map: test drawings by kind, and your drawing', zh: '隐空间地图：按类别着色的测试画，以及你的画' })}>
				<rect width={S} height={S} class="bg" />
				{#if win.x0 < 0 && win.x0 + win.w > 0}<line x1={X(0)} x2={X(0)} y1="0" y2={S} class="axis" />{/if}
				{#if win.y0 < 0 && win.y0 + win.w > 0}<line y1={Y(0)} y2={Y(0)} x1="0" x2={S} class="axis" />{/if}
				{#if m.mode === 'cloud'}
					{#each [2, 1] as k (k)}<ellipse cx={X(m.mu[0])} cy={Y(m.mu[1])} rx={(k * m.sigma[0] * S) / win.w} ry={(k * m.sigma[1] * S) / win.w} class="ring r{k}" />{/each}
					{#if m.sample}
						<g class="answer"><line x1={X(m.mu[0])} y1={Y(m.mu[1])} x2={X(m.z[0])} y2={Y(m.z[1])} class="eps" />
						<circle cx={X(m.z[0])} cy={Y(m.z[1])} r="6" class="z" />
						<text x={X(m.z[0]) + 9} y={Y(m.z[1]) + 4} class="lbl">z</text></g>
					{/if}
				{/if}
				{#if m.mode === 'interp' && m.target}
					<line x1={X(m.mu[0])} y1={Y(m.mu[1])} x2={X(m.target.mu[0])} y2={Y(m.target.mu[1])} class="eps" />
					{#each walk as w, k (k)}<circle cx={X(w.z[0])} cy={Y(w.z[1])} r="3.5" class="z" /><text x={X(w.z[0]) + 5} y={Y(w.z[1]) - 5} class="lbl s">{k + 1}</text>{/each}
				{/if}
				<g class="you"><circle cx={X(m.mu[0])} cy={Y(m.mu[1])} r="7" /><line x1={X(m.mu[0]) - 11} x2={X(m.mu[0]) + 11} y1={Y(m.mu[1])} y2={Y(m.mu[1])} /><line y1={Y(m.mu[1]) - 11} y2={Y(m.mu[1]) + 11} x1={X(m.mu[0])} x2={X(m.mu[0])} /></g>
				<text x={X(m.mu[0]) + 10} y={Y(m.mu[1]) - 9} class="lbl">μ</text>
				{#if m.mode === 'drag'}
					<circle cx={X(probe[0])} cy={Y(probe[1])} r="10" class="probe" />
					<rect width={S} height={S} class="hit" role="slider" tabindex="0" aria-label={t({ en: 'The ring: drag it, or move it with the arrow keys', zh: '圆圈：拖动它，或用方向键移动' })} aria-valuenow={probe[0]} aria-valuetext="z = ({probe[0].toFixed(2)}, {probe[1].toFixed(2)})" onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up} onkeydown={key} />
				{/if}
				<text x="4" y={S - 5} class="tick">z₁: {tk(win.x0)} → {tk(win.x0 + win.w)}</text>
				<text x="4" y="13" class="tick">z₂: {tk(win.y0)} → {tk(win.y0 + win.w)}</text>
			</svg>
		</div>
		<div class="side">
			{#if m.mode === 'cloud'}
				<table class="small">
					<tbody>
						<tr><th></th><th>{t({ en: 'number 1', zh: '第 1 个' })}</th><th>{t({ en: 'number 2', zh: '第 2 个' })}</th></tr>
						<tr><th scope="row">μ</th><td><Num value={m.mu[0]} /></td><td><Num value={m.mu[1]} /></td></tr>
						<tr><th scope="row">log σ²</th><td><Num value={m.logvar[0]} /></td><td><Num value={m.logvar[1]} /></td></tr>
						<tr><th scope="row">σ</th><td><Num value={m.sigma[0]} /></td><td><Num value={m.sigma[1]} /></td></tr>
						{#if m.sample}
							<tr><th scope="row">ε</th><td><Num value={m.eps[0]} /></td><td><Num value={m.eps[1]} /></td></tr>
							<tr class="sum answer"><th scope="row">z = μ + σ·ε</th><td><Num value={m.z[0]} /></td><td><Num value={m.z[1]} /></td></tr>
						{/if}
					</tbody>
				</table>
				{#if m.sample}
					<p class="small mono answer"><span class="nw">z₁ = <Num value={m.mu[0]} /></span> <span class="nw">+ <Num value={m.sigma[0]} /> × {#if m.eps[0] < 0}(<Num value={m.eps[0]} />){:else}<Num value={m.eps[0]} />{/if}</span> <span class="nw">= <Num value={m.z[0]} /></span></p>
					<button type="button" class="btn" onclick={() => set(withExtra(P.rt.input, { draw: m.draw + 1 }))}>{t({ en: 'Draw another ε', zh: '再抽一次 ε' })}</button>
					<p class="small muted">{t({ en: `Draw ${m.draw + 1}. The rings are 1σ and 2σ around μ; the line is σ·ε.`, zh: `第 ${m.draw + 1} 次抽取。圈是 μ 周围的 1σ 和 2σ；线段是 σ·ε。` })}</p>
				{:else}
					<svg viewBox="0 0 100 100" class="inset" role="img" aria-label={t({ en: 'Where the close-up is on the whole map', zh: '特写在整张地图上的位置' })}>
						<rect width="100" height="100" class="bg" />
						{#each m.lat.x as x, i (i)}{#if i % 8 === 0}<circle cx={((x + RANGE) / (2 * RANGE)) * 100} cy={100 - ((m.lat.y[i] + RANGE) / (2 * RANGE)) * 100} r="0.9" class="dot c{m.lat.c[i]}" />{/if}{/each}
						<rect x={((win.x0 + RANGE) / (2 * RANGE)) * 100 - 2} y={100 - ((win.y0 + win.w + RANGE) / (2 * RANGE)) * 100 - 2} width={(win.w / (2 * RANGE)) * 100 + 4} height={(win.w / (2 * RANGE)) * 100 + 4} class="win" />
					</svg>
					<p class="small muted">{t({ en: 'A close-up (its place on the whole map is boxed above): the rings are 1σ and 2σ around μ; the dots are test drawings that land this close.', zh: '特写（它在整张地图上的位置见上面的方框）：圈是 μ 周围的 1σ 和 2σ；点是落得这么近的测试画。' })}</p>
				{/if}
			{:else if m.mode === 'interp' && m.target}
				<div class="segmented" role="group" aria-label={t({ en: 'Walk to', zh: '走到' })}>
					{#each m.lat.classes as _, c (c)}<button type="button" aria-pressed={c === m.target.c} onclick={() => set(withExtra(P.rt.input, { target: m.lat.classes[c] }))}>{name(c)}</button>{/each}
				</div>
				<div class="strip">{#each walk as w, k (k)}<figure><Pic data={w.img} size={52} frame="generated" label={t({ en: `point ${k + 1}`, zh: `第 ${k + 1} 个点` })} /><figcaption>{k + 1}</figcaption></figure>{/each}</div>
				<p class="small muted">{t({ en: `From your μ (1) to a typical ${name(m.target.c)} drawing's μ (7), in equal steps.`, zh: `从你的 μ（1）等距走到一幅典型${zhAfter(name(m.target.c))}的 μ（7）。` })}</p>
			{:else}
				{#if probeImg}<figure><Pic data={probeImg} frame="generated" label={t({ en: 'Decoded here', zh: '这里解码出的画' })} /><figcaption>{m.mode === 'drag' ? t({ en: 'decoded at the ring', zh: '圆圈处解码出的画' }) : t({ en: 'decoded at your μ', zh: '在你的 μ 处解码出的画' })}</figcaption></figure>{/if}
				{#if m.mode === 'drag' && nearP}
					<p class="small">z = (<Num value={probe[0]} />, <Num value={probe[1]} />)</p>
					<p class="small">{t({ en: `${K} nearest test drawings: ${nearText(nearP)}`, zh: `最近的 ${K} 幅测试画：${nearText(nearP)}` })}</p>
					<p class="small" class:ok={atBoundary(m.lat, probe)}>{atBoundary(m.lat, probe) ? t({ en: 'On a border: two kinds meet here.', zh: '在边界上：两类在这里相遇。' }) : t({ en: 'Inside one region.', zh: '在一个区域内部。' })}</p>
				{:else}
					<p class="small">{t({ en: `${K} nearest test drawings to your μ: ${nearText(m.nearMu)}`, zh: `离你的 μ 最近的 ${K} 幅测试画：${nearText(m.nearMu)}` })}</p>
				{/if}
				{#if model}<label class="small"><input type="checkbox" bind:checked={showGrid} /> {t({ en: 'Show what each place decodes to', zh: '显示每个位置解码出的画' })}</label>{/if}
			{/if}
			<ul class="legend small" aria-label={t({ en: 'Kinds', zh: '类别' })}>
				{#each m.lat.classes as _, c (c)}<li><span class="sw c{c}"></span>{name(c)}</li>{/each}
				<li><span class="sw you-sw"></span>{t({ en: 'your μ', zh: '你的 μ' })}</li>
			</ul>
		</div>
	</div>
{/if}

<style>
	.lat { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: flex-start; min-width: 0; }
	.mapbox { position: relative; width: min(100%, 400px); aspect-ratio: 1; background: var(--surface); }
	.mapbox svg, .mapbox canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
	svg.drag { touch-action: none; cursor: crosshair; }
	.side { display: grid; gap: 8px; min-width: 0; flex: 1 1 220px; align-content: start; }
	.bg { fill: none; stroke: var(--rule); }
	.axis { stroke: var(--rule-strong); stroke-dasharray: 3 3; }
	.c0 { fill: var(--cat-0); background: var(--cat-0); } .c1 { fill: var(--cat-1); background: var(--cat-1); } .c2 { fill: var(--cat-2); background: var(--cat-2); } .c3 { fill: var(--cat-3); background: var(--cat-3); } .c4 { fill: var(--cat-4); background: var(--cat-4); }
	.you circle { fill: none; stroke: var(--real); stroke-width: 3; } .you line { stroke: var(--real); stroke-width: 1.5; }
	.ring { fill: none; stroke: var(--real); } .ring.r2 { stroke-dasharray: 4 3; }
	.eps { stroke: var(--generated); stroke-width: 2; } .z { fill: var(--generated); }
	.hit { fill: transparent; touch-action: none; cursor: crosshair; } .hit:focus-visible { outline: 2px solid var(--accent); }
	.probe { fill: var(--generated); fill-opacity: 0.25; stroke: var(--generated); stroke-width: 3; cursor: grab; }
	.lbl { font-size: 14px; fill: var(--ink); font-weight: 700; } .lbl.s { font-size: 10px; font-weight: 400; }
	.tick { font-size: 11px; fill: var(--muted); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
	.small { font-size: var(--fs-sm); } p { margin: 0; } .mono { font-family: var(--font-mono); }
	table { border-collapse: collapse; } th, td { padding: 2px 10px 2px 0; text-align: right; } th[scope='row'] { text-align: left; }
	tr.sum th, tr.sum td { font-weight: 700; border-top: 1px solid var(--rule); }
	figure { margin: 0; display: grid; gap: 4px; justify-items: start; } figcaption { font-size: var(--fs-xs); color: var(--muted); }
	.strip { display: flex; flex-wrap: wrap; gap: 6px; } .strip figure { justify-items: center; }
	.legend { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 4px 12px; }
	.sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }
	.you-sw { border: 2px solid var(--real); background: none; }
	.ok { color: var(--real); font-weight: 700; }
	.inset { width: 120px; height: 120px; }
	:global([inert]) .answer { visibility: hidden; } .win { fill: none; stroke: var(--real); stroke-width: 1.5; }
	.btn { justify-self: start; }
	.nw { white-space: nowrap; }
</style>
