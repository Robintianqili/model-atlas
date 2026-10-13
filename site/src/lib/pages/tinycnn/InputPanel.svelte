<!--
  Input section: the drawing pad (strokes -> preprocessStrokes, the dataset framing), preset
  drawings from the Quick, Draw! test split, photo upload (preprocessPhoto), and "what the model
  sees": the exact 28x28 input tensor, updated on every frame while drawing.
-->
<script lang="ts">
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import PresetGallery from './PresetGallery.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Highlight } from '$lib/kit/types';
	import type { CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();

	let padW = $state(280);
	let photoMsg = $state<string | null>(null);
	let dragOver = $state(false);
	let fileInput: HTMLInputElement | undefined = $state();

	const src = $derived(S.source);
	const presetGroup = $derived(
		src.kind === 'preset' && S.presets
			? (S.presets.heldOut.find((g) => g.id === src.cls) ?? S.presets.classes.find((g) => g.id === src.cls))
			: null
	);
	const items = $derived(
		S.probs ? Array.from(S.probs, (p, i) => ({ label: t(S.spec.classes[i]), p, id: i, color: categorical(i, themeState.resolved) })) : []
	);
	/** pad overlay for the preprocessing crop (stroke path): the square that maps onto the central 26x26 */
	const crop = $derived(src.kind === 'pad' && S.pre.crop && !S.pre.empty ? S.pre.crop : null);

	const rfHighlights = $derived.by<Highlight[]>(() => {
		const hl: Highlight[] = [];
		const b = S.hoverRF;
		if (b) hl.push({ row: b.y0, col: b.x0, h: b.y1 - b.y0 + 1, w: b.x1 - b.x0 + 1, color: 'var(--accent)' });
		return hl;
	});
	const eraseMask = $derived.by(() => {
		const m = S.erase;
		if (m === 'none') return undefined;
		return (r: number, c: number) => (m === 'left' ? c < 14 : m === 'right' ? c >= 14 : m === 'top' ? r < 14 : r >= 14);
	});

	async function onFile(f: File | undefined | null) {
		if (!f) return;
		photoMsg = null;
		const err = await S.loadPhoto(f);
		photoMsg =
			err === null
				? null
				: err === 'empty'
					? t({ en: 'No drawing found in that photo. Try a darker pen on plain paper.', zh: '照片里没找到画。请用深色笔在白纸上画。' })
					: t({ en: 'That file could not be read as an image.', zh: '无法把这个文件读成图片。' });
	}
	function ondrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		onFile(e.dataTransfer?.files?.[0]);
	}
	function backToPad() {
		S.setStrokes([]);
	}
	const eraseLabel = $derived(
		S.erase === 'left'
			? t({ en: 'Left half erased', zh: '已擦去左半边' })
			: S.erase === 'right'
				? t({ en: 'Right half erased', zh: '已擦去右半边' })
				: S.erase === 'top'
					? t({ en: 'Top half erased', zh: '已擦去上半部' })
					: t({ en: 'Bottom half erased', zh: '已擦去下半部' })
	);
</script>

<div class="inpw">
<div class="inp">
	<div class="padcol">
		<div
			class="padwrap"
			class:drag={dragOver}
			bind:clientWidth={padW}
			role="group"
			aria-label={t({ en: 'Drawing pad', zh: '画板' })}
			ondragover={(e) => {
				e.preventDefault();
				dragOver = true;
			}}
			ondragleave={() => (dragOver = false)}
			{ondrop}
		>
			<DrawPad
				bind:strokes={S.strokes}
				brush={0.055}
				showBrush={false}
				onchange={(s) => S.setStrokes(s)}
				ondraw={(s) => S.drawing(s)}
				label={t({ en: 'Draw one of the 10 classes', zh: '画出 10 个类别之一' })}
			/>
			{#if crop}
				<span class="sq" aria-hidden="true" style="width: {padW}px; height: {padW}px">
					<span class="crop" style="left: {crop.x * 100}%; top: {crop.y * 100}%; width: {crop.size * 100}%; height: {crop.size * 100}%"></span>
				</span>
			{/if}
			{#if src.kind === 'preset' && presetGroup}
				<div class="overlay" style="width: {padW}px; height: {padW}px">
					<MapCanvas data={S.pre.input} rows={28} cols={28} domain={[0, 1]} paper size={padW} />
					<span class="ov-note">
						{t({ en: 'Preset:', zh: '预设：' })} {t(presetGroup)} #{src.idx + 1}
						· {t({ en: 'draw on the pad to start your own', zh: '在画板上画即可换成你自己的画' })}
					</span>
				</div>
			{:else if src.kind === 'photo'}
				<div class="overlay photo" style="width: {padW}px; height: {padW}px">
					<img src={src.url} alt={t({ en: 'Your photo', zh: '你的照片' })} />
					<span class="ov-note">{t({ en: 'Photo:', zh: '照片：' })} {src.name} · {t({ en: 'draw on the pad to replace it', zh: '在画板上画即可替换' })}</span>
				</div>
			{/if}
		</div>
		<div class="srcbar">
			<button type="button" class="btn" onclick={() => fileInput?.click()}>
				<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M2 5h2.5l1.2-2h4.6l1.2 2H14v8H2zM8 6.5a2.8 2.8 0 1 0 0 5.6 2.8 2.8 0 0 0 0-5.6z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" /></svg>
				{t({ en: 'Photo of a drawing', zh: '上传画的照片' })}
			</button>
			<input bind:this={fileInput} class="visually-hidden" type="file" accept="image/*" tabindex="-1" aria-label={t({ en: 'Photo of a drawing', zh: '上传画的照片' })} onchange={(e) => onFile((e.currentTarget as HTMLInputElement).files?.[0])} />
			{#if src.kind !== 'pad'}
				<button type="button" class="btn ghost" onclick={backToPad}>{t({ en: 'Back to an empty pad', zh: '回到空白画板' })}</button>
			{/if}
		</div>
		{#if photoMsg}<p class="warn" role="status">{photoMsg}</p>{/if}
		<p class="priv muted">{t({ en: 'Your drawing and photos stay on your device; the model runs in this page.', zh: '你的画和照片只留在你的设备上；模型就在这个页面里运行。' })}</p>
	</div>

	<div class="seecol">
		<div class="see-head">
			<h3>{t({ en: 'What the model sees', zh: '模型看到的' })}</h3>
			<ProvenanceBadge mode="live" compact />
		</div>
		<TensorGrid
			data={S.input}
			rows={28}
			cols={28}
			kind="sequential"
			domain={[0, 1]}
			maxCell={9}
			label={t({ en: 'Model input, 28×28', zh: '模型输入，28×28' })}
			highlights={rfHighlights}
			masked={eraseMask}
		/>
		{#if S.erase !== 'none'}
			<p class="chiprow">
				<span class="chip">{eraseLabel}</span>
				<button type="button" class="btn ghost small" onclick={() => (S.erase = 'none')}>{t({ en: 'Restore', zh: '恢复' })}</button>
			</p>
		{/if}
		<p class="cap">
			<Lang>
				{#snippet en()}28×28 numbers between 0 (no ink) and 1 (full ink): the only thing the model ever gets. Your strokes are cropped to their bounding box (the dashed square on the pad), scaled so the long side spans 26 pixels, centered, and redrawn at the stroke width of the training drawings.{/snippet}
				{#snippet zh()}28×28 个介于 0（无墨）和 1（满墨）之间的数：这是模型唯一能得到的东西。你的笔画先按外接框裁剪（画板上的虚线方框），再缩放到长边 26 像素、居中，并按训练图的笔画粗细重新绘制。{/snippet}
			</Lang>
		</p>
		{#if src.kind === 'photo'}
			<div class="steps" aria-label={t({ en: 'Photo preprocessing steps', zh: '照片预处理步骤' })}>
				<figure>
					<img src={src.url} alt="" style="max-width: 96px; max-height: 96px" />
					<figcaption>1 · {t({ en: 'photo', zh: '照片' })}</figcaption>
				</figure>
				<figure>
					<MapCanvas data={src.ink} rows={src.inkH} cols={src.inkW} kind="sequential" domain={[0, 1]} size={Math.round(96 * Math.min(1, src.inkW / src.inkH))} />
					<figcaption>2 · {src.inverted ? t({ en: 'inverted, background removed', zh: '反色并去除背景' }) : t({ en: 'background removed', zh: '去除背景' })}</figcaption>
				</figure>
				<figure>
					<MapCanvas data={S.pre.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={56} />
					<figcaption>3 · {t({ en: 'framed, 28×28', zh: '取景后 28×28' })}</figcaption>
				</figure>
			</div>
		{/if}
	</div>

	<div class="quick">
		<div class="see-head">
			<h3>{t({ en: 'Prediction', zh: '预测' })}</h3>
			<ProvenanceBadge mode="live" compact precision={S.spec.precision} />
		</div>
		{#if S.loadError}
			<p class="warn" role="alert">{t({ en: 'The model could not be loaded:', zh: '模型加载失败：' })} {S.loadError}</p>
		{:else if !S.trace}
			<p class="muted">{t({ en: 'Loading the model (about 250 KB)…', zh: '正在加载模型（约 250 KB）…' })}</p>
		{:else}
			<ProbBars items={items} k={3} chosen={S.top} />
			{#if S.pre.empty}
				<p class="note">{t({ en: 'The pad is empty, yet the model still answers: every class gets a score from the biases alone. A classifier cannot say "nothing".', zh: '画板是空的，模型仍然给出答案：只靠偏置也能让每个类别得到分数。分类器不会说“什么都不是”。' })}</p>
			{:else if src.kind === 'preset' && src.heldOut}
				<p class="note">
					<Lang>
						{#snippet en()}A <strong>{t(presetGroup ?? '')}</strong> is not one of the 10 classes the model learned. <Term id="softmax" /> still has to share 100% among those 10.{/snippet}
						{#snippet zh()}<strong>{t(presetGroup ?? '')}</strong>不属于模型学过的 10 个类别。<Term id="softmax" />仍然必须把 100% 分给这 10 个类别。{/snippet}
					</Lang>
				</p>
			{/if}
			<p class="timing muted num">
				{t({ en: 'Forward pass in your browser:', zh: '浏览器中的前向传播：' })} {S.run ? S.run.ms.toFixed(1) : '–'} ms · {fmtPct(S.probs?.[S.top] ?? 0)} {t(S.spec.classes[S.top])}
			</p>
		{/if}
	</div>
</div>

<PresetGallery {S} />
</div>

<style>
	/* the input sheet (a 720 px dialog, a bottom sheet on phones) or any other box: columns by its own width */
	.inpw {
		display: grid;
		gap: 20px;
		container-type: inline-size;
	}
	.inp {
		display: grid;
		grid-template-columns: minmax(0, 300px) minmax(0, 1fr) minmax(0, 1fr);
		gap: 20px 28px;
		align-items: start;
	}
	@container (max-width: 900px) {
		.inp {
			grid-template-columns: minmax(0, 280px) minmax(0, 1fr);
		}
		.quick {
			grid-column: 1 / -1;
		}
	}
	@container (max-width: 520px) {
		.inp {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.padcol {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.padwrap {
		position: relative;
		width: min(280px, 100%);
		border-radius: var(--r2);
	}
	.padwrap.drag {
		outline: 3px dashed var(--accent);
		outline-offset: 4px;
	}
	.sq {
		position: absolute;
		top: 0;
		left: 0;
		pointer-events: none;
	}
	.crop {
		position: absolute;
		border: 1.5px dashed var(--accent);
		border-radius: 2px;
		pointer-events: none;
	}
	.overlay {
		position: absolute;
		top: 0;
		left: 0;
		display: grid;
		place-items: center;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		overflow: hidden;
		background: var(--surface);
		pointer-events: none;
	}
	.overlay :global(.mc) {
		position: absolute;
		inset: 0;
	}
	.overlay.photo img {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
	}
	.ov-note {
		position: absolute;
		left: 6px;
		right: 6px;
		bottom: 6px;
		padding: 3px 8px;
		border-radius: var(--r1);
		background: color-mix(in srgb, var(--surface) 88%, transparent);
		font-size: var(--fs-xs);
		color: var(--ink-2);
		line-height: 1.35;
	}
	.srcbar {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.priv {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.warn {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--illus);
	}
	.seecol,
	.quick {
		display: grid;
		gap: 8px;
		min-width: 0;
		align-content: start;
	}
	.see-head {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.cap,
	.note {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.note {
		padding: 8px 10px;
		border-left: 3px solid var(--illus);
		background: var(--illus-soft);
		border-radius: 0 var(--r1) var(--r1) 0;
	}
	.chiprow {
		margin: 0;
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.small {
		min-height: 30px;
		padding: 0 10px;
	}
	.timing {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.steps {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
		align-items: flex-end;
	}
	.steps figure {
		margin: 0;
		display: grid;
		gap: 4px;
		justify-items: start;
	}
	.steps figcaption {
		font-size: var(--t-xs);
		color: var(--muted);
		max-width: 120px;
	}
</style>
