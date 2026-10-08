<!--
  Hook: the 10-class TinyCNN must pick one of its 10 classes for a ladder (live), and what it
  picked for all 1,500 test ladders / envelopes (precomputed, finetune.json baseModelGuesses).
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { Stroke } from '$lib/kit/draw';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import Thumb from './Thumb.svelte';
	import { BASE_CLASSES, decodeB64, simplifyStrokes, type FinetuneJson, type HeldoutJson } from './data';
	import type { Engine } from './engine';

	interface Props {
		engine: Engine | null;
		heldout: HeldoutJson | null;
		finetune: FinetuneJson | null;
		/** the shared parity numbers for the live badge */
		tolerance?: number;
	}
	let { engine, heldout, finetune, tolerance }: Props = $props();

	type Which = 'ladder' | 'envelope' | 'draw';
	let which = $state<Which>('ladder');
	let idx = $state<Record<'ladder' | 'envelope', number>>({ ladder: -1, envelope: -1 });
	let strokes = $state<Stroke[]>([]);
	let drawn = $state.raw<Float32Array | null>(null);

	const classes = $derived(engine?.spec.classes ?? BASE_CLASSES);
	const name = (c: { en: string; zh: string }) => (i18n.lang === 'zh' ? c.zh : c.en);
	const heldIdx = (w: 'ladder' | 'envelope') => (w === 'ladder' ? 0 : 1);
	const zh = $derived(i18n.lang === 'zh');

	// Start each held-out class on its first test drawing whose top guess is the class's most common
	// guess over the full test split (so the default is typical, not cherry-picked for drama).
	$effect(() => {
		if (!engine || !heldout || !finetune) return;
		for (const w of ['ladder', 'envelope'] as const) {
			if (idx[w] >= 0) continue;
			const shares = finetune.baseModelGuesses.byHeldOutClass[w]?.top1Share ?? {};
			const common = Object.entries(shares).sort((a, b) => b[1] - a[1])[0]?.[0];
			const ci = classes.findIndex((c) => c.id === common);
			const tests = heldout.classes[heldIdx(w)].test;
			let pick = 0;
			for (let i = 0; i < tests.length; i++) {
				if (engine.model.predict(decodeB64(tests[i])).top === ci) {
					pick = i;
					break;
				}
			}
			idx[w] = pick;
		}
	});

	const input = $derived.by(() => {
		if (which === 'draw') return drawn;
		if (!heldout || idx[which] < 0) return null;
		return decodeB64(heldout.classes[heldIdx(which)].test[idx[which]]);
	});
	const pred = $derived(engine && input ? engine.model.predict(input) : null);
	const items = $derived(
		pred
			? classes.map((c, i) => ({ label: name(c), p: pred.probs[i], id: i, color: categorical(i, themeState.resolved) }))
			: []
	);
	const total = $derived(heldout ? heldout.classes[0].test.length : 100);

	let timer: ReturnType<typeof setTimeout> | undefined;
	function runDraw(s: Stroke[], now = false) {
		if (!engine) return;
		clearTimeout(timer);
		const go = () => {
			const r = engine!.lib.preprocessStrokes(simplifyStrokes(s));
			drawn = r.empty ? null : r.input;
		};
		if (now) go();
		else timer = setTimeout(go, 60);
	}
	function step(d: number) {
		if (which === 'draw') return;
		idx[which] = (idx[which] + d + total) % total;
	}

	const shareRows = (w: 'ladder' | 'envelope') => {
		const s = finetune?.baseModelGuesses.byHeldOutClass[w]?.top1Share ?? {};
		const rows = Object.entries(s)
			.sort((a, b) => b[1] - a[1])
			.map(([id, p]) => {
				const i = classes.findIndex((c) => c.id === id);
				return { label: i >= 0 ? name(classes[i]) : id, p, id, color: i >= 0 ? categorical(i, themeState.resolved) : undefined };
			});
		const rest = 1 - rows.reduce((a, r) => a + r.p, 0);
		if (rest > 0.0005) rows.push({ label: t({ en: 'other classes', zh: '其他类别' }), p: rest, id: 'other', color: undefined });
		return rows;
	};
	const heldName = (w: 'ladder' | 'envelope') =>
		t(w === 'ladder' ? { en: 'ladder', zh: '梯子' } : { en: 'envelope', zh: '信封' });
	// English article: "a ladder", "an envelope"
	const aHeld = (w: 'ladder' | 'envelope') => (w === 'ladder' ? 'a ladder' : 'an envelope');
</script>

<div class="hook">
	<div class="seg segmented" role="group" aria-label={t({ en: 'Input for the 10-class model', zh: '给 10 类模型的输入' })}>
		{#each [['ladder', { en: 'A ladder', zh: '一架梯子' }], ['envelope', { en: 'An envelope', zh: '一个信封' }], ['draw', { en: 'Draw anything', zh: '随手画' }]] as const as [w, lab] (w)}
			<button type="button" aria-pressed={which === w} class="btn" onclick={() => (which = w)}>{t(lab)}</button>
		{/each}
	</div>

	<div class="grid">
		<div class="inp">
			{#if which === 'draw'}
				<DrawPad
					bind:strokes
					size={240}
					brush={0.055}
					showBrush={false}
					label={{ en: 'Draw anything for the 10-class model', zh: '随手画点什么，给 10 类模型看' }}
					ondraw={(s) => runDraw(s)}
					onchange={(s) => runDraw(s, true)}
				/>
				<p class="muted small">{t({ en: 'Try something it was never taught: a key, a cup, a cat.', zh: '试试它没学过的东西：钥匙、杯子、猫。' })}</p>
			{:else}
				<div class="preset">
					<Thumb data={input} size={168} alt={t({ en: `Test drawing of ${aHeld(which)}`, zh: `一张${heldName(which)}的测试图` })} />
					<div class="nav">
						<button type="button" class="btn ghost icon" onclick={() => step(-1)} aria-label={t({ en: 'Previous drawing', zh: '上一张' })}>‹</button>
						<span class="num muted small" aria-live="polite">{t({ en: `${heldName(which)} ${idx[which] + 1} of ${total}`, zh: `${heldName(which)} 第 ${idx[which] + 1} / ${total} 张` })}</span>
						<button type="button" class="btn ghost icon" onclick={() => step(1)} aria-label={t({ en: 'Next drawing', zh: '下一张' })}>›</button>
					</div>
					<p class="muted small">{t({ en: 'A Quick, Draw! test drawing, framed to 28 × 28 exactly as the model sees it.', zh: '一张 Quick, Draw! 测试图，按模型实际看到的样子缩放到 28 × 28。' })}</p>
				</div>
			{/if}
		</div>
		<div class="out">
			<div class="head-row">
				<h3>{t({ en: 'What the 10-class model says', zh: '10 类模型的回答' })}</h3>
				<ProvenanceBadge mode="live" precision="fp32" {tolerance} against="PyTorch fp32" compact />
			</div>
			{#if pred}
				<ProbBars {items} k={10} chosen={pred.top} label={{ en: 'Probability per known class', zh: '每个已知类别的概率' }} />
				<p class="verdict">
						{#if which === 'draw'}
							{t({ en: 'It answers ', zh: '它的回答是' })}<strong>{name(classes[pred.top])}</strong>{zh ? '（' : ' ('}<Num value={pred.probs[pred.top] * 100} unit="%" />{zh ? '）。' : '). '}
						{:else}
							{t({ en: `It has never seen ${aHeld(which)}, yet it answers `, zh: `它从没见过${heldName(which)}，却回答` })}<strong>{name(classes[pred.top])}</strong>{zh ? '（' : ' ('}<Num value={pred.probs[pred.top] * 100} unit="%" />{zh ? '）。' : '). '}
						{/if}{t({ en: 'The 10 probabilities always add up to 100%: there is no “none of these”.', zh: '这 10 个概率加起来总是 100%：没有“都不是”这个选项。' })}
				</p>
			{:else if which === 'draw'}
				<p class="muted">{t({ en: 'Draw on the pad to see its answer.', zh: '在画板上画一笔，看看它的回答。' })}</p>
			{:else}
				<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
			{/if}
		</div>
	</div>

	<div class="shares">
		<div class="head-row">
			<h3>{t({ en: 'Its top guess over all 1,500 test drawings of each new class', zh: '它对每个新类别全部 1,500 张测试图的首选答案' })}</h3>
			<ProvenanceBadge mode="precomputed" compact />
		</div>
		<div class="share-grid">
			{#each ['ladder', 'envelope'] as const as w (w)}
				<div class="share" class:on={which === w}>
					<p class="sh-title">{t(w === 'ladder' ? { en: 'Ladders are called…', zh: '梯子被认成……' } : { en: 'Envelopes are called…', zh: '信封被认成……' })}</p>
					{#if finetune}
						<ProbBars items={shareRows(w)} k={6} sort={false} label={{ en: 'share of test drawings', zh: '占测试图的比例' }} />
					{/if}
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.hook {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.seg {
		justify-self: start;
		flex-wrap: wrap;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (min-width: 760px) {
		.grid {
			grid-template-columns: minmax(0, 280px) minmax(0, 1fr);
		}
	}
	.preset {
		display: grid;
		justify-items: start;
		gap: 8px;
	}
	.preset :global(canvas) {
		border: 1px solid var(--rule);
	}
	.nav {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.small {
		font-size: var(--fs-sm);
		margin: 0;
	}
	.head-row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 6px 12px;
		margin-bottom: 6px;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.verdict {
		margin: 10px 0 0;
		max-width: var(--measure);
	}
	.shares {
		padding: var(--s4);
		border-radius: var(--r3);
		background: var(--surface-2);
	}
	.share-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
		gap: var(--s4);
	}
	.share {
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface);
		border: 1px solid var(--rule);
	}
	.share.on {
		border-color: var(--accent);
	}
	.sh-title {
		margin: 0 0 4px;
		font-weight: 650;
	}
</style>
