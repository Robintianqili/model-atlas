<!--
  A live demo on the reference pages: the kit's own view drawn on demo data (a tiny model with made-up weights, run by the
  real kit ops), badged Illustrative. It loads when it comes near the screen (the /dev/atlas/ gallery module, one lazy
  chunk) and then stays. `op` shows that op's demo layer with the view `key`; `view` shows a view's own demo and variants.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import type { I18n, StageProps, ViewMeta } from '$lib/atlas/types/index.ts';

	interface Props {
		/** an op name (its demo layer), or a view key (its own demos) */
		op?: string;
		view?: string;
		/** the view that draws the demo */
		key: string;
		minHeight?: number | null;
		/** show the view's text summary of the demo (what a screen reader hears) */
		say?: boolean;
	}
	let { op, view, key, minHeight = 240, say = true }: Props = $props();

	type Stage = Omit<StageProps, 'ctx'>;
	type HostC = Component<{ view: Component<StageProps>; stage: Stage; ctx: NonNullable<StageProps['ctx']> }>;
	let el: HTMLDivElement | undefined = $state();
	let status = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let err = $state('');
	let stages = $state.raw<{ label: I18n; stage: Stage }[]>([]);
	let pick = $state(0);
	let View = $state.raw<Component<StageProps> | null>(null);
	let Host = $state.raw<HostC | null>(null);
	let meta = $state.raw<ViewMeta | null>(null);
	let note = $state.raw<I18n | undefined>(undefined);
	const ctx = $derived({ shown: true, reduced: motion.reduced, playing: false, done: () => {} } as unknown as NonNullable<StageProps['ctx']>);
	const cur = $derived(stages[Math.min(pick, Math.max(0, stages.length - 1))]?.stage ?? null);
	const text = $derived.by((): I18n | null => {
		if (!meta || !cur) return null;
		try {
			return meta.describe({ ...cur, ctx } as StageProps);
		} catch {
			return null;
		}
	});

	async function load() {
		if (status !== 'idle') return;
		status = 'loading';
		try {
			const m = await import('./demo-ui.ts');
			const v = m.allViews().find((x) => x.key === key);
			if (!v?.comp) throw new Error(`no view '${key}'`);
			note = m.DEMO_NOTE;
			Host = m.ViewStage as unknown as HostC;
			View = v.comp;
			meta = v.mod.default;
			if (op) {
				const st = m.opStage(op);
				if (!st) throw new Error(`no demo for op '${op}'`);
				stages = [{ label: { en: 'Demo', zh: '演示' }, stage: st }];
			} else {
				const out: { label: I18n; stage: Stage }[] = [{ label: { en: 'Demo', zh: '演示' }, stage: await v.mod.default.demo() }];
				if (view && v.mod.variants) for (const x of await v.mod.variants()) out.push(x);
				stages = out;
			}
			status = 'ready';
		} catch (e) {
			err = e instanceof Error ? e.message : String(e);
			status = 'error';
		}
	}

	onMount(() => {
		if (!el || typeof IntersectionObserver === 'undefined') {
			void load();
			return;
		}
		const io = new IntersectionObserver(
			(es) => {
				if (es.some((e) => e.isIntersecting)) {
					io.disconnect();
					void load();
				}
			},
			{ rootMargin: '400px 0px' }
		);
		io.observe(el);
		return () => io.disconnect();
	});
</script>

<div class="demo" bind:this={el}>
	<div class="bar">
		<span class="lbl">{t({ en: 'Live demo', zh: '实时演示' })} · <code>{key}</code></span>
		<ProvenanceBadge mode="illustrative" compact {note} />
	</div>
	{#if stages.length > 1}
		<div class="segmented" role="group" aria-label={t({ en: 'Demos of this view', zh: '这个视图的演示' })}>
			{#each stages as s, i (i)}
				<button type="button" aria-pressed={i === pick} onclick={() => (pick = i)}>{t(s.label)}</button>
			{/each}
		</div>
	{/if}
	<div class="stage" style="min-height: {minHeight ?? 240}px">
		{#if status === 'ready' && View && Host && cur}
			{@const H = Host}
			{#each [`${key}:${pick}`] as k (k)}
				<H view={View} stage={cur} {ctx} />
			{/each}
		{:else if status === 'error'}
			<p class="bad">{t({ en: 'The demo could not be shown:', zh: '演示无法显示：' })} {err}</p>
		{:else}
			<p class="muted wait">{t({ en: 'The demo loads when you scroll here…', zh: '滚动到这里时加载演示…' })}</p>
		{/if}
	</div>
	{#if say && text}
		<p class="desc"><span class="muted">{t({ en: 'What a screen reader hears:', zh: '读屏软件读出的内容：' })}</span> {t(text)}</p>
	{/if}
</div>

<style>
	.demo {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 6px 10px;
	}
	.lbl {
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.lbl code {
		font-size: 0.85em;
		font-weight: 400;
	}
	.stage {
		min-width: 0;
		padding: var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	@media (max-width: 560px) {
		.stage {
			padding: 10px;
		}
	}
	.wait {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.bad {
		margin: 0;
		color: var(--div-neg-text);
	}
	.desc {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>
