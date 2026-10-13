<!--
  The atlas filter bar (redesign spec §4.5, §6.4): search (44 px) · Kind segment (All 67 / Models 47 / Training methods 20)
  · input chips with counts. Sticky under the header on desktop; on phones only the search stays (the page adds a sticky
  jump bar). The input chips hide when only training methods are shown (they have no input of their own).
    <FilterBar bind:q bind:kind bind:input counts={index.counts} inputs={index.inputs} />
-->
<script lang="ts">
	import Icon from '../Icon.svelte';
	import { search } from '../icons';
	import { draw, type, points, game, graph } from './glyphs';
	import { t } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { I18n } from '$lib/i18n/types';
	import type { CatalogCounts, InputBucket } from './types';

	interface Props {
		q: string;
		kind: 'all' | 'model' | 'concept';
		input: InputBucket | null;
		counts: CatalogCounts;
		inputs: { bucket: InputBucket; n: number }[];
	}
	let { q = $bindable(''), kind = $bindable('all'), input = $bindable(null), counts, inputs }: Props = $props();

	const BUCKET: Record<InputBucket, { d: string; label: I18n }> = {
		draw: { d: draw, label: { en: 'Draw', zh: '画' } },
		type: { d: type, label: { en: 'Type', zh: '打字' } },
		points: { d: points, label: { en: 'Points', zh: '点与序列' } },
		play: { d: game, label: { en: 'Play', zh: '游戏' } },
		graphs: { d: graph, label: { en: 'Graphs', zh: '图与评分' } }
	};
	const kinds = $derived([
		{ k: 'all' as const, label: ct('kindAll'), n: counts.lessons },
		{ k: 'model' as const, label: ct('kindModels'), n: counts.models },
		{ k: 'concept' as const, label: ct('kindMethods'), n: counts.methods }
	]);
	const id = $props.id();
</script>

<div class="bar" role="search">
	<div class="find">
		<label for="{id}-q" class="visually-hidden">{t({ en: 'Search lessons', zh: '搜索课程' })}</label>
		<span class="glass" aria-hidden="true"><Icon d={search} size={18} /></span>
		<input id="{id}-q" type="search" bind:value={q} placeholder={t({ en: 'Search by name, idea or term', zh: '按名称、概念或术语搜索' })} autocomplete="off" enterkeyhint="search" />
	</div>
	<div class="seg kind" role="group" aria-label={ct('kind')}>
		{#each kinds as k (k.k)}
			<button type="button" aria-pressed={kind === k.k} onclick={() => (kind = k.k)}>{k.label} <span class="num">{k.n}</span></button>
		{/each}
	</div>
	{#if kind !== 'concept'}
		<div class="inputs" role="group" aria-label={t({ en: 'Your input', zh: '你的输入' })}>
			{#each inputs as b (b.bucket)}
				{#if b.n}
					<button type="button" class="chip fill" aria-pressed={input === b.bucket} onclick={() => (input = input === b.bucket ? null : b.bucket)}>
						<Icon d={BUCKET[b.bucket].d} size={15} />{t(BUCKET[b.bucket].label)} <span class="c num">{b.n}</span>
					</button>
				{/if}
			{/each}
		</div>
	{/if}
</div>

<style>
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px 16px;
	}
	.find {
		position: relative;
		flex: 1 1 260px;
		max-width: 380px;
	}
	.glass {
		position: absolute;
		left: 14px;
		top: 50%;
		transform: translateY(-50%);
		color: var(--muted);
		pointer-events: none;
	}
	input {
		width: 100%;
		height: 44px;
		padding: 0 14px 0 42px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-md);
	}
	input::placeholder {
		color: var(--muted);
		opacity: 1;
	}
	.kind .num {
		font-size: var(--t-xs);
		font-weight: var(--w-body);
	}
	.inputs {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		margin-left: auto;
	}
	.inputs .chip {
		min-height: 36px;
	}
	@media (max-width: 759px) {
		.find {
			flex-basis: 100%;
			max-width: none;
		}
		.kind,
		.inputs {
			display: none;
		}
	}
</style>
