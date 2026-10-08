<!--
  Color-scope switch: which values set the darkest color (this map / whole layer / all layers).
  <ScopeSwitch bind:value={scope} />            (hide options with `options={['unit','layer']}`)
-->
<script lang="ts">
	import type { ColorScope } from './color';
	import { ui } from '$lib/i18n/i18n.svelte';
	import Tooltip from './Tooltip.svelte';

	interface Props {
		value?: ColorScope;
		options?: ColorScope[];
	}
	let { value = $bindable('unit'), options = ['unit', 'layer', 'global'] }: Props = $props();

	const name = (s: ColorScope) => (s === 'unit' ? ui('scopeUnit') : s === 'layer' ? ui('scopeLayer') : ui('scopeGlobal'));
	const gid = $props.id();

	function onkey(e: KeyboardEvent) {
		const i = options.indexOf(value);
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') value = options[(i + 1) % options.length];
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') value = options[(i - 1 + options.length) % options.length];
		else return;
		e.preventDefault();
		const el = (e.currentTarget as HTMLElement).querySelector<HTMLButtonElement>(`[data-v="${value}"]`);
		el?.focus();
	}
</script>

<div class="scope">
	<span class="lbl" id={gid}>
		<Tooltip tip={ui('scopeHelp')}>{ui('colorScope')} <span aria-hidden="true">ⓘ</span></Tooltip>
	</span>
	<div class="segmented" role="radiogroup" aria-labelledby={gid} tabindex="-1" onkeydown={onkey}>
		{#each options as o (o)}
			<button
				type="button"
				role="radio"
				data-v={o}
				aria-checked={value === o}
				tabindex={value === o ? 0 : -1}
				onclick={() => (value = o)}>{name(o)}</button
			>
		{/each}
	</div>
</div>

<style>
	.scope {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 10px;
		font-size: var(--fs-sm);
	}
	.lbl {
		color: var(--muted);
		font-size: var(--fs-xs);
	}
</style>
