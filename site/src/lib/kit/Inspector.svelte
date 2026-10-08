<!--
  Inspector: the shared layer inspector (brief §4.3). A panel (or a drawer) with tabs
  Values / Stats / Math / Code and previous/next layer buttons; "[" and "]" move between
  layers from anywhere on the page (except while typing in a field).

  Minimal (default views from the trace):
    <Inspector layers={spec.layers} bind:current tensorOf={(id) => trace?.t[id]} provenance="live" />
  Custom tabs (snippets get the current layer id):
    <Inspector layers={…} bind:current tensorOf={…}>
      {#snippet math(id)}<FormulaCard tex={…} values={…} />{/snippet}
      {#snippet code(id)}<CodeBlock code={…} highlight={[3]} />{/snippet}
    </Inspector>

  A tab is shown when its snippet is given; Values and Stats also appear when `tensorOf`
  returns a tensor (TensorView with a color-scope switch, TensorStats).
  `maxRows` (opt-in, DevKit K1): a [rows, cols] tensor with more rows shows a window of that many rows with
  previous / next buttons (a [128, 64] layer drew thousands of pixels of cells).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import TensorView from './TensorView.svelte';
	import TensorStats from './TensorStats.svelte';
	import ScopeSwitch from './ScopeSwitch.svelte';
	import ProvenanceBadge from './ProvenanceBadge.svelte';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmtShape } from './format';
	import type { ColorScope, Domain, ScaleKind } from './color';
	import type { LayerNode, Provenance, TensorLike } from './types';

	type Tab = 'values' | 'stats' | 'math' | 'code';

	interface Props {
		layers: LayerNode[];
		current?: string;
		tensorOf?: (id: string) => TensorLike | undefined;
		kindOf?: (id: string) => ScaleKind;
		scope?: ColorScope;
		scopeDomainsOf?: (id: string) => { layer?: Domain; global?: Domain };
		values?: Snippet<[string]>;
		stats?: Snippet<[string]>;
		math?: Snippet<[string]>;
		code?: Snippet<[string]>;
		/** extra content under the title (any layer) */
		header?: Snippet<[string]>;
		provenance?: Provenance;
		precision?: string;
		tolerance?: number;
		title?: Text;
		/** render as a fixed drawer (right side; bottom sheet on phones) */
		drawer?: boolean;
		open?: boolean;
		/** listen for [ and ] on the whole page */
		keys?: boolean;
		tab?: Tab;
		/** show at most this many rows of a 2-D tensor at a time (opt-in) */
		maxRows?: number;
	}
	let {
		layers,
		current = $bindable(layers[0]?.id ?? ''),
		tensorOf,
		kindOf,
		scope = $bindable('layer'),
		scopeDomainsOf,
		values,
		stats,
		math,
		code,
		header,
		provenance,
		precision,
		tolerance,
		title,
		drawer = false,
		open = $bindable(true),
		keys = true,
		tab = $bindable('values'),
		maxRows
	}: Props = $props();

	const gid = $props.id();
	const idx = $derived(Math.max(0, layers.findIndex((l) => l.id === current)));
	const layer = $derived(layers[idx]);
	const tensor = $derived(tensorOf?.(current));
	const kind = $derived<ScaleKind>(kindOf?.(current) ?? 'diverging');
	const tabs = $derived(
		(
			[
				['values', !!values || !!tensorOf],
				['stats', !!stats || !!tensorOf],
				['math', !!math],
				['code', !!code]
			] as [Tab, boolean][]
		)
			.filter(([, ok]) => ok)
			.map(([k]) => k)
	);
	const tabName = (k: Tab) =>
		k === 'values' ? ui('tabValues') : k === 'stats' ? ui('tabStats') : k === 'math' ? ui('tabMath') : ui('tabCode');
	let tabEls: Record<string, HTMLButtonElement> = $state({});
	let channel = $state(0);
	/** the first row of the window of a tall 2-D tensor (maxRows) */
	let rowFrom = $state(0);
	const rows = $derived(tensor && tensor.shape.length === 2 && maxRows && tensor.shape[0] > maxRows ? tensor.shape[0] : 0);
	const shown = $derived.by((): TensorLike | undefined => {
		if (!tensor || !rows || !maxRows) return tensor;
		const cols = tensor.shape[1];
		const a = Math.max(0, Math.min(rowFrom, rows - maxRows));
		const d = tensor.data as ArrayLike<number> & { subarray?: (s: number, e: number) => ArrayLike<number> };
		const data = d.subarray ? d.subarray(a * cols, (a + maxRows) * cols) : Array.prototype.slice.call(d, a * cols, (a + maxRows) * cols);
		return { ...tensor, shape: [maxRows, cols], data } as TensorLike;
	});

	$effect(() => {
		if (tabs.length && !tabs.includes(tab)) tab = tabs[0];
	});
	$effect(() => {
		void current;
		channel = 0;
		rowFrom = 0;
	});

	function move(d: number) {
		const j = Math.max(0, Math.min(layers.length - 1, idx + d));
		current = layers[j].id;
	}
	function onWindowKey(e: KeyboardEvent) {
		if (!keys || (drawer && !open)) return;
		if (e.key !== '[' && e.key !== ']') return;
		// leave Cmd+[ / Ctrl+[ (browser back, editor shortcuts) alone; Ctrl+Alt is AltGr on some layouts
		if (e.defaultPrevented || e.metaKey || e.ctrlKey !== e.altKey) return;
		const el = e.target as HTMLElement | null;
		if (el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))) return;
		e.preventDefault();
		move(e.key === ']' ? 1 : -1);
	}
	function onTabKey(e: KeyboardEvent) {
		const i = tabs.indexOf(tab);
		let j = i;
		if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
		else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = tabs.length - 1;
		else return;
		e.preventDefault();
		tab = tabs[j];
		tabEls[tab]?.focus();
	}
	function onPanelKey(e: KeyboardEvent) {
		if (drawer && e.key === 'Escape') open = false;
	}
</script>

<svelte:window onkeydown={onWindowKey} />

{#if !drawer || open}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions (Escape closes the drawer) -->
	<aside class="insp" class:drawer aria-labelledby="{gid}-h" onkeydown={onPanelKey}>
		<header class="top">
			<div class="ttl">
				<span class="eyebrow">{title ? t(title) : ui('inspector')}</span>
				<h3 id="{gid}-h">
					{layer ? t(layer.label) : ''}
					{#if layer?.op}<span class="op num">{layer.op}</span>{/if}
				</h3>
				{#if layer?.outShape}<span class="shape num">{fmtShape(layer.outShape)}</span>{/if}
			</div>
			<div class="nav">
				{#if provenance}<ProvenanceBadge mode={provenance} {precision} {tolerance} compact />{/if}
				<button type="button" class="btn icon" onclick={() => move(-1)} disabled={idx === 0} aria-label={ui('prevLayer')} title="{ui('prevLayer')} ([)">
					<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M10.5 3 5 8l5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
				</button>
				<span class="pos num">{idx + 1}/{layers.length}</span>
				<button type="button" class="btn icon" onclick={() => move(1)} disabled={idx >= layers.length - 1} aria-label={ui('nextLayer')} title="{ui('nextLayer')} (])">
					<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M5.5 3 11 8l-5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
				</button>
				{#if drawer}
					<button type="button" class="btn icon" onclick={() => (open = false)} aria-label={ui('close')}>✕</button>
				{/if}
			</div>
		</header>
		{#if header}<div class="hdr">{@render header(current)}</div>{/if}
		{#if tabs.length}
			<div class="tabs" role="tablist" aria-label={ui('inspector')} tabindex="-1" onkeydown={onTabKey}>
				{#each tabs as k (k)}
					<button
						bind:this={tabEls[k]}
						type="button"
						role="tab"
						id="{gid}-tab-{k}"
						aria-selected={tab === k}
						aria-controls="{gid}-panel"
						tabindex={tab === k ? 0 : -1}
						onclick={() => (tab = k)}>{tabName(k)}</button
					>
				{/each}
			</div>
			<div class="tabpanel" role="tabpanel" id="{gid}-panel" aria-labelledby="{gid}-tab-{tab}">
				{#if tab === 'values'}
					{#if values}
						{@render values(current)}
					{:else if tensor && layer}
						<div class="scope"><ScopeSwitch bind:value={scope} /></div>
						{#if rows && maxRows && shown}
							{@const a = Math.max(0, Math.min(rowFrom, rows - maxRows))}
							<div class="win num">
								<button type="button" class="btn ghost" disabled={a === 0} onclick={() => (rowFrom = Math.max(0, a - maxRows))}>↑</button>
								<span>{t({ en: `rows ${a}–${a + maxRows - 1} of ${rows} (counting from 0)`, zh: `第 ${a}–${a + maxRows - 1} 行，共 ${rows} 行（从 0 数起）` })}</span>
								<button type="button" class="btn ghost" disabled={a + maxRows >= rows} onclick={() => (rowFrom = Math.min(rows - maxRows, a + maxRows))}>↓</button>
							</div>
							<TensorView tensor={shown} label={layer.label} {kind} {scope} scopeDomains={scopeDomainsOf?.(current) ?? {}} bind:channel />
						{:else}
							<TensorView {tensor} label={layer.label} {kind} {scope} scopeDomains={scopeDomainsOf?.(current) ?? {}} bind:channel />
						{/if}
					{:else}
						<p class="muted">{ui('noDataYet')}</p>
					{/if}
				{:else if tab === 'stats'}
					{#if stats}
						{@render stats(current)}
					{:else if tensor}
						<TensorStats {tensor} {kind} />
					{:else}
						<p class="muted">{ui('noDataYet')}</p>
					{/if}
				{:else if tab === 'math' && math}
					{@render math(current)}
				{:else if tab === 'code' && code}
					{@render code(current)}
				{/if}
			</div>
		{/if}
		<p class="keys">{ui('layerKeys')}</p>
	</aside>
{/if}

<style>
	.win {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-xs);
	}
	.insp {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		padding: 14px 16px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		min-width: 0;
	}
	.drawer {
		position: fixed;
		z-index: 40;
		top: calc(var(--header-h) + 8px);
		right: 8px;
		bottom: 8px;
		width: min(460px, calc(100vw - 16px));
		overflow-y: auto;
		box-shadow: var(--shadow-2);
	}
	@media (max-width: 720px) {
		.drawer {
			top: auto;
			left: 0;
			right: 0;
			bottom: 0;
			width: 100%;
			max-height: 72vh;
			border-radius: var(--r3) var(--r3) 0 0;
		}
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: flex-start;
		gap: 8px;
	}
	.ttl {
		display: grid;
		gap: 2px;
		min-width: 0;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-lg);
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 8px;
	}
	.op {
		font-size: var(--fs-xs);
		font-weight: 500;
		color: var(--muted);
		padding: 1px 6px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
	}
	.shape {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.nav {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.nav .btn.icon {
		width: 36px;
		min-height: 36px;
	}
	.pos {
		font-size: var(--fs-xs);
		color: var(--muted);
		min-width: 3em;
		text-align: center;
	}
	.tabs {
		display: flex;
		gap: 2px;
		border-bottom: 1px solid var(--rule);
		overflow-x: auto;
	}
	.tabs button {
		padding: 8px 12px;
		border: 0;
		border-bottom: 2px solid transparent;
		background: none;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		white-space: nowrap;
	}
	.tabs button[aria-selected='true'] {
		color: var(--ink);
		font-weight: 700;
		border-bottom-color: var(--accent);
	}
	.tabpanel {
		min-width: 0;
		min-height: 80px;
	}
	.scope {
		margin-bottom: 8px;
	}
	.keys {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
