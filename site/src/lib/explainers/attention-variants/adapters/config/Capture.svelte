<!--
  attention-variants Capture for the local 'config' adapter: preset buttons (data.ts PRESETS) and the fields of one
  attention configuration (layers, query heads, KV heads as the divisors of the query heads, head width, tokens, bytes per
  number; the latent width only for MLA), with the cache it gives. Editing a field makes the preset 'custom'; choosing KV
  heads leaves MLA. Reports the raw config.
-->
<script lang="ts">
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { BYTES, configOf, divisors, KIND, kindOf, LIMITS, PRESETS, presetConfig, perToken, total, bytesText, type Config } from '../../data.ts';

	let { raw, onchange, compact = false }: CaptureProps<Config> = $props();
	const c = $derived(configOf(raw));
	const set = (patch: Partial<Config>) => onchange?.(configOf({ ...c, preset: 'custom', ...patch }));
	const num = (e: Event) => Number((e.currentTarget as HTMLInputElement).value);
	const FIELDS = [
		{ key: 'layers', label: { en: 'Layers (blocks) L', zh: '层数（块数）L' }, lim: LIMITS.layers },
		{ key: 'heads', label: { en: 'Query heads H', zh: '查询头数 H' }, lim: LIMITS.heads },
		{ key: 'dHead', label: { en: 'Head width d', zh: '头宽 d' }, lim: LIMITS.dHead },
		{ key: 'tokens', label: { en: 'Tokens T', zh: '词元数 T' }, lim: LIMITS.tokens }
	] as const;
	const BYTE_LABEL: Record<number, string> = { 4: 'fp32', 2: 'fp16 / bf16', 1: 'int8 / fp8' };
</script>

<div class="cfg" class:compact>
	<div class="presets" role="group" aria-label={t({ en: 'Start from a published configuration', zh: '从一个已发表的配置开始' })}>
		{#each PRESETS as p (p.id)}
			<button type="button" class="btn" aria-pressed={c.preset === p.id} onclick={() => onchange?.(presetConfig(p))}>{t(p.label)}</button>
		{/each}
	</div>
	<div class="fields">
		{#each FIELDS as f (f.key)}
			<label>
				<span>{t(f.label)}</span>
				<input type="number" min={f.lim[0]} max={f.lim[1]} step="1" value={c[f.key]} onchange={(e) => set({ [f.key]: num(e) })} />
			</label>
		{/each}
		{#if c.latent > 0}
			<label>
				<span>{t({ en: 'Latent width (MLA)', zh: '潜向量宽度（MLA）' })}</span>
				<input type="number" min="1" max={LIMITS.latent[1]} step="1" value={c.latent} onchange={(e) => set({ latent: Math.max(1, num(e)) })} />
			</label>
		{/if}
		<label>
			<span>{t({ en: 'KV heads K', zh: 'KV 头数 K' })}</span>
			<select value={c.latent > 0 ? 0 : c.kvHeads} onchange={(e) => set({ kvHeads: num(e), latent: 0 })}>
				{#if c.latent > 0}<option value={0}>{t({ en: 'none: a latent (MLA)', zh: '无：缓存潜向量（MLA）' })}</option>{/if}
				{#each divisors(c.heads) as k (k)}<option value={k}>{k}</option>{/each}
			</select>
		</label>
		<label>
			<span>{t({ en: 'Bytes per number b', zh: '每个数的字节数 b' })}</span>
			<select value={c.bytes} onchange={(e) => set({ bytes: num(e) })}>
				{#each BYTES as b (b)}<option value={b}>{b} ({BYTE_LABEL[b]})</option>{/each}
			</select>
		</label>
	</div>
	<p class="out">
		{t(KIND[kindOf(c)])}{colon()}
		<strong class="num">{bytesText(perToken(c))}</strong>
		{t({ en: 'per token,', zh: '每个词元，' })}
		<strong class="num">{bytesText(total(c))}</strong>
		{t({ en: `for ${c.tokens.toLocaleString('en-US')} tokens`, zh: `${c.tokens.toLocaleString('en-US')} 个词元共计` })}
	</p>
</div>

<style>
	.cfg {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 8px;
		min-width: 0;
		max-width: 640px;
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.presets .btn {
		font-size: var(--fs-xs);
	}
	.presets .btn[aria-pressed='true'] {
		border-color: var(--accent);
		font-weight: 600;
	}
	.fields {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));
		gap: 6px 10px;
	}
	.compact .fields {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}
	label {
		display: grid;
		gap: 2px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		min-width: 0;
	}
	input,
	select {
		min-width: 0;
		width: 100%;
		font: inherit;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
	}
	.out {
		margin: 0;
		font-size: var(--fs-sm);
	}
</style>
