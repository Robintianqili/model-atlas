<!--
  decoding Capture for the local 'prompt' adapter: the kit's text-bpe widget (the prompt and the tokens it becomes, with
  TinyGPT's tokenizer) and, under it, the decoding settings: temperature (0 = greedy), top-k (0 = off), top-p (1 = off)
  and the seed of the random numbers, with a button for a new one. Reports { text, ids, temperature, topK, topP, seed }.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import TextBpe from '$lib/atlas/adapters/text-bpe/Capture.svelte';
	import type { CaptureProps } from '$lib/atlas/ui/capture';
	import { HIGH_T, settingsOf, type Settings } from '../../decode.ts';
	import type { PromptRaw } from './adapter.ts';
	import { fmtSetting } from '../../../../atlas/core/format.ts';

	let { rt, raw, presets = [], onchange, compact = false }: CaptureProps = $props();

	const s = $derived(settingsOf(raw));
	const r = $derived(raw && typeof raw === 'object' ? (raw as NonNullable<PromptRaw>) : null);
	const text = $derived(r ? { text: r.text, ids: r.ids } : null);
	function set(patch: Partial<Settings>, live = false) {
		onchange({ ...(r ?? {}), ...s, ...patch }, { live });
	}
	const fmtT = (v: number) => (v === 0 ? t({ en: '0 (greedy)', zh: '0（贪心）' }) : fmtSetting(v));
	const dials = $derived([
		{ k: 'temperature' as const, min: 0, max: HIGH_T, step: 0.05, name: { en: 'Temperature T', zh: '温度 T' }, short: 'T', shown: fmtT(s.temperature) },
		{ k: 'topK' as const, min: 0, max: 50, step: 1, name: { en: 'Top-k (0 = off)', zh: 'top-k（0 = 关）' }, short: 'k', shown: s.topK === 0 ? t({ en: 'off', zh: '关' }) : String(s.topK) },
		{ k: 'topP' as const, min: 0.05, max: 1, step: 0.05, name: { en: 'Top-p (1 = off)', zh: 'top-p（1 = 关）' }, short: 'p', shown: s.topP >= 1 ? t({ en: 'off', zh: '关' }) : fmtSetting(s.topP) }
	]);
	const round = (v: number) => Math.round(v * 100) / 100;
</script>

<div class="prompt" class:compact>
	<TextBpe {rt} raw={text} {presets} {compact} onchange={(next, o) => onchange({ ...s, ...(next as object) }, o)} />
	<div class="dials" role="group" aria-label={t({ en: 'Decoding settings', zh: '解码设置' })}>
		{#each dials as dl (dl.k)}
			<label class="row">
				<span class="name">{compact ? dl.short : t(dl.name)}</span>
				<input
					type="range"
					min={dl.min}
					max={dl.max}
					step={dl.step}
					value={s[dl.k]}
					oninput={(e) => set({ [dl.k]: round(Number((e.currentTarget as HTMLInputElement).value)) }, true)}
					onchange={(e) => set({ [dl.k]: round(Number((e.currentTarget as HTMLInputElement).value)) })}
				/>
				<output class="val num">{dl.shown}</output>
			</label>
		{/each}
		<div class="seed">
			<span class="name">{t({ en: 'Seed of the random numbers', zh: '随机数种子' })}</span>
			<output class="num">{s.seed}</output>
			<button type="button" class="btn" onclick={() => set({ seed: s.seed + 1 })}>{t({ en: 'New seed', zh: '换个种子' })}</button>
		</div>
	</div>
</div>

<style>
	.prompt {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.dials {
		display: grid;
		gap: 6px;
		max-width: 520px;
		min-width: 0;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 9rem) minmax(0, 1fr) 5.5rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 1.2rem minmax(0, 1fr) 5rem;
	}
	.name {
		color: var(--ink-2);
	}
	input[type='range'] {
		width: 100%;
		min-width: 0;
		accent-color: var(--accent);
	}
	.val {
		text-align: right;
	}
	.seed {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.seed .btn {
		font-size: var(--fs-xs);
	}
	@media (max-width: 420px) {
		.prompt:not(.compact) .row {
			grid-template-columns: minmax(0, 1fr) 5.5rem;
		}
		.prompt:not(.compact) .row .name {
			grid-column: 1 / -1;
		}
	}
</style>
