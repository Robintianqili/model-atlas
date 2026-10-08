<!--
  TextInputBar: a text field (input or textarea) with bilingual presets and a length cap
  (counted in characters / code points, so 我 counts as 1).

  <TextInputBar bind:value maxLength={64} label={{ en: 'Prompt', zh: '提示词' }}
    presets={[{ text: 'Once upon a time' }, { text: '我爱自然语言处理', label: { en: 'Chinese', zh: '中文' } }]}
    oninput={(v) => run(v)} />

  Text never leaves the device; the bar says so.
-->
<script lang="ts">
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		value?: string;
		maxLength?: number;
		multiline?: boolean;
		rows?: number;
		label?: Text;
		placeholder?: Text;
		presets?: { text: string; label?: Text }[];
		/** show the Run button and call onsubmit (Enter in a single-line field also submits) */
		submit?: boolean;
		oninput?: (value: string) => void;
		onsubmit?: (value: string) => void;
	}
	let {
		value = $bindable(''),
		maxLength = 256,
		multiline = false,
		rows = 3,
		label,
		placeholder,
		presets = [],
		submit = false,
		oninput,
		onsubmit
	}: Props = $props();

	const id = $props.id();
	const length = $derived([...value].length);
	const full = $derived(length >= maxLength);

	function cap(v: string): string {
		const cps = [...v];
		return cps.length > maxLength ? cps.slice(0, maxLength).join('') : v;
	}
	function handle(e: Event) {
		const el = e.currentTarget as HTMLInputElement | HTMLTextAreaElement;
		const capped = cap(el.value);
		if (capped !== el.value) el.value = capped;
		value = capped;
		oninput?.(value);
	}
	function onkeydown(e: KeyboardEvent) {
		// Enter that confirms an IME candidate (Chinese/Japanese input) must not submit.
		// keyCode 229 covers browsers that report the composing Enter after compositionend.
		if (e.isComposing || e.keyCode === 229) return;
		if (e.key === 'Enter' && !multiline && submit) {
			e.preventDefault();
			onsubmit?.(value);
		}
	}
	function usePreset(text: string) {
		value = cap(text);
		oninput?.(value);
		onsubmit?.(value);
	}
</script>

<div class="tib">
	<div class="head">
		{#if label}<label for={id} class="lbl">{t(label)}</label>{/if}
		<span class="meta">
			<span class="dev"><svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true"><rect x="2" y="5.2" width="8" height="5.6" rx="1.2" fill="currentColor" /><path d="M3.8 5.2V3.8a2.2 2.2 0 0 1 4.4 0v1.4" fill="none" stroke="currentColor" stroke-width="1.4" /></svg> {ui('staysOnDevice')}</span>
			<span class="count num" class:full aria-live="polite">{ui('charCount', { n: length, max: maxLength })}</span>
		</span>
	</div>
	<div class="field">
		{#if multiline}
			<textarea
				{id}
				{rows}
				{value}
				placeholder={placeholder ? t(placeholder) : ui('typeHere')}
				oninput={handle}
				spellcheck="false"
				aria-label={label ? undefined : ui('typeHere')}
			></textarea>
		{:else}
			<input
				{id}
				type="text"
				{value}
				placeholder={placeholder ? t(placeholder) : ui('typeHere')}
				oninput={handle}
				{onkeydown}
				spellcheck="false"
				autocomplete="off"
				aria-label={label ? undefined : ui('typeHere')}
			/>
		{/if}
		{#if submit}
			<button type="button" class="btn primary" onclick={() => onsubmit?.(value)}>{ui('run')}</button>
		{/if}
	</div>
	{#if full}<p class="warn">{ui('limitReached')}</p>{/if}
	{#if presets.length}
		<div class="presets" role="group" aria-label={ui('presets')}>
			{#each presets as p, i (i)}
				<button type="button" class="preset" onclick={() => usePreset(p.text)} lang={/[一-鿿]/.test(p.text) ? 'zh-CN' : undefined}>
					{p.label ? t(p.label) : p.text}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.tib {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 6px;
		min-width: 0;
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: baseline;
		gap: 4px 12px;
	}
	.lbl {
		font-weight: 650;
		font-size: var(--fs-sm);
	}
	.meta {
		display: inline-flex;
		gap: 12px;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.dev {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.count.full {
		color: var(--illus);
		font-weight: 700;
	}
	.field {
		display: flex;
		gap: 6px;
	}
	input,
	textarea {
		flex: 1;
		min-width: 0;
		padding: 10px 12px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		font-family: var(--font-mono);
		font-size: 1rem;
		line-height: 1.5;
		resize: vertical;
	}
	input:focus-visible,
	textarea:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
		border-color: var(--accent);
	}
	.field .btn {
		min-height: 46px;
	}
	.warn {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--illus);
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.preset {
		min-height: 32px;
		padding: 0 10px;
		border-radius: var(--r-pill);
		border: 1px solid var(--rule-strong);
		background: var(--surface);
		font-size: var(--fs-sm);
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.preset:hover {
		background: var(--surface-2);
	}
</style>
