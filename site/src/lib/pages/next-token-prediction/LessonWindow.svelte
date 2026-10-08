<!--
  The window the lesson follows ("Window k of N: context -> next token"), with buttons to follow
  the previous or next one. Shown on top of the lesson's stages 3-6.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { visibleToken } from '$lib/kit/format';
	import type { NtpLesson } from './lessonState.svelte';
	import { tailText } from './lessonMath';

	let { L }: { L: NtpLesson } = $props();
	const S = $derived(L.S);

	const tail = $derived(tailText(L.ctx, (ids) => S.decode(ids), 44, (id) => S.disp(id)));
</script>

{#if L.pos >= 1}
	<div class="wp" role="group" aria-label={t({ en: 'The window the lesson follows', zh: '课程跟随的窗口' })}>
		<button
			type="button"
			class="btn icon"
			aria-disabled={L.pos <= 1}
			aria-label={t({ en: 'Previous window', zh: '上一个窗口' })}
			onclick={() => L.pos > 1 && L.pick(L.pos - 1)}
		>
			<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M10.5 3 5 8l5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
		</button>
		<div class="mid">
			<span class="k num">
				{t({ en: `Window ${L.pos} of ${L.windows}`, zh: `窗口 ${L.pos} / ${L.windows}` })}{#if L.picked === null}<span class="muted"
						>{t({ en: ' · the last one', zh: ' · 最后一个' })}</span
					>{/if}
			</span>
			<span class="pair">
				<span class="ctx">{visibleToken(tail)}</span>
				<span class="arr" aria-hidden="true">→</span>
				<span class="tgt">{visibleToken(S.disp(L.target))}</span>
			</span>
		</div>
		<button
			type="button"
			class="btn icon"
			aria-disabled={L.pos >= L.windows}
			aria-label={t({ en: 'Next window', zh: '下一个窗口' })}
			onclick={() => L.pos < L.windows && L.pick(L.pos + 1)}
		>
			<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M5.5 3 11 8l-5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
		</button>
	</div>
{/if}

<style>
	.wp {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 8px;
		padding: 6px 8px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.btn.icon {
		min-width: 36px;
		min-height: 36px;
		padding: 0;
		justify-content: center;
	}
	.btn[aria-disabled='true'] {
		opacity: 0.4;
		cursor: default;
	}
	.mid {
		display: grid;
		gap: 2px;
		min-width: 0;
	}
	.k {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.pair {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 2px 6px;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		min-width: 0;
	}
	.ctx {
		color: var(--ink-2);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		min-width: 0;
	}
	.arr {
		color: var(--muted);
	}
	.tgt {
		white-space: pre;
		border: 1.5px solid var(--real);
		background: color-mix(in srgb, var(--real) 16%, var(--surface));
		border-radius: 4px;
		padding: 0 5px;
		font-weight: 700;
		color: var(--ink);
	}
</style>
