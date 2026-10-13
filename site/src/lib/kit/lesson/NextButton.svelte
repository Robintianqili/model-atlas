<!--
  NextButton (redesign spec §4.4): the one primary of the lesson. Two lines: "Next · step k" (12 px) over the next step's
  title (ellipsis; the whole title is in the accessible name). Variants: next, start (the cover), continue (after
  feedback), skip (an unanswered predict: outline, not the act style), finish (the last step).
-->
<script lang="ts">
	import Icon from '../Icon.svelte';
	import { right } from '../icons';
	import { lui } from '$lib/i18n/i18n.svelte';

	interface Props {
		kind: 'next' | 'start' | 'continue' | 'skip' | 'finish' | 'resume';
		/** the 1-based number of the step it opens */
		k: number;
		/** that step's title (finish: what it does) */
		title: string;
		onclick: () => void;
		disabled?: boolean;
		el?: HTMLButtonElement;
	}
	let { kind, k, title, onclick, disabled = false, el = $bindable() }: Props = $props();
	const top = $derived(
		kind === 'start'
			? lui('startStep', { k })
			: kind === 'continue'
				? lui('continueStep', { k })
				: kind === 'skip'
					? lui('skipStep', { k })
					: kind === 'resume'
						? lui('resumeAt', { k })
						: kind === 'finish'
							? lui('finish')
							: lui('nextStep', { k })
	);
</script>

<button bind:this={el} type="button" class="btn btn-next" class:btn-act={kind !== 'skip'} class:skip={kind === 'skip'} aria-label={`${top}: ${title}`} aria-disabled={disabled || undefined} {onclick}>
	<span class="nx"><small>{top}</small><span class="nt">{title}</span></span>
	<Icon d={right} size={18} />
</button>

<style>
	.btn-next {
		justify-content: space-between;
		gap: 14px;
		min-width: 180px;
		max-width: 380px;
		min-height: 52px;
		padding: 0 14px 0 18px;
		text-align: left;
	}
	.nx {
		display: flex;
		flex-direction: column;
		min-width: 0;
		line-height: 1.2;
	}
	small {
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
		opacity: 0.78;
	}
	.nt {
		overflow: hidden;
		font-size: var(--t-md);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.skip small {
		opacity: 1;
		color: var(--muted);
	}
	.btn-next[aria-disabled='true'] {
		opacity: 0.5;
		cursor: default;
	}
	@media (max-width: 1099px) {
		.btn-next {
			min-width: 0;
			max-width: 260px;
		}
	}
	@media (max-width: 759px) {
		.btn-next {
			width: 100%;
			max-width: none;
			min-height: 48px;
			padding: 0 10px 0 14px;
		}
		.nt {
			font-size: var(--t-sm);
		}
	}
</style>
