<!--
  FeedbackBar (redesign spec §4.4, §5.7): after a predict answer it replaces the action bar on screens 760 px and up,
  so the stage never moves: a ✓ / ✕ badge, "Right." / "Not quite.", the explanation with its real number, "Why it
  matters" (when the step has a Why) and Continue. Icon + word + colour, never colour alone.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from '../Icon.svelte';
	import { check, x } from '../icons';
	import { lui, t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		right: boolean;
		explain: Text;
		onwhy?: () => void;
		/** the Continue button */
		next: Snippet;
	}
	let { right, explain, onwhy, next }: Props = $props();
</script>

<!-- the Walkthrough's live region announces the feedback once; this bar is what the eye reads -->
<div class="fb" class:no={!right}>
	<span class="badge" aria-hidden="true"><Icon d={right ? check : x} size={24} /></span>
	<p class="msg"><b>{right ? lui('right') : lui('notQuite')}</b> {t(explain)}</p>
	<div class="act">
		{#if onwhy}<button type="button" class="btn btn-quiet why" onclick={onwhy}>{lui('whyItMatters')}</button>{/if}
		{@render next()}
	</div>
</div>

<style>
	.fb {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 20px;
		min-height: var(--abar-h);
		padding: 12px 20px;
		border-top: 2px solid var(--ok);
		background: var(--ok-soft);
	}
	.fb.no {
		border-top-color: var(--no);
		background: var(--no-soft);
	}
	.badge {
		display: grid;
		place-items: center;
		width: 44px;
		height: 44px;
		border-radius: 50%;
		background: var(--ok);
		color: var(--ok-soft);
	}
	.no .badge {
		background: var(--no);
		color: var(--no-soft);
	}
	.msg {
		max-width: 760px;
		margin: 0;
		color: var(--ink);
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.msg b {
		margin-right: 6px;
		color: var(--ok);
		font-size: var(--t-lg);
	}
	.no .msg b {
		color: var(--no);
	}
	.act {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.why {
		white-space: nowrap;
	}
	@media (max-width: 1099px) {
		.why {
			display: none;
		}
	}
	@media (prefers-reduced-motion: no-preference) {
		.fb {
			animation: fb-in var(--dur-3) var(--ease);
		}
	}
	@keyframes fb-in {
		from {
			opacity: 0;
		}
	}
</style>
