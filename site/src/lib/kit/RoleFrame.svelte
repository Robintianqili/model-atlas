<!--
  Frames a tensor view with its role from the visual grammar: a double-bordered slab with a
  W chip for parameters, a dashed ∇ outline for gradients, desaturated + lock when frozen,
  an accent outline when trainable.

  <RoleFrame role="param" label="conv1.weight [8×1×3×3]"><TensorGrid … /></RoleFrame>
  <RoleFrame role="param" status="frozen" label="…">…</RoleFrame>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { keyScroll } from './keyscroll';
	import RoleChip from './RoleChip.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Role } from './types';

	interface Props {
		role?: Extract<Role, 'param' | 'grad' | 'activation' | 'cache'>;
		status?: 'frozen' | 'trainable';
		label?: Text;
		children: Snippet;
	}
	let { role = 'activation', status, label, children }: Props = $props();
</script>

<figure class="rf {role} {status ?? ''}">
	<figcaption>
		<RoleChip {role} iconOnly={!!label} />
		{#if status}<RoleChip role={status} iconOnly />{/if}
		{#if label}<span class="lbl">{t(label)}</span>{/if}
	</figcaption>
	<div class="body" use:keyScroll>{@render children()}</div>
</figure>

<style>
	.rf {
		margin: 0;
		display: inline-flex;
		flex-direction: column;
		gap: 6px;
		padding: 8px;
		border-radius: var(--r2);
		background: var(--surface);
		border: 1px solid var(--rule);
		max-width: 100%;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.lbl {
		font-family: var(--font-mono);
	}
	.body {
		max-width: 100%;
		overflow-x: auto;
	}
	.param {
		border: 4px double var(--param);
	}
	.cache {
		border: 4px double var(--rule-strong);
	}
	.grad {
		border: 2px dashed var(--grad);
	}
	.frozen .body {
		filter: grayscale(0.85) opacity(0.75);
	}
	.frozen {
		background: var(--surface-2);
	}
	.trainable {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
</style>
