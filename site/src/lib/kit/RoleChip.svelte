<!--
  Tensor-role and training-status markers from the visual grammar (brief §4.7):
    param      double-bordered slab with a "W" chip     (weights, biases)
    grad       dashed outline with a "∇" chip           (gradients)
    activation plain chip                               (data on the forward path)
    cache      slab with a "⟲" chip                     (KV cache, cached tensors)
    optimizer  small "m v" badge                        (optimizer state)
    frozen     lock (desaturated)                        (not updated in training)
    trainable  accent outline                            (updated in training)
    lora       thin side branch with the rank            (adapter)
  <RoleChip role="param" />  <RoleChip role="frozen" label="conv1" />  <RoleChip role="lora" rank={8} />
-->
<script lang="ts">
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Role } from './types';

	interface Props {
		role: Role;
		/** text after the glyph; defaults to the role name */
		label?: Text;
		/** LoRA rank */
		rank?: number;
		/** glyph only (the role name is still read by screen readers) */
		iconOnly?: boolean;
	}
	let { role, label, rank, iconOnly = false }: Props = $props();

	const names: Record<Role, () => string> = {
		param: () => ui('roleParameter'),
		grad: () => ui('roleGradient'),
		activation: () => ui('roleActivation'),
		cache: () => ui('roleCache'),
		optimizer: () => ui('roleOptimizer'),
		frozen: () => ui('frozen'),
		trainable: () => ui('trainable'),
		lora: () => 'LoRA'
	};
	const text = $derived(label ? t(label) : names[role]());
</script>

<span class="rc {role}" title={iconOnly ? text : undefined}>
	<span class="g" aria-hidden="true">
		{#if role === 'param'}W{:else if role === 'grad'}∇{:else if role === 'cache'}⟲{:else if role === 'optimizer'}m&thinsp;v{:else if role === 'frozen'}<svg viewBox="0 0 12 12" width="11" height="11"><rect x="2" y="5.2" width="8" height="5.6" rx="1.2" fill="currentColor" /><path d="M3.8 5.2V3.8a2.2 2.2 0 0 1 4.4 0v1.4" fill="none" stroke="currentColor" stroke-width="1.4" /></svg>{:else if role === 'trainable'}<svg viewBox="0 0 12 12" width="11" height="11"><path d="M6 1.5v9M1.5 6h9" stroke="currentColor" stroke-width="1.6" /></svg>{:else if role === 'lora'}<svg viewBox="0 0 14 12" width="13" height="11"><path d="M1 6h12M4 6c0-3 6-3 6 0" fill="none" stroke="currentColor" stroke-width="1.3" /></svg>{:else}<svg viewBox="0 0 12 12" width="11" height="11"><rect x="1.5" y="3.5" width="9" height="5" rx="1" fill="none" stroke="currentColor" stroke-width="1.2" /></svg>{/if}
	</span>
	{#if iconOnly}<span class="visually-hidden">{text}</span>{:else}<span class="t">{text}{#if role === 'lora' && rank} r={rank}{/if}</span>{/if}
</span>

<style>
	.rc {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 1px 7px 1px 3px;
		border-radius: var(--r1);
		font-size: var(--fs-xs);
		line-height: 1.6;
		white-space: nowrap;
		vertical-align: middle;
		color: var(--ink-2);
		background: var(--surface);
		border: 1px solid var(--rule-strong);
	}
	.g {
		display: inline-grid;
		place-items: center;
		min-width: 18px;
		height: 16px;
		padding: 0 3px;
		border-radius: 3px;
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: 11px;
	}
	.param {
		border: 3px double var(--param);
	}
	.param .g {
		background: var(--param);
		color: var(--surface);
	}
	.grad {
		border: 1.5px dashed var(--grad);
		color: var(--grad);
	}
	.grad .g {
		color: var(--grad);
	}
	.cache {
		border: 3px double var(--rule-strong);
	}
	.cache .g {
		background: var(--surface-3);
	}
	.optimizer .g {
		background: var(--surface-3);
		font-size: 10px;
	}
	.frozen {
		filter: grayscale(1);
		color: var(--muted);
		background: var(--surface-2);
	}
	.trainable {
		border: 2px solid var(--accent);
		color: var(--ink);
	}
	.trainable .g {
		color: var(--accent);
	}
	.lora {
		border-color: var(--accent);
		border-style: solid;
	}
	.lora .g {
		color: var(--accent);
	}
</style>
