<!--
  Renders one view with its StageProps (gallery). Lives in the gallery's lazy module, so the spread of the stage props
  happens there and not in the gallery's route node (see gallery.ts on chunk splits).
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import type { StageProps } from '../../../types/index.ts';

	interface Props {
		view: Component<StageProps>;
		stage: Omit<StageProps, 'ctx'>;
		ctx: NonNullable<StageProps['ctx']>;
	}
	let { view, stage, ctx }: Props = $props();
	const View = $derived(view);
</script>

<View {...stage} {ctx} />
