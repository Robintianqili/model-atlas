<!-- tinyseq2seq: the kit's text-char input box, with the preset dates of presets.json (adapter.ts). -->
<script lang="ts">
	import TextChar from '../../../../atlas/adapters/text-char/Capture.svelte';
	import type { CaptureProps } from '../../../../atlas/ui/capture.ts';

	let p: CaptureProps = $props();
</script>

<TextChar {...p} />
