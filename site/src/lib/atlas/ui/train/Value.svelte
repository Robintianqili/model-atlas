<!--
  A number in the visual grammar for the kind-B views: monospace, 3 significant digits (fmt3), the full value as the
  title and accessible name. (The kit's <Num> adds a tap tooltip, but importing it here would move it out of the chunk
  /m/tinycnn/ preloads; see notes/devkit-engine.md.) '—' for a value a frame does not have.
-->
<script lang="ts">
	import { fmt3, fmtFull } from '../../core/format.ts';

	let { value, digits = 3, signed = false, unit = '' }: { value: number | null | undefined; digits?: number; signed?: boolean; unit?: string } = $props();
	const ok = $derived(typeof value === 'number' && Number.isFinite(value));
	const short = $derived(ok ? fmt3(value, digits) : '—');
	const full = $derived(ok ? fmtFull(value) : '');
</script>

<span class="num v" class:neg={signed && ok && (value as number) < 0} class:pos={signed && ok && (value as number) > 0} title={full ? full + unit : undefined} aria-label={full ? full + unit : undefined}>{short}{ok ? unit : ''}</span>

<style>
	.v {
		white-space: nowrap;
	}
	.neg {
		color: var(--div-neg-text);
	}
	.pos {
		color: var(--div-pos-text);
	}
</style>
