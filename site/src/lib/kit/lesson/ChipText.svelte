<!--
  ChipText (redesign spec §3.4 NumberChip, R1): a narration or look line whose bound numbers name a stage target
  ({name@ref} in the template; the value arrives wrapped in private-use marks, lesson/narrate.ts CHIP). Each such number
  is a chip: mono at weight 600 with a dotted underline; hover, focus or tap moves the stage's ring to its target
  (onref), and leaving it gives the ring back to the step.
-->
<script lang="ts">
	let { text, onref }: { text: string; onref?: (ref: string | null) => void } = $props();
	const RE = /([^]*)([^]*)/g;
	const parts = $derived.by(() => {
		const out: { text: string; ref?: string }[] = [];
		let last = 0;
		for (const m of text.matchAll(RE)) {
			if (m.index! > last) out.push({ text: text.slice(last, m.index) });
			out.push({ ref: m[1], text: m[2] });
			last = m.index! + m[0].length;
		}
		if (last < text.length) out.push({ text: text.slice(last) });
		return out;
	});
</script>

{#each parts as p, i (i)}{#if p.ref}<span
			class="nc"
			data-ref={p.ref}
			role="button"
			tabindex="0"
			onmouseenter={() => onref?.(p.ref!)}
			onmouseleave={() => onref?.(null)}
			onfocus={() => onref?.(p.ref!)}
			onblur={() => onref?.(null)}
			onclick={() => onref?.(p.ref!)}
			onkeydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					onref?.(p.ref!);
				}
			}}>{p.text}</span
		>{:else}{p.text}{/if}{/each}
