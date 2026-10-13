<!--
  A UI icon on the DevKit pages: the same drawing as $lib/kit/Icon.svelte (24 grid, 1.75 stroke, round caps), decorative
  (aria-hidden; a word always sits beside it). A local copy on purpose: kit Icon.svelte is otherwise imported only by
  the site header, and a direct import from a DevKit page splits it out of the shared shell chunk into a chunk of its own,
  which every page then preloads (+179 B gzip on the planned model pages, measured).
    <Glyph d={arrow} size={18} />      d: a named export of ./icons.ts
-->
<script lang="ts">
	let { d, size = 20, class: klass = '' }: { d: string; size?: number; class?: string } = $props();
</script>

<svg
	class="icon {klass}"
	width={size}
	height={size}
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width="1.75"
	stroke-linecap="round"
	stroke-linejoin="round"
	aria-hidden="true"
	focusable="false">{@html d}</svg
>

<style>
	.icon {
		flex: none;
		display: block;
	}
</style>
