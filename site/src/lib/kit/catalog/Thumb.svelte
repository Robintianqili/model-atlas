<!--
  A lesson's poster thumbnail (redesign spec §4.5, §10.5): one tile of a CSS sprite from `static/thumbs/` (atlas thumbs),
  the light or the dark file by theme. Two layers carry the two files and only the active theme's layer is displayed,
  so only that sprite downloads. The url sits in each layer's own style attribute: prerendered pages use a relative base
  path, and a relative url() resolves against the document there (in a stylesheet or a custom property it would resolve
  against the CSS file). Without a tile it shows the area's glyph on the stage colour. Decorative: the row or card it
  sits in carries the title.
    <Thumb tile={l.thumb} sprite={index.sprites.rows} group={l.group} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '../Icon.svelte';
	import { glyphFor } from './glyphs';
	import type { Tile } from './types';

	let { tile, sprite, group }: { tile?: Tile; sprite?: { light: string; dark: string }; group: string } = $props();
</script>

{#if tile && sprite}
	<span class="thumb" aria-hidden="true">
		<i class="lt" style="background-image: url({base}{sprite.light}); background-size: {tile.size}; background-position: {tile.pos}"></i>
		<i class="dk" style="background-image: url({base}{sprite.dark}); background-size: {tile.size}; background-position: {tile.pos}"></i>
	</span>
{:else}
	<span class="thumb fb" aria-hidden="true"><Icon d={glyphFor(group)} glyph size={28} /></span>
{/if}

<style>
	.thumb {
		position: relative;
		display: block;
		aspect-ratio: 16 / 10;
		width: 100%;
		background-color: var(--stage);
		border-radius: inherit;
		overflow: hidden;
	}
	i {
		position: absolute;
		inset: 0;
		background-repeat: no-repeat;
	}
	.dk {
		display: none;
	}
	:global(:root[data-theme='dark']) .lt {
		display: none;
	}
	:global(:root[data-theme='dark']) .dk {
		display: block;
	}
	@media (prefers-color-scheme: dark) {
		:global(:root:not([data-theme='light'])) .lt {
			display: none;
		}
		:global(:root:not([data-theme='light'])) .dk {
			display: block;
		}
	}
	.fb {
		display: grid;
		place-items: center;
		color: var(--muted);
	}
</style>
