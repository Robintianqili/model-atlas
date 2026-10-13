<!--
  One icon from $lib/kit/icons (redesign spec §4.2). Decorative by default (aria-hidden): put a word beside it, or a label
  on its button. With `label` it is the only content and gets role="img".
    <Icon d={search} />                 24-grid UI icon, 20 px, 1.75 stroke
    <Icon d={glyphVision} glyph />      20-grid area or section glyph, 1.6 stroke
    <Icon d={info} size={18} label={ui('about')} />
-->
<script lang="ts">
	interface Props {
		/** the icon's inner SVG markup (a named export of $lib/kit/icons) */
		d: string;
		/** rendered size in px (default 20) */
		size?: number;
		/** a 20-grid glyph (areas, sections) instead of a 24-grid UI icon */
		glyph?: boolean;
		/** accessible name when the icon is the only content */
		label?: string;
		class?: string;
	}
	let { d, size = 20, glyph = false, label, class: klass = '' }: Props = $props();
</script>

<svg
	class="icon {klass}"
	width={size}
	height={size}
	viewBox={glyph ? '0 0 20 20' : '0 0 24 24'}
	fill="none"
	stroke="currentColor"
	stroke-width={glyph ? 1.6 : 1.75}
	stroke-linecap="round"
	stroke-linejoin="round"
	role={label ? 'img' : undefined}
	aria-label={label}
	aria-hidden={label ? undefined : 'true'}
	focusable="false">{@html d}</svg
>

<style>
	.icon {
		flex: none;
		display: block;
	}
</style>
