<!--
  Input dock: once the input section has scrolled away, a small floating panel keeps the input in
  view (what the model sees + its answer) and, on wide screens, a mini pad bound to the same
  strokes, so you can keep drawing while watching the layers. Collapsible; the choice is
  remembered in localStorage (per viewer convenience only). Hidden while the guided lesson is on
  screen (the lesson has its own small copy of the input) and once the article starts.
-->
<script lang="ts">
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();

	let past = $state(false);
	/** closed with ✕: stays hidden for this page view */
	let closed = $state(false);
	let wide = $state(false);
	// collapsed by default (thumbnail + answer); expanding shows a mini pad on wide screens
	let small = $state(true);
	const KEY = 'atlas.tinycnn.dock';

	$effect(() => {
		try {
			small = localStorage.getItem(KEY) !== 'big';
		} catch {
			/* storage blocked */
		}
		const mq = matchMedia('(min-width: 1100px)');
		wide = mq.matches;
		const onmq = (e: MediaQueryListEvent) => (wide = e.matches);
		mq.addEventListener('change', onmq);
		// shown between the end of the input section and the start of the article, except while the
		// lesson is on screen
		let raf = 0;
		const check = () => {
			raf = 0;
			const inp = document.getElementById('input');
			const art = document.getElementById('article');
			const les = document.getElementById('lesson');
			const pastInput = !!inp && inp.getBoundingClientRect().bottom < 0;
			const atArticle = !!art && art.getBoundingClientRect().top < innerHeight * 0.6;
			const lr = les?.getBoundingClientRect();
			const inLesson = !!lr && lr.top < innerHeight && lr.bottom > 80;
			past = pastInput && !atArticle && !inLesson;
		};
		const onscroll = () => {
			if (!raf) raf = requestAnimationFrame(check);
		};
		addEventListener('scroll', onscroll, { passive: true });
		addEventListener('resize', onscroll, { passive: true });
		check();
		return () => {
			removeEventListener('scroll', onscroll);
			removeEventListener('resize', onscroll);
			if (raf) cancelAnimationFrame(raf);
			mq.removeEventListener('change', onmq);
		};
	});
	function toggle() {
		small = !small;
		try {
			localStorage.setItem(KEY, small ? 'small' : 'big');
		} catch {
			/* ignore */
		}
	}
	function toInput() {
		document.getElementById('input')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}
</script>

{#if past && S.trace && !closed}
	<aside class="dock" class:small aria-label={t({ en: 'Your input (shortcut)', zh: '你的输入（快捷面板）' })}>
		<div class="top">
			<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={small ? 30 : 56} alt={t({ en: 'What the model sees', zh: '模型看到的' })} />
			<div class="ans">
				<span class="cls"><span class="sw" style="background: {categorical(S.top, themeState.resolved)}"></span>{t(S.spec.classes[S.top])}</span>
				<span class="num p">{fmtPct(S.probs?.[S.top] ?? 0)}</span>
			</div>
			<div class="acts">
				<button type="button" class="btn ghost ic" onclick={toInput} aria-label={t({ en: 'Back to the input section', zh: '回到输入部分' })} title={t({ en: 'Back to the input', zh: '回到输入' })}>↑</button>
				<button type="button" class="btn ghost ic" onclick={() => (closed = true)} aria-label={t({ en: 'Close this shortcut panel', zh: '关闭这个快捷面板' })} title={t({ en: 'Close', zh: '关闭' })}>✕</button>
				{#if wide}
					<button type="button" class="btn ghost ic" onclick={toggle} aria-expanded={!small} aria-label={small ? t({ en: 'Show the mini pad', zh: '显示小画板' }) : t({ en: 'Hide the mini pad', zh: '隐藏小画板' })}>{small ? '✎' : '–'}</button>
				{/if}
			</div>
		</div>
		{#if wide && !small}
			<DrawPad
				bind:strokes={S.strokes}
				brush={0.055}
				size={168}
				showBrush={false}
				onchange={(s) => S.setStrokes(s)}
				ondraw={(s) => S.drawing(s)}
				label={t({ en: 'Mini drawing pad', zh: '小画板' })}
			/>
		{/if}
	</aside>
{/if}

<style>
	.dock {
		position: fixed;
		z-index: 30;
		right: 12px;
		bottom: 12px;
		display: grid;
		gap: 6px;
		padding: 8px 10px;
		width: 196px;
		background: var(--surface);
		border: 1px solid var(--rule-strong);
		border-radius: var(--r3);
		box-shadow: var(--shadow-2);
	}
	.dock.small {
		width: auto;
		padding: 4px 6px;
		gap: 4px;
	}
	@media (max-width: 1099px) {
		.dock {
			width: auto;
			left: auto;
			right: 8px;
			bottom: 8px;
			padding: 3px 5px;
		}
	}
	.dock.small .ic {
		width: 26px;
		min-height: 26px;
	}
	.top {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.ans {
		display: grid;
		font-size: var(--fs-xs);
		line-height: 1.3;
		min-width: 0;
		flex: 1;
	}
	.cls {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-weight: 700;
		white-space: nowrap;
	}
	.sw {
		width: 8px;
		height: 8px;
		border-radius: 2px;
	}
	.p {
		color: var(--ink-2);
	}
	.acts {
		display: flex;
		gap: 2px;
	}
	.ic {
		width: 30px;
		min-height: 30px;
		padding: 0;
	}
	.dock :global(.tools .btn) {
		min-height: 30px;
		padding: 0 8px;
		font-size: var(--fs-xs);
	}
</style>
