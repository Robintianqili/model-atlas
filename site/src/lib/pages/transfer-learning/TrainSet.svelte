<!--
  The two training classes as columns of 28×28 thumbnails (exactly what the network sees).
  Hover/focus links to the PCA map; click inspects the drawing in the network view.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import Thumb from './Thumb.svelte';
	import type { SetItem } from './data';

	interface Props {
		items: SetItem[];
		names: [string, string];
		inspect: string;
		hover?: string | null;
		onselect: (id: string) => void;
		onremove: (id: string) => void;
	}
	let { items, names, inspect, hover = $bindable(null), onselect, onremove }: Props = $props();

	const cols = $derived([items.filter((e) => e.cls === 0), items.filter((e) => e.cls === 1)]);
	const wrong = (e: SetItem) => e.pA !== null && (e.cls === 0 ? e.pA < 0.5 : e.pA >= 0.5);
	const desc = (e: SetItem, i: number) => {
		const who = e.drawn ? t({ en: 'your drawing', zh: '你画的' }) : t({ en: 'preset drawing', zh: '预置图' });
		const cls = names[e.cls];
		const p =
			e.pA === null
				? ''
				: t({
						en: `; head says ${names[0]} ${(e.pA * 100).toFixed(0)}%`,
						zh: `；输出头判为${names[0]}的概率 ${(e.pA * 100).toFixed(0)}%`
					});
		const fl = e.flipped ? t({ en: ' (wrong label on purpose)', zh: '（故意标错）' }) : '';
		return `${cls} ${i + 1}, ${who}${fl}${p}`;
	};
</script>

<div class="set">
	{#each cols as col, k (k)}
		<div class="col" class:a={k === 0} class:b={k === 1}>
			<p class="ch">
				<span class="key" aria-hidden="true"></span>
				<strong>{names[k]}</strong>
				<span class="muted num">{t({ en: `${col.length} example${col.length === 1 ? '' : 's'}`, zh: `${col.length} 个样本` })}</span>
			</p>
			{#if col.length}
				<ul class="thumbs">
					{#each col as e, i (e.id)}
						<li class="item">
							<button
								type="button"
								class="th"
								class:on={inspect === e.id}
								class:hov={hover === e.id}
								class:wrong={wrong(e)}
								class:flipped={e.flipped}
								aria-label={desc(e, i)}
								title={desc(e, i)}
								aria-pressed={inspect === e.id}
								onclick={() => onselect(e.id)}
								onpointerenter={() => (hover = e.id)}
								onpointerleave={() => (hover = null)}
								onfocus={() => (hover = e.id)}
								onblur={() => (hover = null)}
							>
								<Thumb data={e.input} size={38} alt="" />
								{#if e.drawn}<span class="badge" aria-hidden="true">✎</span>{/if}
								{#if e.flipped}<span class="badge fl" aria-hidden="true">!</span>{/if}
							</button>
							{#if e.drawn}
								<button type="button" class="rm" aria-label={t({ en: `Remove ${desc(e, i)}`, zh: `删除：${desc(e, i)}` })} onclick={() => onremove(e.id)}>×</button>
							{/if}
						</li>
					{/each}
				</ul>
			{:else}
				<p class="empty muted">{t({ en: 'No examples yet. Draw one below and add it here.', zh: '还没有样本。在下面画一张并加到这里。' })}</p>
			{/if}
		</div>
	{/each}
</div>
<p class="hint muted">
	{t({
		en: 'Thumbnails show exactly what the network receives (28 × 28). Click one to inspect it in the network view. ✎ = your drawing. A dashed frame means the head currently gets it wrong.',
		zh: '缩略图就是网络实际收到的 28 × 28 输入。点一张，可以在网络视图里查看它。✎ 表示你画的。虚线框表示输出头目前判错了它。'
	})}
</p>

<style>
	.set {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 10px;
	}
	.col {
		padding: 8px 10px 10px;
		border-radius: var(--r2);
		background: var(--surface);
		border: 1px solid var(--rule);
		border-top: 4px solid var(--c);
		min-width: 0;
	}
	.col.a {
		--c: var(--div-pos);
	}
	.col.b {
		--c: var(--div-neg);
	}
	.ch {
		margin: 0 0 6px;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		font-size: var(--fs-sm);
	}
	.key {
		width: 10px;
		height: 10px;
		background: var(--c);
		border-radius: 50%;
	}
	.b .key {
		border-radius: 2px;
	}
	.thumbs {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 5px;
	}
	.item {
		position: relative;
	}
	.th {
		position: relative;
		display: block;
		padding: 2px;
		border: 2px solid var(--rule);
		border-radius: 5px;
		background: var(--surface);
		cursor: pointer;
	}
	.th:hover,
	.th.hov {
		border-color: var(--ink-2);
	}
	.th.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.th.wrong {
		border-style: dashed;
		border-color: var(--ink);
	}
	.badge {
		position: absolute;
		left: -4px;
		top: -6px;
		font-size: var(--t-xs);
		line-height: 1;
		padding: 1px 3px;
		border-radius: 4px;
		background: var(--surface-3);
		color: var(--ink);
	}
	.badge.fl {
		left: auto;
		right: -4px;
		background: var(--illus-soft);
		color: var(--illus);
		font-weight: 800;
	}
	.rm {
		position: absolute;
		right: -6px;
		bottom: -6px;
		width: 20px;
		height: 20px;
		padding: 0;
		border-radius: 50%;
		border: 1px solid var(--rule-strong);
		background: var(--surface);
		color: var(--ink-2);
		font-size: 13px;
		line-height: 1;
		cursor: pointer;
	}
	.rm:hover {
		color: var(--ink);
		border-color: var(--ink-2);
	}
	.empty {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.hint {
		margin: 6px 0 0;
		font-size: var(--fs-xs);
	}
</style>
