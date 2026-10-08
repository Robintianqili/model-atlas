<!--
  Lesson step 2: the text cut into pre-tokens by TinyGPT's regular expression (the same one the
  training uses). Left/top: the pre-tokens in order (spaces drawn as ␣). Right/below: the different
  pre-tokens by frequency; hover, focus or tap one to find it in the text.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { fmtInt, visibleToken } from '$lib/kit/format';
	import { pretokenStats } from '../lesson';

	let { text, pattern }: { text: string; pattern: string } = $props();

	const SHOW = 240;
	const TOP = 8;
	const ps = $derived(pretokenStats(text, pattern));
	let pick = $state<string | null>(null);
	let hov = $state<string | null>(null);
	// a new text starts without a pick
	$effect(() => {
		void text;
		pick = null;
	});
	const on = $derived(hov ?? pick ?? ps.types[0]?.text ?? null);
	const onType = $derived(ps.types.find((x) => x.text === on) ?? null);
	const maxN = $derived(ps.types[0]?.n ?? 1);
	const isZh = (s: string) => /[一-鿿]/.test(s);
</script>

<div class="ls">
	<h4>{t({ en: 'Your text, cut into pre-tokens', zh: '你的文本被切成预分词片段' })}</h4>
	{#if ps.list.length}
		<p class="flow" aria-live="polite">
			<span><b class="num big">{fmtInt(ps.list.length)}</b> {t({ en: 'pre-tokens', zh: '个预分词片段' })}</span>
			<span>{t({ en: 'of which', zh: '其中' })} <b class="num">{fmtInt(ps.types.length)}</b> {t({ en: 'are different', zh: '种不同' })}</span>
		</p>
		<div class="grid">
			<div class="text" role="list" use:keyScroll aria-label={t({ en: 'Pre-tokens in order', zh: '按顺序排列的预分词片段' })}>
				{#each ps.list.slice(0, SHOW) as p, i (i)}
					<span class="pt" class:on={p === on} role="listitem" lang={isZh(p) ? 'zh-CN' : undefined}>{visibleToken(p)}</span>
				{/each}
				{#if ps.list.length > SHOW}
					<span class="muted more">… {t({ en: `first ${SHOW} of ${fmtInt(ps.list.length)} shown`, zh: `只显示前 ${SHOW} 个（共 ${fmtInt(ps.list.length)} 个）` })}</span>
				{/if}
			</div>
			<div class="freq">
				<p class="sh">{t({ en: 'Most frequent pre-tokens', zh: '出现最多的预分词片段' })}</p>
				<ol>
					{#each ps.types.slice(0, TOP) as ty (ty.text)}
						<li>
							<button
								type="button"
								class="fr"
								aria-pressed={pick === ty.text}
								class:on={ty.text === on}
								onpointerenter={() => (hov = ty.text)}
								onpointerleave={() => (hov = null)}
								onfocus={() => (hov = ty.text)}
								onblur={() => (hov = null)}
								onclick={() => (pick = pick === ty.text ? null : ty.text)}
							>
								<span class="pt small" lang={isZh(ty.text) ? 'zh-CN' : undefined}>{visibleToken(ty.text)}</span>
								<span class="track"><span class="fill" style:width="{(100 * ty.n) / maxN}%"></span></span>
								<span class="num n">×{ty.n}</span>
							</button>
						</li>
					{/each}
				</ol>
				{#if onType}
					<p class="hint">
						<span class="pt small" lang={isZh(onType.text) ? 'zh-CN' : undefined}>{visibleToken(onType.text)}</span>
						{t({ en: 'occurs', zh: '出现了' })} <b class="num">{onType.n}</b> {t({ en: 'times and is', zh: '次，占' })} <b class="num">{onType.bytes}</b> {onType.bytes === 1 ? t({ en: 'byte', zh: '个字节' }) : t({ en: 'bytes', zh: '个字节' })}{t({ en: '.', zh: '。' })}
						{#if onType.bytes > 1}
							{t({ en: 'In the next step, each pair of neighbouring bytes inside it counts', zh: '下一步统计相邻对时，它里面每一对相邻的字节都计' })} <b class="num">{onType.n}</b> {t({ en: 'times.', zh: '次。' })}
						{:else}
							{t({ en: 'A single byte has no pair inside, so it never changes.', zh: '单个字节内部没有相邻对，所以它永远不会改变。' })}
						{/if}
					</p>
				{/if}
			</div>
		</div>
		<p class="note">{t({ en: 'Each box is one pre-token. Merges happen only inside a box, never across two boxes.', zh: '每个方框是一个预分词片段。合并只发生在方框内部，绝不会跨越两个方框。' })}</p>
	{:else}
		<p class="muted">{t({ en: 'Your text is empty. Pick a preset or type something under “Your text”.', zh: '你的文本是空的。在“你的文本”里选一个预设，或者输入一些内容。' })}</p>
	{/if}
</div>

<style>
	.ls {
		container: ls / inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.flow {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
		color: var(--ink-2);
	}
	.big {
		font-size: var(--fs-xl);
		color: var(--ink);
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	@container ls (min-width: 600px) {
		.grid {
			grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
		}
	}
	.text {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 3px;
		align-content: flex-start;
		max-height: 260px;
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: 6px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.pt {
		display: inline-block;
		max-width: 100%;
		padding: 1px 5px;
		border: 1px solid var(--rule-strong);
		border-radius: 4px;
		background: var(--surface-2);
		font-family: var(--font-mono);
		font-size: 0.88rem;
		line-height: 1.45;
		overflow-wrap: anywhere;
		color: var(--ink);
	}
	.pt.on {
		border-color: var(--accent);
		background: var(--accent-soft);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.pt.small {
		font-size: 0.8rem;
		padding: 0 4px;
	}
	.more {
		font-size: var(--fs-xs);
		align-self: center;
	}
	.freq {
		display: grid;
		gap: 6px;
		align-content: start;
		min-width: 0;
	}
	.sh {
		margin: 0;
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	.fr {
		width: 100%;
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(30px, 1fr) 3em;
		align-items: center;
		gap: 8px;
		padding: 3px 6px;
		border: 1px solid transparent;
		border-radius: var(--r1);
		background: none;
		color: inherit;
		text-align: left;
		cursor: pointer;
	}
	.fr:hover,
	.fr.on {
		background: var(--surface-2);
	}
	.fr[aria-pressed='true'] {
		border-color: var(--accent);
	}
	.fr .pt {
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.track {
		height: 9px;
		border-radius: 3px;
		background: var(--surface-2);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		background: var(--accent);
	}
	.n {
		text-align: right;
		font-size: var(--fs-sm);
	}
	.hint,
	.note {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
</style>
