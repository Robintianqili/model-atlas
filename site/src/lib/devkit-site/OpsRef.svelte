<!--
  /devkit/reference/ops/: every core op, generated at build time from schema/ops.json and the op definitions (CORE_OPS):
  label, category, zoom view, formula (MathML; ⟨name⟩ is the slot the view fills with a live number), the auto lesson's
  narration templates in English and Chinese, attributes, weights, parity bar, the PyTorch line, and a live demo drawn
  by the op's own view on a tiny made-up model.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import Demo from './Demo.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { withBase } from './base.ts';
	import type { OpsData } from './types.ts';

	let { page }: { page: OpsData } = $props();
	const bar = (v: number) => v.toExponential(0).replace('e-', 'e−');
	/** a narration template with its {placeholders} marked */
	const parts = (s: string) => s.split(/(\{[^{}]+\})/g).map((x, i) => ({ x, ph: i % 2 === 1 }));
	/** an attribute's doc with its `code` spans (ops.json docs use Markdown backticks) */
	const docParts = (s: string) => s.split(/`([^`]*)`/g).map((x, i) => ({ x, code: i % 2 === 1 }));
	const NAMES: Record<string, { en: string; zh: string }> = {
		what: { en: 'what', zh: '是什么' },
		how: { en: 'how', zh: '怎么算' },
		why: { en: 'why', zh: '为什么' },
		look: { en: 'look at', zh: '看哪里' },
		unit: { en: 'one unit', zh: '一个单元' },
		repeat: { en: 'repeat', zh: '重复出现' }
	};
	const byCat = $derived(page.categories.map((c) => ({ ...c, ops: page.ops.filter((o) => o.category === c.id) })));
</script>

<div class="container ref">
	<header class="head">
		<p class="eyebrow">DevKit · {t({ en: 'Reference', zh: '参考' })}</p>
		<h1>{t({ en: 'Op reference', zh: '算子参考' })}</h1>
		<p class="lede">
			{t({
				en: `The ${page.ops.length} core ops of the kit, generated from schema/ops.json and the op definitions when the site is built. Each card has the formula a zoom view fills with live numbers, the auto lesson's narration templates in both languages, the attributes the Python builder is checked against (AK106), the parity bar, the PyTorch equivalent, and a demo drawn by the op's own view.`,
				zh: `工具包的 ${page.ops.length} 个核心算子，在网站构建时由 schema/ops.json 和算子定义生成。每张卡片有：放大视图用实时数字填写的公式、自动课程的中英文旁白模板、Python 构建器要遵守的属性（AK106）、一致性阈值、对应的 PyTorch 写法，以及用该算子自己的视图画出的演示。`
			})}
		</p>
		<p class="muted small">
			{t({ en: 'The same table in the guide:', zh: '指南里的同一张表：' })} <a href={withBase(page.guide, base)}>{t({ en: 'Op reference (guide)', zh: '算子参考（指南）' })}</a>
			· {t({ en: 'Demos use made-up weights; every number in them is computed by the real kit ops.', zh: '演示用的是随意设定的权重，但其中每个数都由真实的工具包算子算出。' })}
		</p>
		<nav class="jump" aria-label={t({ en: 'Ops by category', zh: '按类别列出的算子' })}>
			{#each byCat as c (c.id)}
				<span class="cat">{t(c.label)}:</span>
				{#each c.ops as o (o.op)}<a href="#op-{o.op}"><code>{o.op}</code></a>{/each}
			{/each}
		</nav>
	</header>

	{#each page.ops as o (o.op)}
		<article class="op" id="op-{o.op}" aria-labelledby="op-{o.op}-h">
			<header class="oh">
				<h2 id="op-{o.op}-h"><code>{o.op}</code> <span class="lab">{t(o.label)}</span></h2>
				<div class="chips">
					<span class="chip">{t(page.categories.find((c) => c.id === o.category)?.label ?? { en: o.category, zh: o.category })}</span>
					<a class="chip" href="{base}/devkit/reference/views/#view-{o.view.replace(/[^\w-]/g, '-')}"><code>{o.view}</code></a>
					<span class="chip">{t({ en: 'parity', zh: '一致性' })} ≤ {bar(o.parityBar)}</span>
				</div>
			</header>
			<div class="grid">
				<div class="facts">
					{#if o.formula}
						<div class="f">
							<p class="k">{t({ en: 'Formula', zh: '公式' })}</p>
							<div class="math" lang="en" use:keyScroll>{@html o.formula}</div>
							<p class="hint">{t({ en: 'In a lesson, each ⟨name⟩ shows the live number of the selected unit (see the demo).', zh: '在课程中，每个 ⟨名字⟩ 显示所选单元的实时数字（见演示）。' })}</p>
						</div>
					{/if}
					{#if o.pytorch}
						<div class="f">
							<p class="k">PyTorch</p>
							<pre class="py" use:keyScroll><code>{o.pytorch}</code></pre>
						</div>
					{/if}
					<dl class="dl">
						{#if o.attrs.length}
							<div class="row wide">
								<dt>{t({ en: 'Attributes', zh: '属性' })}</dt>
								<dd>
									<ul class="attrs">
										{#each o.attrs as a (a.name)}
											<li><code>{a.name}</code>: {a.type}{#if a.required}{' '}<strong>{t({ en: '(required)', zh: '（必需）' })}</strong>{:else if a.def !== null}{' = '}<code>{a.def}</code>{/if}{#if a.doc}{' '}<span class="muted" lang="en">· {#each docParts(a.doc) as d, k (k)}{#if d.code}<code>{d.x}</code>{:else}{d.x}{/if}{/each}</span>{/if}</li>
										{/each}
									</ul>
								</dd>
							</div>
						{/if}
						<div class="row"><dt>{t({ en: 'Weights', zh: '权重' })}</dt><dd>{#if o.weights.length}{#each o.weights as w, i (w)}{i ? ', ' : ''}<code>{w}</code>{/each}{#if o.weights.some((w) => w.endsWith('?'))}{' '}<span class="muted small">{t({ en: '(? optional)', zh: '（? 表示可选）' })}</span>{/if}{:else}—{/if}</dd></div>
						{#if o.subs.length}<div class="row"><dt>{t({ en: 'Sub-steps', zh: '子步骤' })}</dt><dd>{#each o.subs as s, i (s)}{i ? ', ' : ''}<code>{o.op}#{s}</code>{/each}</dd></div>{/if}
						{#if o.aliases.length}<div class="row"><dt>{t({ en: 'Aliases', zh: '别名' })}</dt><dd>{#each o.aliases as a, i (a)}{i ? ', ' : ''}<code>{a}</code>{/each}</dd></div>{/if}
						<div class="row"><dt>{t({ en: 'Demo layer', zh: '演示层' })}</dt><dd><code>{o.demo.model}/{o.demo.layer}</code>{#if o.demo.unit}{' · '}{t({ en: 'unit', zh: '单元' })}{' '}<code>[{o.demo.unit.join(', ')}]</code>{/if}</dd></div>
					</dl>
				</div>
				<Demo op={o.op} key={o.view} minHeight={260} />
			</div>
			<details class="narr">
				<summary>{t({ en: `Narration templates of the auto lesson (${o.narration.length}, English and Chinese)`, zh: `自动课程的旁白模板（${o.narration.length} 条，中英文）` })}</summary>
				<div class="nt">
					{#each o.narration as n (n.key)}
						<div class="nrow">
							<p class="nk"><code>{n.key}</code> <span class="muted">{t(NAMES[n.key] ?? { en: n.key, zh: n.key })}</span></p>
							<p class="ne" lang="en">{#each parts(n.en) as p, i (i)}{#if p.ph}<span class="ph">{p.x}</span>{:else}{p.x}{/if}{/each}</p>
							<p class="nz" lang="zh-CN">{#each parts(n.zh) as p, i (i)}{#if p.ph}<span class="ph">{p.x}</span>{:else}{p.x}{/if}{/each}</p>
						</div>
					{/each}
				</div>
				<p class="hint">{t({ en: '{placeholders} are filled from the live run; {name|fmt} picks the number format (guide section 8.3).', zh: '{占位符} 由实时运行的结果填写；{name|fmt} 选择数字格式（指南 8.3 节）。' })}</p>
			</details>
		</article>
	{/each}
</div>

<style>
	.ref {
		padding-top: var(--s6);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.small {
		font-size: var(--fs-sm);
	}
	.jump {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		margin-top: var(--s3);
		font-size: var(--fs-sm);
	}
	.jump .cat {
		color: var(--muted);
		margin-left: 6px;
	}
	.jump code {
		font-size: 0.85em;
	}
	.op {
		padding-top: var(--s5);
		border-top: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
		display: grid;
		gap: var(--s3);
		min-width: 0;
	}
	.oh {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 8px 16px;
	}
	.oh h2 {
		margin: 0;
		font-size: var(--fs-xl);
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
	}
	.oh h2 code {
		font-size: 0.85em;
	}
	.lab {
		font-family: var(--font-body);
		font-weight: 400;
		font-size: var(--fs-md);
		color: var(--ink-2);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.chips a.chip {
		text-decoration: none;
	}
	.chips code {
		font-size: 0.95em;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (max-width: 900px) {
		.grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.facts {
		display: grid;
		gap: var(--s3);
		min-width: 0;
	}
	.f {
		min-width: 0;
	}
	.k {
		margin: 0 0 4px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.math {
		overflow-x: auto;
		padding: 8px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		font-size: 1.12em;
	}
	.math :global(math) {
		margin: 0;
	}
	.hint {
		margin: 4px 0 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.py {
		margin: 0;
		padding: 8px 10px;
		font-size: 0.8rem;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.dl {
		margin: 0;
		display: grid;
		gap: 6px;
		font-size: var(--fs-sm);
	}
	.row {
		display: grid;
		grid-template-columns: 9em minmax(0, 1fr);
		gap: 8px;
	}
	.row dt {
		color: var(--muted);
	}
	.row dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	@media (max-width: 560px) {
		.row {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
		}
	}
	.attrs {
		margin: 0;
		padding-left: 1.1em;
	}
	.attrs li + li {
		margin-top: 2px;
	}
	.narr {
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		padding: 8px 12px;
	}
	.narr summary {
		cursor: pointer;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.nt {
		display: grid;
		gap: 10px;
		margin-top: 10px;
	}
	.nrow {
		display: grid;
		grid-template-columns: 8em minmax(0, 1fr) minmax(0, 1fr);
		gap: 6px 14px;
		padding-top: 8px;
		border-top: 1px solid var(--rule);
		font-size: var(--fs-sm);
	}
	.nrow p {
		margin: 0;
	}
	.nk code {
		font-size: 0.9em;
	}
	.nk .muted {
		display: block;
		font-size: var(--fs-xs);
	}
	@media (max-width: 760px) {
		.nrow {
			grid-template-columns: minmax(0, 1fr);
		}
		.nk .muted {
			display: inline;
		}
	}
	.ph {
		padding: 0 2px;
		border-radius: 3px;
		background: var(--accent-soft);
		font-family: var(--font-mono);
		font-size: 0.88em;
	}
</style>
