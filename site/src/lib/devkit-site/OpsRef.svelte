<!--
  /devkit/reference/ops/: every core op, generated at build time from schema/ops.json and the op definitions (CORE_OPS):
  label, category, zoom view, formula (MathML; ⟨name⟩ is the slot the view fills with a live number), the auto lesson's
  narration templates in English and Chinese, attributes, weights, parity bar, the PyTorch line, and a live demo drawn
  by the op's own view on a tiny made-up model. An index by category comes first; each op is a section with its facts
  beside its demo (stacked on narrow screens).
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
	const catOf = (id: string) => page.categories.find((c) => c.id === id)?.label ?? { en: id, zh: id };
</script>

<div class="ref">
	<header class="head">
		<p class="eyebrow"><a href="{base}/devkit/reference/">{t({ en: 'Reference', zh: '参考' })}</a></p>
		<h1>{t({ en: 'Op reference', zh: '算子参考' })}</h1>
		<p class="lede">
			{t({
				en: `The ${page.ops.length} core ops of the kit, generated from schema/ops.json and the op definitions when the site is built. Each has the formula a zoom view fills with live numbers, the auto lesson's narration templates in both languages, the attributes the Python builder is checked against (AK106), the parity bar, the PyTorch equivalent, and a demo drawn by the op's own view.`,
				zh: `工具包的 ${page.ops.length} 个核心算子，在网站构建时由 schema/ops.json 和算子定义生成。每个算子都有：放大视图用实时数字填写的公式、自动课程的中英文旁白模板、Python 构建器要遵守的属性（AK106）、一致性阈值、对应的 PyTorch 写法，以及用该算子自己的视图画出的演示。`
			})}
		</p>
		<p class="small">
			{t({ en: 'The same table in the guide:', zh: '指南里的同一张表：' })} <a href={withBase(page.guide, base)}>{t({ en: 'Op reference (guide)', zh: '算子参考（指南）' })}</a>{t({ en: '.', zh: '。' })}
			{t({ en: 'Demos use made-up weights; every number in them is computed by the real kit ops.', zh: '演示用的是随意设定的权重，但其中每个数都由真实的工具包算子算出。' })}
		</p>
	</header>

	<nav class="index" aria-label={t({ en: 'Ops by category', zh: '按类别列出的算子' })}>
		{#each byCat as c (c.id)}
			<div class="cat">
				<p class="cl">{t(c.label)}</p>
				<ul>
					{#each c.ops as o (o.op)}<li><a href="#op-{o.op}"><code>{o.op}</code></a></li>{/each}
				</ul>
			</div>
		{/each}
	</nav>

	{#each page.ops as o (o.op)}
		<section class="op" id="op-{o.op}" aria-labelledby="op-{o.op}-h">
			<header class="oh">
				<h2 id="op-{o.op}-h"><code>{o.op}</code> <span class="lab">{t(o.label)}</span></h2>
				<p class="chips">
					<span class="chip">{t(catOf(o.category))}</span>
					<a class="chip" href="{base}/devkit/reference/views/#view-{o.view.replace(/[^\w-]/g, '-')}">{t({ en: 'view', zh: '视图' })} <code>{o.view}</code></a>
					<span class="chip">{t({ en: 'parity', zh: '一致性' })} ≤ <span class="num">{bar(o.parityBar)}</span></span>
				</p>
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
							<div class="row">
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
						<div class="row"><dt>{t({ en: 'Weights', zh: '权重' })}</dt><dd>{#if o.weights.length}{#each o.weights as w, i (w)}{i ? ', ' : ''}<code>{w}</code>{/each}{#if o.weights.some((w) => w.endsWith('?'))}{' '}<span class="muted">{t({ en: '(? optional)', zh: '（? 表示可选）' })}</span>{/if}{:else}—{/if}</dd></div>
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
		</section>
	{/each}
</div>

<style>
	.ref {
		container-type: inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s6);
	}
	.head .eyebrow {
		margin: 0 0 var(--s2);
	}
	.eyebrow a {
		color: inherit;
	}
	.head h1 {
		margin-bottom: var(--s3);
	}
	.small {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
		max-width: 75ch;
	}

	/* the index: categories in columns */
	.index {
		columns: 3 180px;
		column-gap: var(--s5);
		padding: var(--s4) var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.cat {
		break-inside: avoid;
		padding-bottom: var(--s3);
	}
	.cl {
		margin: 0 0 2px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--ink);
	}
	.index ul {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-wrap: wrap;
		gap: 2px var(--s3);
		font-size: var(--t-sm);
		line-height: 26px;
	}

	/* one op */
	.op {
		display: grid;
		gap: var(--s4);
		min-width: 0;
		padding-top: var(--s6);
		border-top: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.oh {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: var(--s2) var(--s4);
	}
	.oh h2 {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: var(--s1) var(--s3);
	}
	.lab {
		font-weight: var(--w-body);
		font-size: var(--t-md);
		color: var(--ink-2);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s2);
		margin: 0;
	}
	.chips a.chip {
		min-height: 0;
		padding: 1px 10px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-body);
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@container (min-width: 820px) {
		.grid {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
		}
	}
	.facts {
		display: grid;
		gap: var(--s4);
		min-width: 0;
	}
	.f {
		min-width: 0;
	}
	.k {
		margin: 0 0 var(--s1);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
	.math {
		overflow-x: auto;
		padding: var(--s2) var(--s3);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		font-size: var(--t-lg);
	}
	.math :global(math) {
		margin: 0;
	}
	.hint {
		margin: var(--s1) 0 0;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.py {
		margin: 0;
		padding: var(--s2) var(--s3);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.dl {
		margin: 0;
		display: grid;
		gap: var(--s2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.row {
		display: grid;
		grid-template-columns: 7.5em minmax(0, 1fr);
		gap: var(--s2);
	}
	.row dt {
		color: var(--muted);
	}
	.row dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	@container (max-width: 480px) {
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
		border-radius: var(--r3);
		background: var(--panel);
		padding: 0 var(--s4);
	}
	.narr summary {
		display: flex;
		align-items: center;
		min-height: 44px;
		cursor: pointer;
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
	}
	.narr[open] {
		padding-bottom: var(--s3);
	}
	.nt {
		display: grid;
		gap: var(--s3);
		margin-top: var(--s1);
	}
	.nrow {
		display: grid;
		grid-template-columns: 7.5em minmax(0, 1fr) minmax(0, 1fr);
		gap: var(--s1) var(--s4);
		padding-top: var(--s2);
		border-top: 1px solid var(--rule);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.nrow p {
		margin: 0;
	}
	.nk .muted {
		display: block;
		font-size: var(--t-xs);
	}
	@container (max-width: 700px) {
		.nrow {
			grid-template-columns: minmax(0, 1fr);
		}
		.nk .muted {
			display: inline;
		}
	}
	.nz {
		line-height: 1.8;
	}
	.ph {
		padding: 0 3px;
		border-radius: var(--r1);
		background: var(--here-bg);
		font-family: var(--font-mono);
		font-size-adjust: var(--mono-adjust);
	}
</style>
