<!-- sasrec 'seq': your history and what the model does with it (props.mode; meta.ts lists the modes). Weights and
     shares use the kit's sequential ramp, signed numbers its diverging scale; a bar's length is its value. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { ITEMS, NEW_ITEM, catColor, type Clicks } from '../../world.ts';
	import { K, T, attn, clicksOf, logClicks, most, popTop, ranked, reversed, row, newPlace } from '../../seq.ts';
	import Shop from '../../parts/Shop.svelte';
	import Cells from '../../parts/Cells.svelte';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'pick'));
	const c = $derived(clicksOf(P.rt));
	const n = $derived(c.length);
	const seqc = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const a = $derived(attn(P.rt));
	const top = $derived(most(P.rt));
	const W = $derived(T(P.rt, 'att.w'));
	const r = $derived(ranked(P.rt));
	const rv = $derived(mode === 'order' ? reversed(P.rt) : null);
	const np = $derived(mode === 'limits' ? newPlace(P.rt) : null);
	const emb = $derived(['item', 'pos', 'x'].map((id) => T(P.rt, id)));
	const max = $derived(Math.max(1e-9, ...emb.flatMap((e) => (e ? Array.from(e.data, Math.abs) : []))));
	const pmax = $derived(Math.max(1e-9, ...r.slice(0, K + 3).map((x) => x.p), ...(rv?.top ?? []).map((x) => x.p)));
	const cnt = (v: number) => (Number.isFinite(v) ? v.toLocaleString('en-US') : '…');
	const fact = (k: string) => P.rt.num(`fact:${k}`);
	const name = (i: number) => t(ITEMS[i].name);
	const setInput = (x: Clicks) => (P.setInput ? P.setInput(x) : P.rt.setInput(x));
</script>

{#snippet dot(i: number)}<i class="d" style:background={catColor(ITEMS[i].cat)}></i>{/snippet}
{#snippet list(xs: { i: number; p: number }[], cut = K)}
	<ol class="ls">
		{#each xs as x, k (x.i)}
			{#if k === cut}<li class="cut">{t({ en: `— the top ${K} end here —`, zh: `—— 前 ${K} 名到此为止 ——` })}</li>{/if}
			<li class="r" class:in={k < cut}><span class="num k">{k + 1}</span>{@render dot(x.i)}<span class="ti">{name(x.i)}</span>
				<span class="bar"><i style:width="{(100 * x.p) / pmax}%" style:background={seqc.css(0.75)}></i></span><b class="num">{fmtPct(x.p)}</b></li>
		{/each}
	</ol>
{/snippet}

<div class="sv">
	{#if mode === 'pick'}
		<Shop raw={P.rt.input} onchange={setInput} />
	{:else if !n}
		<p class="muted">{t({ en: 'No clicks yet: add some in step 1 or in the shop below the lesson.', zh: '还没有点击：在第 1 步或课程下方的商店里点几件。' })}</p>
	{:else if mode === 'embed'}
		<p class="hd">{t({ en: 'Each click: its product’s row + its position’s row = the row the block reads (16 colors each; hover for the numbers)', zh: '每次点击：商品的行 + 位置的行 = 注意力块读到的行（每行 16 个颜色；悬停可看数值）' })}</p>
		<div class="sc" use:keyScroll><table class="tb emb">
			<thead><tr><th>{t({ en: 'click', zh: '点击' })}</th><th>{t({ en: 'product row', zh: '商品行' })}</th><th></th><th>{t({ en: 'position row', zh: '位置行' })}</th><th></th><th>{t({ en: 'sum', zh: '和' })}</th></tr></thead>
			<tbody>{#each c as i, k (i)}<tr class:last={k === n - 1}><td class="nm"><span class="num">{k}</span> {@render dot(i)}{name(i)} <span class="num muted">#{i}</span></td>
				{#each emb as e, m (m)}{#if m}<td class="op">{m === 1 ? '+' : '='}</td>{/if}<td><Cells data={row(e, k) ?? []} {max} values={false} label="{name(i)}: {t([{ en: 'product row', zh: '商品行' }, { en: 'position row', zh: '位置行' }, { en: 'sum', zh: '和' }][m])}" /></td>{/each}</tr>{/each}</tbody>
		</table></div>
		<p class="small">{t({ en: `The last click in numbers, features 0 to 3 of 16: product ${(row(emb[0], n - 1) ?? []).slice(0, 4).map((v) => fmt3(v)).join(', ')} + position ${(row(emb[1], n - 1) ?? []).slice(0, 4).map((v) => fmt3(v)).join(', ')} = ${(row(emb[2], n - 1) ?? []).slice(0, 4).map((v) => fmt3(v)).join(', ')} …`, zh: `最后一次点击的数，16 个特征里的第 0 到 3 个：商品 ${(row(emb[0], n - 1) ?? []).slice(0, 4).map((v) => fmt3(v)).join('，')} + 位置 ${(row(emb[1], n - 1) ?? []).slice(0, 4).map((v) => fmt3(v)).join('，')} = ${(row(emb[2], n - 1) ?? []).slice(0, 4).map((v) => fmt3(v)).join('，')} ……` })}</p>
	{:else if mode === 'attend' && a}
		<p class="hd">{t({ en: `Where your last click, ${ITEMS[c[n - 1]].name.en}, looks: its attention weights add up to 100%`, zh: `你最后一次点击“${ITEMS[c[n - 1]].name.zh}”看向哪里：它的注意力权重加起来是 100%` })}</p>
		<ol class="ls answer">
			{#each c as i, k (i)}<li class="r" class:top={top?.j === k}><span class="num k">{k}</span>{@render dot(i)}<span class="ti">{name(i)}{k === n - 1 ? t({ en: ' (itself)', zh: '（它自己）' }) : ''}</span>
				<span class="bar"><i style:width="{100 * a[k]}%" style:background={seqc.css(a[k])}></i></span><b class="num">{fmtPct(a[k])}</b></li>{/each}
		</ol>
		<p class="small">{t({ en: 'Bars on a 0 to 100% scale; the stronger the color, the larger the weight. The outlined row gets the most.', zh: '条长从 0 到 100%；颜色越浓，权重越大。描边的一行得到的最多。' })}</p>
	{:else if mode === 'mask' && W}
		<p class="hd">{t({ en: 'Every click’s attention weights, one row each (row t guesses the click at position t + 1, counting from 0); hatched cells are masked: exactly 0', zh: '每次点击的注意力权重，一行一次（行 t 猜位置 t + 1 上的点击，从 0 数起）；斜线格被遮住：正好为 0' })}</p>
		<div class="sc" use:keyScroll><table class="tb mx">
			<thead><tr><th>{t({ en: 'reads →', zh: '读 →' })}</th>{#each c as _, j (j)}<th class="num">{j}</th>{/each}</tr></thead>
			<tbody>{#each c as i, q (i)}<tr><th class="nm"><span class="num">{q}</span> {name(i)}</th>
				{#each c as _, j (j)}{@const v = W.data[q * n + j]}{#if j > q}<td class="m" title={t({ en: 'masked', zh: '已遮住' })}></td>{:else}<td class="num w" style:background={seqc.css(v)} style:color={seqc.ink(v)}>{Math.round(100 * v)}</td>{/if}{/each}</tr>{/each}</tbody>
		</table></div>
		<p class="small">{t({ en: 'Numbers are percentages; each row adds up to 100.', zh: '数字是百分比；每一行加起来是 100。' })}</p>
	{:else if mode === 'topk'}
		<p class="hd">{t({ en: `Next-click shares of the ${r.length} products you have not clicked, best first`, zh: `你没点过的 ${r.length} 件商品作为下一次点击的份额，从高到低` })}</p>
		{@render list(r.slice(0, K + 3))}
		<p class="small">{t({ en: `Already in your history, so left out: ${c.map((i) => ITEMS[i].name.en).join(', ')}.`, zh: `已经在你的历史里，所以不列：${c.map((i) => ITEMS[i].name.zh).join('、')}。` })}</p>
	{:else if mode === 'order' && rv}
		<div class="two">
			{#each [{ h: c, x: r.slice(0, K), l: { en: 'Your order', zh: '你的顺序' } }, { h: rv.clicks, x: rv.top, l: { en: 'The same clicks, reversed', zh: '同样的点击，倒过来' } }] as s, m (m)}
				<div class="card"><p class="hd">{t(s.l)}</p><p class="small">{s.h.map((i) => name(i)).join(' → ')}</p>{@render list(s.x)}</div>
			{/each}
		</div>
		<div class="sc" use:keyScroll><table class="tb">
			<tbody>
				<tr><td>{t({ en: 'Test sessions: hidden next click in the top 5, history in its order', zh: '测试会话：被藏起来的下一次点击落在前 5 名，历史按原顺序' })}</td><td class="num">{fmtPct(fact('hit5'))}</td></tr>
				<tr><td>{t({ en: 'The same, each history shuffled', zh: '同上，每段历史都打乱顺序' })}</td><td class="num">{fmtPct(fact('hit5Shuffled'))}</td></tr>
			</tbody>
		</table></div>
		<p class="small">{t({ en: 'Compare: ', zh: '对比：' })}<a href="{base}/m/matrix-factorization/">{t({ en: 'matrix factorization (one learned row per user) →', zh: '矩阵分解（每个用户学一行）→' })}</a> · <a href="{base}/m/two-tower/">{t({ en: 'two-tower retrieval (your picks as a bag) →', zh: '双塔召回（把你的选择当成一袋）→' })}</a></p>
	{:else if mode === 'limits'}
		<div class="two">
			<div class="card"><p class="hd">{@render dot(NEW_ITEM)}{t({ en: 'A new product: the drone', zh: '新商品：无人机' })}</p>
				<p class="small">{t({ en: `clicks in the training logs: ${cnt(logClicks(P.rt, NEW_ITEM))}`, zh: `训练日志里的点击：${cnt(logClicks(P.rt, NEW_ITEM))} 次` })}</p>
				<p class="small">{np ? t({ en: `for you: place ${np.place} of ${np.of}, share ${fmtPct(np.p)}`, zh: `对你：${np.of} 件里排第 ${np.place}，份额 ${fmtPct(np.p)}` }) : t({ en: 'you clicked it yourself', zh: '你自己点过它' })}</p>
				<p class="small">{t({ en: `test sessions with it in the top ${K}: ${cnt(fact('newItemInTop5'))} of ${cnt(fact('testSessions'))}`, zh: `前 ${K} 名里有它的测试会话：${cnt(fact('testSessions'))} 个里有 ${cnt(fact('newItemInTop5'))} 个` })}</p></div>
			<div class="card"><p class="hd">{t({ en: 'The 5 most clicked products', zh: '点击最多的 5 件商品' })}</p>
				<p class="small">{t({ en: popTop(P.rt).map((i) => ITEMS[i].name.en).join(', '), zh: popTop(P.rt).map((i) => ITEMS[i].name.zh).join('、') })}</p>
				{#each [{ l: { en: 'share of the next clicks', zh: '在下一次点击里的份额' }, v: fact('popShareTarget') }, { l: { en: 'share of the top-5 slots', zh: '在前 5 名推荐位里的份额' }, v: fact('popShareRec') }] as b (b.l.en)}
					<p class="pb"><span class="small">{t(b.l)}</span><span class="bar"><i style:width="{100 * b.v}%" style:background={seqc.css(0.75)}></i></span><b class="num">{fmtPct(b.v)}</b></p>{/each}
				<p class="small">{t({ en: 'Test sessions; bars on a 0 to 100% scale.', zh: '测试会话；条长从 0 到 100%。' })}</p></div>
		</div>
	{/if}
</div>

<style>
	.sv { display: grid; gap: 6px; min-width: 0; font-size: var(--fs-sm); }
	.hd { margin: 2px 0; font-weight: 600; } .small { margin: 0; font-size: var(--fs-xs); } .muted { color: var(--muted); }
	.d { display: inline-block; flex: none; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; }
	.ls { display: grid; gap: 1px; margin: 0; padding: 0; list-style: none; }
	.r { display: grid; grid-template-columns: 22px auto minmax(0, 1fr) minmax(60px, 35%) 48px; align-items: center; gap: 4px; padding: 1px 4px; border-radius: 3px; }
	.r.in { background: var(--accent-soft); } .r.top { outline: 2px solid var(--ink); outline-offset: -2px; }
	.k { color: var(--muted); font-size: 11px; } .ti { overflow: hidden; font-size: var(--fs-xs); text-overflow: ellipsis; white-space: nowrap; }
	.bar { position: relative; height: 10px; border-left: 1px solid var(--rule); } .bar i { position: absolute; top: 1px; bottom: 1px; left: 0; }
	b.num { font-size: 11px; text-align: right; }
	.cut { color: var(--accent-strong); font-size: 11px; font-weight: 600; text-align: center; }
	.sc { overflow-x: auto; max-width: 100%; }
	.tb { border-collapse: collapse; font-size: var(--fs-xs); } .tb th, .tb td { padding: 2px 4px; border-bottom: 1px solid var(--rule); text-align: left; } .tb td.num { text-align: right; }
	.emb tr.last td { background: var(--accent-soft); } .op { color: var(--muted); } .nm { white-space: nowrap; }
	.mx td.w { min-width: 26px; font: 11px var(--font-mono); text-align: center; }
	.mx td.m { min-width: 26px; background: repeating-linear-gradient(45deg, transparent 0 3px, var(--rule) 3px 5px); }
	.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr)); gap: 8px; }
	.card { display: grid; gap: 4px; align-content: start; min-width: 0; padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px; } .card p { margin: 0; }
	.pb { display: grid; grid-template-columns: minmax(0, 10em) minmax(60px, 1fr) 48px; align-items: center; gap: 6px; } .pb .bar { height: 14px; }
	/* the weights answer step 3's predict question: hidden while the kit's predict gate makes the stage inert */
	:global([inert]) .answer { visibility: hidden; }
</style>
