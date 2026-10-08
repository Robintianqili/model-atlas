<!--
  tiny-unet-seg Capture (local adapter 'shapes32'): the 32×32 picture with the classes you gave it tinted, the presets,
  and the tools: pick a class (or the eraser), then stamp a whole shape with a tap or paint pixels by dragging; a size
  slider (the training set used 3 to 8), new noise, clear. Any edit drops the preset mark, so the task can tell your own
  scene from a preset. The compact copy beside the lesson shows the picture and opens the tools on request.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '$lib/atlas/ui/capture';
	import { say } from '$lib/atlas/ui/strings';
	import { CLASSES, PRESETS, S, TRAIN_R, presetRaw, render, type SegRaw } from '../../scenes.ts';
	import { rawOf } from './adapter.ts';
	import Seg from '../../parts/Seg.svelte';

	let { raw, onchange, compact = false }: CaptureProps = $props();
	const r = $derived(rawOf(raw));
	const pic = $derived(render(r));
	let k = $state(1), mode = $state<'stamp' | 'paint'>('stamp'), size = $state(5), open = $state(false), cur = $state<[number, number]>([16, 16]), last = -1;
	let draft: number[] | null = null; // the paint of a drag in progress (live updates reach `raw` only once per frame)
	const send = (p: Partial<SegRaw>, live = false) => onchange?.({ shapes: r.shapes ?? [], paint: r.paint ?? [], seed: r.seed ?? 0, ...p }, { live });
	function pick([y, x]: [number, number], key = false) {
		cur = [y, x];
		if (mode === 'stamp' && k > 0) {
			if (key) last = -1;
			if (last >= 0) return;
			last = y * S + x;
			return send({ shapes: [...(r.shapes ?? []), { k, x: x + 0.5, y: y + 0.5, r: size }] });
		}
		const near: number[] = [];
		for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dx * dx + dy * dy <= 1 && y + dy >= 0 && y + dy < S && x + dx >= 0 && x + dx < S) near.push((y + dy) * S + x + dx);
		last = y * S + x;
		draft = [...(draft ?? r.paint ?? []).filter((e) => !near.includes(e >> 2)), ...near.map((i) => i * 4 + k)];
		send({ paint: draft }, !key);
		if (key) (last = -1), (draft = null);
	}
	function up() {
		if (draft) send({ paint: draft });
		(last = -1), (draft = null);
	}
	const TOOLS = [...CLASSES.slice(1).map((c, i) => ({ k: i + 1, label: c })), { k: 0, label: { en: 'eraser', zh: '橡皮' } }];
</script>

<div class="cap" class:compact>
	<div class="pad" onpointerup={up} onpointerleave={up} role="presentation">
		<Seg img={pic.image} cls={pic.label} alpha={0.35} size={compact ? 168 : 256} sel={!compact || open ? cur : null} label={t({ en: 'Your picture: tap to stamp, drag to paint (keys: arrows move, Enter puts down)', zh: '你的图：轻点盖章，拖动涂画（键盘：方向键移动，回车放下）' })} onpick={!compact || open ? (rc, key) => pick(rc, key) : undefined} onmove={(rc) => (cur = rc)} />
	</div>
	<div class="row" role="group" aria-label={say('presets')}>
		{#each PRESETS as p (p.id)}<button type="button" class="chip" aria-pressed={r.preset === p.id} onclick={() => onchange?.(presetRaw(p.id))}>{t(p.label)}</button>{/each}
	</div>
	{#if !compact || open}
		<div class="row" role="group" aria-label={t({ en: 'What to put down', zh: '要放什么' })}>
			{#each TOOLS as q (q.k)}<button type="button" class="chip" aria-pressed={k === q.k} onclick={() => ((k = q.k), q.k === 0 && (mode = 'paint'))}><i class="sw k{q.k}"></i>{t(q.label)}</button>{/each}
		</div>
		<div class="row" role="group" aria-label={t({ en: 'How', zh: '怎么放' })}>
			<button type="button" class="chip" aria-pressed={mode === 'stamp'} disabled={k === 0} onclick={() => (mode = 'stamp')}>{t({ en: 'stamp a whole shape', zh: '盖一个完整形状' })}</button>
			<button type="button" class="chip" aria-pressed={mode === 'paint'} onclick={() => (mode = 'paint')}>{t({ en: 'paint pixels', zh: '逐像素涂画' })}</button>
		</div>
		<label class="row small">{t({ en: 'Stamp size', zh: '盖章大小' })} <input type="range" min="2" max="14" step="0.5" bind:value={size} disabled={mode === 'paint'} /> <span class="num">{size}</span><span class="muted">{t({ en: `(training: ${TRAIN_R[0]} to ${TRAIN_R[1]})`, zh: `（训练时：${TRAIN_R[0]} 到 ${TRAIN_R[1]}）` })}</span></label>
		<div class="row">
			<button type="button" class="btn small" onclick={() => send({ seed: (r.seed ?? 0) + 1 })}>{t({ en: 'New noise', zh: '换一种噪声' })}</button>
			<button type="button" class="btn small" onclick={() => send({ shapes: [], paint: [] })}>{t({ en: 'Clear', zh: '清空' })}</button>
		</div>
		<p class="muted small">{t({ en: 'The tint shows the class you gave each pixel; it is what the model’s answer is scored against.', zh: '着色表示你给每个像素定的类别；模型的答案就和它比对打分。' })}</p>
	{/if}
	{#if compact}<button type="button" class="btn small" aria-expanded={open} onclick={() => (open = !open)}>{open ? say('hideInput') : t({ en: 'Edit the picture here', zh: '在这里修改图' })}</button>{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; justify-items: start; min-width: 0; width: 100%; }
	.pad { width: 100%; max-width: 256px; touch-action: pinch-zoom; }
	.compact .pad { max-width: 168px; }
	.row { display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: center; font-size: var(--fs-xs); }
	.chip { cursor: pointer; font-size: var(--fs-xs); padding: 2px 8px; display: inline-flex; gap: 4px; align-items: center; }
	.chip[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); }
	.sw { width: 10px; height: 10px; border-radius: 2px; border: 1px solid var(--rule); }
	.k1 { background: var(--cat-0); } .k2 { background: var(--cat-1); } .k3 { background: var(--cat-2); } .k0 { background: var(--surface-2); }
	.small { font-size: var(--fs-sm); margin: 0; }
	.btn.small { min-height: 32px; padding: 0 12px; }
	input[type='range'] { width: 120px; }
</style>
