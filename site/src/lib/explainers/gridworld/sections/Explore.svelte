<!-- gridworld: the page shell's Overview, "Inside a layer", "The arithmetic" and Output sections expect a network; here they
     show the learned maze, the Q-table, the latest Q update and the bandit race, on the current input. -->
<script lang="ts">
	import { untrack } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../atlas/types/index.ts';
	import { atFor } from '../../../atlas/runtime/sources.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import GridBoard from '../views/grid-board/View.svelte';
	import ArmBars from '../views/arm-bars/View.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import source from '../process.ts?raw';
	/** the loop body of one Q-learning move, cut from the source that ran */
	const MOVE = source.slice(source.indexOf('\t\t\twhile (moves'), source.indexOf('\t\t\tif (how === 1)')).replace(/^\t\t\t/gm, '');

	let { store, part }: { store: AtlasStore; part: 'overview' | 'inspector' | 'unit' | 'output' | 'code' } = $props();
	const src = $derived(part === 'output' ? 'race' : 'q');
	const PROPS = { overview: { show: 'heat,arrows,answer,edit' }, inspector: { show: 'tri' }, unit: { show: 'tri,update', formula: 'update' }, output: { mode: 'regret', reveal: true }, code: {} };
	const INTRO = {
		overview: { en: 'Your maze after Q-learning: each cell’s best Q and its arrow. Edit the maze below and the agent learns it again from scratch.', zh: 'Q 学习之后的你的迷宫：每个格子的最大 Q 和它的箭头。在下面改迷宫，智能体会从头再学一遍。' },
		inspector: { en: 'This model has no layers: everything it knows is the Q-table, four numbers per cell, shown here after the last episode.', zh: '这个模型没有层：它知道的一切都在 Q 表里，每个格子四个数；这里显示最后一个回合之后的 Q 表。' },
		unit: { en: 'The last update of the run that moved a Q value by more than 0.01, worked out with its numbers.', zh: '这次运行中最后一次让某个 Q 值变化超过 0.01 的更新，代入数字算一遍。' },
		code: { en: 'The code that ran: one move of Q-learning, from process.ts (the Python reference repeats it frame by frame).', zh: '实际运行的代码：Q 学习的一步，摘自 process.ts（Python 参考实现逐帧重复它）。' },
		output: { en: 'The four slot-machine strategies on the same machines: regret after every pull.', zh: '四种老虎机策略玩同样的机器：每一次拉动之后的遗憾。' }
	};
	$effect(() => {
		void store.inputGen;
		untrack(() => void store.rt.ensure(src, 'last').then(() => store.touch()));
	});
	const stage = $derived.by((): StageProps | null => {
		const rt = store.r, tape = rt.sources[src]?.tape;
		if (!tape?.length) return null;
		let i = tape.length - 1;
		if (part === 'unit') while (i > 0 && !(tape.frame(i).phase === 'update' && Math.abs(tape.frame(i).metrics!.delta) > 0.01)) i--;
		return { rt, source: src, at: atFor(rt, src, i), focus: [], sel: {}, provenance: rt.sources[src].provenance, autoplay: false, onend: () => {}, props: PROPS[part] };
	});
</script>

<p class="muted">{t(INTRO[part])}</p>
{#if part === 'code'}
	<CodeBlock code={MOVE} lang="ts" caption="src/lib/explainers/gridworld/process.ts" />
{:else if stage}
	{#if part === 'output'}<ArmBars {...stage} />{:else}<GridBoard {...stage} />{/if}
{:else}
	<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>
{/if}
