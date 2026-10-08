<!-- mlp-playground: the page's Unit section on the live process sources (why: notes/mlp-playground.md, Kit issues 1 and 3) -->
<script lang="ts">import Live from '../parts/LiveStage.svelte'; let { store } = $props();</script>
<Live {store} source="sgd1" view="train-step" at={{ phase: 'update', step: 0 }} note={{ en: 'One gradient-descent update of one weight, with the numbers that ran (lesson step 8).', zh: '一次梯度下降对一个权重的更新，带着实际算出的数字（课程第 8 步）。' }} />
