<!-- mlp-playground: the page's Overview section on the live process sources (why: notes/mlp-playground.md, Kit issues 1 and 3) -->
<script lang="ts">import Live from '../parts/LiveStage.svelte'; let { store } = $props();</script>
<Live {store} source="run" view="mlp-playground/neuron-grid" props={{ show: 'net' }} note={{ en: 'The whole network on your points after training: every hidden neuron, the weights between them and the output.', zh: '训练之后，整个网络在你的点上的样子：每个隐藏神经元、它们之间的权重，以及输出。' }} />
