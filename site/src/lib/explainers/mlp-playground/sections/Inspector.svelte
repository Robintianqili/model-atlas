<!-- mlp-playground: the page's Inspector section on the live process sources (why: notes/mlp-playground.md, Kit issues 1 and 3) -->
<script lang="ts">import Live from '../parts/LiveStage.svelte'; let { store } = $props();</script>
<Live {store} source="run" view="mlp-playground/neuron-grid" props={{ show: 'layer', layer: 1 }} note={{ en: 'Layer 2, neuron by neuron: each square is already a curved region, made from the lines of layer 1.', zh: '第 2 层，一个神经元一个神经元地看：每个方块已经是一块弯曲的区域，由第 1 层的线组合而成。' }} />
