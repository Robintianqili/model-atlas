<!-- mlp-playground: the page's Code section: the network and its training in PyTorch (pipeline/mlp-playground/twin.py is checked by atlas parity) -->
<script lang="ts">
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import Tex from '$lib/kit/Tex.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	const code = `net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.03)
for step in range(300):                         # every step uses all the points
    loss = F.cross_entropy(net(xy), label)      # xy: [n, 2], label: [n]
    opt.zero_grad(); loss.backward(); opt.step()`;
	const loss = String.raw`L = -\frac{1}{n}\sum_{i=1}^{n} \ln p_\theta(y_i \mid x_i)`;
	const step = String.raw`\theta \leftarrow \theta - \eta \, \nabla_\theta L`;
</script>

<CodeBlock {code} lang="python" caption={{ en: 'The same network and training in PyTorch. Its float64 twin (pipeline/mlp-playground/twin.py) matches the page’s trainer within 1e−7 over 50 steps.', zh: '同样的网络和训练，用 PyTorch 写。它的 float64 孪生版本（pipeline/mlp-playground/twin.py）在 50 步内与本页的训练器一致，误差在 1e−7 以内。' }} />
<p>{t({ en: 'The loss it lowers (lesson step 6) and one gradient-descent step on it (step 8; ln is the natural logarithm, η the learning rate):', zh: '它要降低的损失（课程第 6 步），以及在它上面走的一步梯度下降（第 8 步；ln 是自然对数，η 是学习率）：' })}</p>
<Tex display tex={loss} />
<Tex display tex={step} />
