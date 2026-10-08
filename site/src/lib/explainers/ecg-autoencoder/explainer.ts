// ecg-autoencoder: the package manifest (design §4.9). A small dense autoencoder trained offline (PyTorch) on synthetic
// normal heartbeats only (beats.ts, made in TS; pipeline/ecg-autoencoder), traced live on the learner's beat:
//   fwd    the exported model on the learner's 140 numbers: encoder -> 4-number code -> decoder -> the rebuilt beat
//   data   kind C: the synthetic beats (12 training beats here; train.py read the full sets this process wrote)
//   sweep  kind C: the threshold slides down over the test set's scores (testset.json, PyTorch) -> the ROC curve and its
//          area; checked frame by frame against pipeline/ecg-autoencoder/reference.py
// The threshold is part of the learner's input (raw.thr, set by the slider on the stage).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { I } from './beats.ts';
import { testsetOf } from './ecg.ts';

const manifest: ExplainerManifest = {
	slug: 'ecg-autoencoder',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/ecg-autoencoder', auto: ['spec.json', 'weights.bin', 'facts.json', 'testset.json'] },
	input: { adapter: 'series', presetsFromAdapter: true, options: { mode: 'curve', yRange: [-0.6, 1.4] } },
	output: { view: 'ecg-autoencoder/beat', layer: 'out' },
	sources: {
		fwd: { kind: 'trace' },
		data: { kind: 'process', process: 'data', seed: 2026, reference: false },
		sweep: { kind: 'process', process: 'sweep', seed: 1, input: (rt) => { const s = testsetOf(rt); return s ? { scores: s.scores, kinds: s.kinds } : null; } }
	},
	page: {
		goals: [
			I('Follow one heartbeat through an autoencoder: 140 numbers squeezed into a 4-number code, then rebuilt by the decoder.', '跟着一次心跳走过自编码器：140 个数被挤成 4 个数的编码，再由解码器重建出来。'),
			I('Explain why an autoencoder trained only on normal beats rebuilds odd beats badly, and turn the rebuild error into one anomaly score.', '解释为什么只用正常心跳训练的自编码器重建不好异常心跳，并把重建误差变成一个异常分数。'),
			I('Choose a threshold and read the trade-off: odd beats caught against false alarms, precision and recall, the ROC curve and the area under it.', '选一个阈值并读懂其中的取舍：抓到的异常和误报、精确率和召回率、ROC 曲线和它下面的面积。'),
			I('Say what this detector cannot do: its beats are synthetic, odd is not the same as dangerous, and it gives no medical advice.', '说出这个检测器做不到什么：它的心跳是合成的，“异常”不等于“危险”，它也不提供医疗建议。')
		],
		prequestion: {
			id: 'ecg-autoencoder-pre',
			question: I('An autoencoder learns to copy its input through a narrow middle. This one saw only normal heartbeats. How could it spot an odd beat?', '自编码器学的是让输入穿过一个狭窄的中间层再复制出来。这个自编码器只见过正常的心跳。它怎样才能发现异常的心跳？'),
			options: [I('It cannot: it never saw an odd beat, so it has nothing to compare with', '做不到：它从没见过异常心跳，没有可比较的东西'), I('It copies odd beats badly, and a large copying error marks them', '它复制异常心跳复制得不好，大的复制误差就是标记'), I('It learned a list of anomaly types and names the one it sees', '它学过一张异常类型表，能说出看到的是哪一种')],
			answer: 1,
			explain: I('It never needs examples of odd beats: it learned to rebuild normal ones. An odd beat comes back looking more normal, and the difference between the beat and its rebuild becomes the anomaly score.', '它根本不需要异常心跳的例子：它学会的是重建正常心跳。异常心跳被重建得更像正常心跳，心跳和它的重建之间的差别就成了异常分数。')
		},
		tryThis: [
			{ id: 'noisy', title: I('A noisy recording', '噪声大的记录'), setup: { preset: 'noisy' }, predict: I('Noise is not a heart problem. Will the model flag this beat?', '噪声并不是心脏的问题。模型会把这个心跳标为异常吗？'), look: I('Its score is far above the threshold: the model flags anything it cannot rebuild, a bad recording included.', '它的分数远高于阈值：模型会标记一切它重建不了的东西，糟糕的记录也不例外。'), goto: { step: 'score' } },
			{ id: 'noP', title: I('P wave missing', '缺 P 波'), setup: { preset: 'noP' }, predict: I('The P wave is the smallest bump. Will losing it move the score much?', 'P 波是最小的鼓包。少了它，分数会变很多吗？'), look: I('Compare its errors with the normal preset’s: a small part leaves a small error, so this beat only just crosses the starting threshold, and many beats with a missing P wave pass as normal.', '把它的误差和“正常心跳”预设比一比：小部件只留下小误差，所以这个心跳只是勉强越过起始阈值，很多缺 P 波的心跳则会被当作正常。'), goto: { step: 'error' } }
		],
		quiz: [
			{
				q: I('You lower the threshold. What happens on the test beats?', '你把阈值调低。测试心跳上会发生什么？'),
				options: [I('More odd beats are caught, and more normal beats become false alarms', '抓到的异常心跳更多，被误报的正常心跳也更多'), I('Fewer odd beats are caught, and fewer false alarms', '抓到的异常更少，误报也更少'), I('Nothing changes: the scores stay the same', '什么都不变：分数还是那些分数')],
				answer: 0,
				explain: I('The scores stay the same, but more of them reach the threshold: odd and normal beats alike. Recall rises and precision usually falls.', '分数本身不变，但会有更多分数达到阈值：异常的和正常的都一样。召回率上升，精确率通常下降。')
			},
			{
				q: I('Half of these test beats are odd; in real recordings odd beats may be rare. At the same threshold, what happens to precision when odd beats are rare?', '这里的测试心跳有一半是异常的；在真实记录里，异常心跳可能很少。阈值不变时，异常心跳很少会让精确率怎样？'),
				options: [I('It falls: even a small false-alarm rate can then flag more normal beats than odd ones', '下降：这时即使误报率很小，被标记的正常心跳也可能比异常心跳多'), I('It stays the same', '保持不变'), I('It rises', '上升')],
				answer: 0,
				explain: I('Recall and the false-alarm rate do not depend on how common odd beats are, but precision does: the flagged normal beats grow with the number of normal beats.', '召回率和误报率与异常心跳有多常见无关，精确率却有关：被标记的正常心跳会随着正常心跳的数量一起增长。')
			}
		],
		misconception: {
			claim: I('A high anomaly score means the heart has a problem.', '异常分数高，就说明心脏有问题。'),
			truth: I('It means the model could not rebuild this beat from what it learned. A noisy recording scores high too, and an anomaly that looks normal enough can pass.', '它只说明模型没能用学过的东西重建这次心跳。噪声大的记录分数也会高，而看起来足够正常的异常则可能蒙混过关。'),
			step: 'limits'
		},
		metrics: {
			train_mse: { label: I('Mean squared rebuild error, training beats', '训练心跳的平均平方重建误差'), fmt: 'n3' },
			val_mse: { label: I('Mean squared rebuild error, normal validation beats', '正常验证心跳的平均平方重建误差'), fmt: 'n3' },
			test_auc: { label: I('Area under the ROC curve, test beats', '测试心跳的 ROC 曲线下面积'), fmt: 'n3' }
		},
		lessonNote: I('Ten steps: your beat goes through the encoder and the decoder, you see what happens to an odd beat, turn the error into one score, choose a threshold on the test beats, read the ROC curve, draw your own odd beat, and see the limits.', '十步：你的心跳穿过编码器和解码器，你会看到异常心跳会怎样，把误差变成一个分数，在测试心跳上选阈值，读懂 ROC 曲线，画出你自己的异常心跳，最后看局限。')
	},
	glossary: [
		{ id: 'ecg', term: I('electrocardiogram (ECG)', '心电图（ECG）'), def: I('A recording of the heart’s electrical activity. Each beat shows a small P wave, a tall QRS spike and a T wave.', '心脏电活动的记录。每次心跳都有一个小的 P 波、一个高的 QRS 尖峰和一个 T 波。') },
		{ id: 'autoencoder', term: I('autoencoder', '自编码器'), def: I('A network trained to rebuild its own input through a narrow middle layer, the code. It learns what typical inputs look like, so inputs unlike its training data come back changed.', '一种被训练来重建自身输入的网络，输入要穿过一个狭窄的中间层，即编码。它学到典型输入的样子，所以和训练数据不像的输入被重建后会走样。'), related: ['encoder-decoder', 'bottleneck'] },
		{ id: 'bottleneck', term: I('bottleneck (code)', '瓶颈（编码）'), def: I('The narrowest layer of an autoencoder. Everything the decoder knows about the input must pass through these few numbers.', '自编码器最窄的那一层。解码器对输入所知道的一切，都必须经过这几个数。') },
		{ id: 'anomaly-score', term: I('anomaly score', '异常分数'), def: I('One number for how unusual an input looks to a model; here the typical size of the rebuild error (the root mean square error). At or above a chosen threshold, the input is flagged.', '表示一个输入在模型看来有多反常的一个数；这里是重建误差的典型大小（均方根误差）。达到或超过选定的阈值，这个输入就被标记。'), related: ['rmse'] },
		{ id: 'decision-threshold', term: I('threshold', '阈值'), def: I('The cut-off that turns a score into yes or no: scores at or above it are flagged. Lowering it catches more cases but raises more false alarms.', '把分数变成“是”或“否”的分界线：达到或超过它的分数被标记。把它调低能抓到更多，但误报也更多。') },
		{ id: 'precision-recall', term: I('precision and recall', '精确率和召回率'), def: I('Recall: the share of the real cases that are flagged. Precision: the share of the flagged cases that are real. A lower threshold raises recall and usually lowers precision.', '召回率：真实情况中被标记出来的比例。精确率：被标记的情况中真实的比例。阈值调低会提高召回率，通常也会降低精确率。') },
		{ id: 'roc-curve', term: I('ROC curve and AUC', 'ROC 曲线和 AUC'), def: I('The catch rate (recall) plotted against the false-alarm rate, for every threshold. The area under it (AUC) is the chance that a random real case scores higher than a random normal one: 0.5 is guessing, 1 is perfect.', '对每个阈值，把检出率（召回率）对误报率画出来的曲线。曲线下面积（AUC）是随机一个真实情况比随机一个正常情况得分更高的概率：0.5 等于瞎猜，1 是完美。'), related: ['precision-recall', 'decision-threshold'] }
	],
	credits: [
		{
			name: 'Anomagram',
			url: 'https://github.com/victordibia/anomagram',
			license: 'MIT',
			usedFor: 'Ideas only (no code or data): an autoencoder trained on normal ECG beats of 140 points flags the beats it cannot rebuild; drawing a beat, the rebuild overlay, the threshold, the error histogram and the ROC curve',
			attribution: 'Anomagram by Victor Dibia (Cloudera Fast Forward Labs), MIT License'
		}
	],
	catalog: {
		hook: I('Draw a heartbeat; the network flags the beats it cannot rebuild.', '画一段心跳；网络会标出它无法重建的心跳。'),
		shows: I('Encoder, bottleneck and decoder, the rebuild overlay, the error at every point, the anomaly score, a threshold on a histogram of test scores, and the ROC curve.', '编码器、瓶颈层与解码器，重建叠加图，每个点的误差，异常分数，测试分数直方图上的阈值，以及 ROC 曲线。'),
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）'),
		params: I('9,424', '9,424'),
		size: I('38 KB of weights', '权重 38 KB')
	}
};

export default manifest;
