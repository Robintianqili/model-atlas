// The auto page's Training section (K1, K-R3; pure TS, tested in ui.test.ts): readable, bilingual names and formats for
// training metrics (spec.training.metrics of an exported model, the metrics of a train source's frames), and the frame
// whose loss curve and numbers the section shows. A package names its own metrics in manifest.page.metrics.
import type { I18n } from '../types/core.ts';
import type { Frame, Tape } from '../types/process.ts';
import { fmt3, fmtInt, fmtPct } from '../core/format.ts';

export type MetricFmt = 'pct' | 'int' | 'n3' | 'n4';
export interface MetricRow {
	key: string;
	label: I18n;
	text: string;
}

const L = (en: string, zh: string): I18n => ({ en, zh });
const SET: Record<string, I18n> = { test: L('the test set', '测试集'), val: L('the validation set', '验证集'), valid: L('the validation set', '验证集'), train: L('the training set', '训练集') };

/** Names of common metric keys. Keys of the form <set>_<metric> (test_acc, val_loss, train_n) are built from SET. */
const KNOWN: Record<string, { label: I18n; fmt?: MetricFmt }> = {
	loss: { label: L('Loss', '损失'), fmt: 'n4' },
	acc: { label: L('Accuracy', '准确率'), fmt: 'pct' },
	accuracy: { label: L('Accuracy', '准确率'), fmt: 'pct' },
	correct: { label: L('Right', '正确数'), fmt: 'int' },
	lr: { label: L('Learning rate', '学习率'), fmt: 'n3' },
	gradNorm: { label: L('Gradient size', '梯度大小'), fmt: 'n3' },
	step: { label: L('Training step', '训练步'), fmt: 'int' },
	steps: { label: L('Training steps', '训练步数'), fmt: 'int' },
	epochs: { label: L('Epochs (passes over the data)', '轮数（把数据过一遍的次数）'), fmt: 'int' },
	bpc: { label: L('Bits per character', '每个字符的比特数'), fmt: 'n4' },
	mse: { label: L('Mean squared error', '均方误差'), fmt: 'n4' },
	mae: { label: L('Mean absolute error', '平均绝对误差'), fmt: 'n4' }
};
const PER_SET: Record<string, { en: (s: string) => string; zh: (s: string) => string; fmt: MetricFmt }> = {
	acc: { en: (s) => `Right on ${s}`, zh: (s) => `${s}上的正确率`, fmt: 'pct' },
	accuracy: { en: (s) => `Right on ${s}`, zh: (s) => `${s}上的正确率`, fmt: 'pct' },
	loss: { en: (s) => `Loss on ${s}`, zh: (s) => `${s}上的损失`, fmt: 'n4' },
	n: { en: (s) => `Examples in ${s}`, zh: (s) => `${s}的样本数`, fmt: 'int' },
	bpc: { en: (s) => `Bits per character on ${s}`, zh: (s) => `${s}上每个字符的比特数`, fmt: 'n4' },
	mse: { en: (s) => `Mean squared error on ${s}`, zh: (s) => `${s}上的均方误差`, fmt: 'n4' },
	mae: { en: (s) => `Mean absolute error on ${s}`, zh: (s) => `${s}上的平均绝对误差`, fmt: 'n4' }
};

/** The name and format of one metric key: the package's own (page.metrics), else a known key, else the key in words. */
export function metricName(key: string, own?: Record<string, { label: I18n; fmt?: MetricFmt }>): { label: I18n; fmt?: MetricFmt } {
	if (own?.[key]) return own[key];
	if (KNOWN[key]) return KNOWN[key];
	const m = /^(test|val|valid|train)[_.-]?(acc|accuracy|loss|n|bpc|mse|mae)$/i.exec(key);
	if (m) {
		const set = SET[m[1].toLowerCase()];
		const per = PER_SET[m[2].toLowerCase()];
		return { label: { en: per.en(set.en), zh: per.zh(set.zh!) }, fmt: per.fmt };
	}
	const words = key.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[_.-]+/g, ' ').trim();
	const w = words.charAt(0).toUpperCase() + words.slice(1);
	return { label: { en: w, zh: w } };
}

/** A metric value as text: by its format, else a share in [0, 1] of an accuracy-like key as a percentage, else an
 *  integer as is, else 4 significant digits. */
export function metricText(key: string, v: number | string, fmt?: MetricFmt): string {
	if (typeof v === 'string') return v;
	if (!Number.isFinite(v)) return fmt3(v);
	const f = fmt ?? (/acc|accuracy|share|rate$/i.test(key) && v >= 0 && v <= 1 ? 'pct' : Number.isInteger(v) ? 'int' : 'n4');
	if (f === 'pct') return fmtPct(v, 2);
	if (f === 'int') return fmtInt(v);
	return fmt3(v, f === 'n3' ? 3 : 4);
}

/** Rows for a metrics record (numbers and short strings), in the record's order. `skip`: keys not to list. */
export function metricRows(metrics: Record<string, unknown> | undefined, own?: Record<string, { label: I18n; fmt?: MetricFmt }>, skip: string[] = []): MetricRow[] {
	const out: MetricRow[] = [];
	for (const [k, v] of Object.entries(metrics ?? {})) {
		if (skip.includes(k)) continue;
		if (!((typeof v === 'number' && Number.isFinite(v)) || (typeof v === 'string' && v.length <= 40))) continue;
		const n = metricName(k, own);
		out.push({ key: k, label: n.label, text: metricText(k, v, n.fmt) });
	}
	return out;
}

/** The frame the Training section shows for a train source: its last 'snapshot' (weights and the loss so far), else the
 *  last frame with a loss, else the last frame. A run that goes on after training (Euler sampling after a flow model's
 *  training) is not shown at its last frame, which has no loss (K1: the curve read "step 257, loss —"). -1: no frames. */
export function trainingFrame(tape: Pick<Tape, 'length' | 'frame'> | null | undefined): number {
	if (!tape || tape.length === 0) return -1;
	let withLoss = -1;
	for (let i = tape.length - 1; i >= 0; i--) {
		const f: Frame = tape.frame(i);
		if (f.phase === 'snapshot') return i;
		if (withLoss < 0 && typeof f.metrics?.loss === 'number' && Number.isFinite(f.metrics.loss)) withLoss = i;
	}
	return withLoss >= 0 ? withLoss : tape.length - 1;
}
