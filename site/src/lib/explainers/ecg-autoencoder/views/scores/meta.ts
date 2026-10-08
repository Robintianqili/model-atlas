// View 'ecg-autoencoder/scores': the test set's anomaly scores (testset.json, PyTorch). props.mode: hist (a histogram
// of the normal and the odd test beats, the threshold, the learner's beat, what the threshold catches; a slider sets
// the threshold in the learner's input), roc (on the 'sweep' source: the ROC curve as far as the sweep has run, the
// learner's threshold as a dot, the area under the curve; the same slider), limits (the code-size screen from
// facts.json: how well each size rebuilds normal beats and how well it separates odd ones). Drawing: View.svelte.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { I } from '../../beats.ts';
import { countAt, recOf, testsetOf, thrOf } from '../../ecg.ts';

export type Mode = 'hist' | 'roc' | 'limits';
export const SIZES = [{ c: 2, auc: 'c2Auc', p: 'c2AucNoP', e: 'c2TrainScore' }, { c: 4, auc: 'testAuc', p: 'auc_noP', e: 'trainScore' }, { c: 8, auc: 'c8Auc', p: 'c8AucNoP', e: 'c8TrainScore' }, { c: 16, auc: 'c16Auc', p: 'c16AucNoP', e: 'c16TrainScore' }];

export function scoresData(p: Omit<StageProps, 'ctx'>) {
	const mode = (['hist', 'roc', 'limits'].includes(p.props.mode as string) ? p.props.mode : 'hist') as Mode;
	const set = testsetOf(p.rt), thr = thrOf(p.rt.input);
	const at = set ? countAt(set, thr) : null;
	const m = p.at.frame?.metrics ?? {};
	return { mode, set, thr, at, mine: recOf(p.rt)?.score ?? NaN, k: p.at.frame?.phase === 'area' ? Infinity : (m.k ?? 0), auc: p.at.frame?.phase === 'area' ? m.auc : NaN, m };
}

export function describeScores(p: Omit<StageProps, 'ctx'>): I18n {
	const d = scoresData(p);
	if (d.mode === 'limits') {
		const rows = SIZES.map((s) => ({ c: s.c, auc: p.rt.num(`fact:${s.auc}`), e: p.rt.num(`fact:${s.e}`) }));
		return I(`Code sizes compared: ${rows.map((r) => `${r.c} numbers: rebuild error ${fmt3(r.e)}, AUC ${fmt3(r.auc)}`).join('; ')}.`, `比较编码大小：${rows.map((r) => `${r.c} 个数：重建误差 ${fmt3(r.e)}，AUC ${fmt3(r.auc)}`).join('；')}。`);
	}
	if (!d.set || !d.at) return I('The test set is not loaded yet.', '测试集还没有载入。');
	const a = d.at;
	const head = I(`${d.set.scores.length} test beats: ${a.neg} normal, ${a.pos} odd. At the threshold ${fmt3(d.thr)}, ${a.tp} odd beats are caught (recall ${fmtPct(a.recall)}) and ${a.fp} normal beats are false alarms; precision ${fmtPct(a.precision)}.`, `${d.set.scores.length} 个测试心跳：正常 ${a.neg} 个，异常 ${a.pos} 个。阈值为 ${fmt3(d.thr)} 时，抓到 ${a.tp} 个异常心跳（召回率 ${fmtPct(a.recall)}），${a.fp} 个正常心跳被误报；精确率 ${fmtPct(a.precision)}。`);
	if (d.mode === 'hist') return I(`${head.en}${Number.isFinite(d.mine) ? ` Your beat scores ${fmt3(d.mine)}.` : ''}`, `${head.zh}${Number.isFinite(d.mine) ? `你的心跳得分 ${fmt3(d.mine)}。` : ''}`);
	return I(`The ROC curve: catch rate against false-alarm rate as the threshold slides down.${Number.isFinite(d.auc) ? ` Area under it: ${fmt3(d.auc)}.` : ''} ${head.en}`, `ROC 曲线：阈值往下滑时，检出率随误报率的变化。${Number.isFinite(d.auc) ? `曲线下面积：${fmt3(d.auc)}。` : ''}${head.zh}`);
}

const meta = {
	key: 'ecg-autoencoder/scores',
	title: I('The test set’s scores, a threshold and the ROC curve', '测试集的分数、阈值和 ROC 曲线'),
	kinds: ['trace', 'process'],
	describe: describeScores,
	async demo() {
		// made-up scores: 40 normal beats around 0.02, 40 odd ones around 0.04
		const scores = Array.from({ length: 80 }, (_, i) => (i < 40 ? 0.014 + (i % 20) * 0.0007 : 0.022 + (i % 40) * 0.0009)), kinds = scores.map((_, i) => (i < 40 ? 0 : 1 + (i % 4)));
		const rt = fakeRuntime({ slug: 'ecg-autoencoder', input: { thr: 0.027 } });
		(rt as unknown as { assets: Record<string, unknown> }).assets = { 'testset.json': { scores, kinds } };
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'hist' } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
