/**
 * Real-scale evidence from our BERT reproduction. Every number here is copied from
 * research/bert_facts.json (keys named next to each block) or, for MNLI → RTE, from
 * research/bert_improve.json; nothing is derived except simple differences that the files
 * themselves state (gaps, "+2.0", "+14.3"). Provenance on the page: precomputed.
 *
 * Public slides: https://robintianqili.github.io/bert-reproduction/
 */
export const BERT_SLIDES = 'https://robintianqili.github.io/bert-reproduction/';

/** scale.* and paper_pipeline.* */
export const BERT_SCALE = {
	runs: 218,
	gpuHours: 8.9,
	baseParamsM: 110,
	recipe: { batch: 32, epochs: 3, lrs: ['2e-5', '3e-5', '4e-5', '5e-5'] }
};

/** glue_data.rows (training-set sizes) */
export const GLUE_TRAIN = {
	MNLI: 392702,
	SST2: 67349,
	MRPC: 3668,
	RTE: 2490
};

/** finetune_claims.features_vs_finetuning */
export const FEATURES_VS_FT = {
	rows: [
		{
			id: 'ner',
			task: { en: 'CoNLL-2003 NER (BERT paper)', zh: 'CoNLL-2003 命名实体识别（BERT 论文）' },
			metric: 'F1',
			frozen: 96.1,
			ft: 96.4,
			gap: 0.3,
			ours: false
		},
		{ id: 'sst2', task: { en: 'SST-2 (ours)', zh: 'SST-2（我们的复现）' }, metric: 'acc', frozen: 86.8, ft: 93.0, ftStd: 0.2, gap: 6.2, ours: true },
		{ id: 'mrpc', task: { en: 'MRPC (ours)', zh: 'MRPC（我们的复现）' }, metric: 'acc', frozen: 76.0, ft: 84.1, gap: 8.1, ours: true }
	],
	perLayerMean: [77.9, 78.1, 78.7, 79.5, 80.4, 80.6, 82.0, 83.0, 84.4, 85.4, 85.4, 85.1, 85.3],
	perLayerCls: [50.9, 74.8, 78.1, 77.9, 77.0, 76.6, 79.4, 81.8, 82.6, 83.6, 85.0, 86.5, 86.7],
	majority: 50.9,
	bestFrozen: 86.8,
	ftSst2: 93.0,
	ftSst2Std: 0.2,
	probeSubsample: 20000,
	combos: 27
};

/** finetune_claims.lr_search: mean and sample std per learning rate */
export const LR_SEARCH = {
	mrpc: {
		points: [
			{ lr: '2e-5', mean: 84.1, std: 1.3 },
			{ lr: '3e-5', mean: 85.4, std: 0.6 },
			{ lr: '4e-5', mean: 86.1, std: 1.3 },
			{ lr: '5e-5', mean: 85.8, std: 0.5 }
		],
		best: '4e-5',
		gain: 2.0,
		paper: 86.7,
		paperIsTest: false
	},
	rte: {
		points: [
			{ lr: '2e-5', mean: 63.0, std: 2.7 },
			{ lr: '3e-5', mean: 63.5, std: 1.8 },
			{ lr: '4e-5', mean: 64.8, std: 0.8 },
			{ lr: '5e-5', mean: 64.2, std: 1.6 }
		],
		best: '4e-5',
		gain: 1.8,
		paper: 66.4,
		paperIsTest: true
	}
};

/** finetune_claims.bert_large_stability (BERT-Large on RTE, 6 seeds, 3 epochs, lr 2e-5) */
export const LARGE_RTE = {
	original: { perSeed: [47.3, 47.3, 52.7, 67.5, 47.3, 63.9], mean: 54.3, std: 9.1, collapsed: 4 },
	adamw: { perSeed: [70.8, 73.3, 69.0, 66.4, 71.1, 70.8], mean: 70.2, std: 2.3, collapsed: 0 },
	collapseValues: [47.3, 52.7],
	lift: 15.9
};

/**
 * Intermediate-task training, MNLI first, then RTE (BERT-base). Copied from research/bert_improve.json,
 * our BERT reproduction (5 seeds, runs tagged `repro` vs `mnli_inter` in the reproduction's run log,
 * lr 2e-5, 3 epochs, batch 32, same seeds, paired). Re-checked against that run log on 2026-10-06: every
 * per-seed and per-epoch value matches; means and sample std (n − 1) round to the values below.
 */
export const MNLI_RTE = {
	seeds: [1, 2, 3, 4, 5],
	direct: { perSeed: [59.6, 60.6, 64.6, 65.0, 65.3], mean: 63.0, std: 2.7, perEpoch: [57.8, 63.0, 63.0] },
	mnliFirst: { perSeed: [78.0, 78.0, 76.2, 78.0, 76.5], mean: 77.3, std: 0.9, perEpoch: [75.5, 77.0, 77.3] },
	gain: 14.3,
	higherOn: 5,
	rte: { trainPairs: 2490, devPairs: 277, updates: 234 },
	mnli: { trainPairs: 392702, ratio: 158, steps: 36816, minutes: 36, devMatched: 83.9, devMismatched: 84.2 },
	/** our seed-1 models on RTE dev pair 219 */
	example: {
		devIndex: 219,
		a: 'Napkins, invitations and plain old paper cost more than they did a month ago.',
		b: 'The cost of paper is rising.',
		aZh: '餐巾纸、请柬和普通的纸都比一个月前贵了。',
		bZh: '纸的价格在上涨。',
		directP: 0.33,
		mnliP: 0.94,
		fixes: 68,
		breaks: 20
	},
	/** STILTs (Phang et al., 2018): BERT-base RTE, best of 20 restarts (not means) */
	phang: { direct: 68.6, mnliFirst: 79.1, restarts: 20 }
};
