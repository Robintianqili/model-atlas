// View 'attention-variants/presets': the KV cache per token of every published configuration (data.ts PRESETS, at their
// usual bytes per number) and of the learner's configuration, as bars on one linear axis in KiB, colored by the variant
// (categorical: MHA, GQA, MQA, MLA, named in the legend). Each bar prints its value; the learner's row is outlined. No props.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { bytesText, KIND_SHORT, kindOf, perToken, PRESETS, presetConfig, type Config } from '../../data.ts';
import { configAt } from '../heads/meta.ts';

export type Bar = { id: string; label: I18n; c: Config; mine: boolean };
export function barsOf(mine: Config): Bar[] {
	const bars: Bar[] = PRESETS.map((p) => ({ id: p.id, label: p.label, c: presetConfig(p), mine: false }));
	const same = bars.find((b) => perToken(b.c) === perToken(mine) && b.c.layers === mine.layers && b.c.heads === mine.heads);
	if (same && (mine.preset === same.id || mine.preset === 'custom')) same.mine = true;
	else bars.push({ id: 'mine', label: { en: 'Your configuration', zh: '你的配置' }, c: mine, mine: true });
	return bars;
}

const meta = {
	key: 'attention-variants/presets',
	title: { en: 'KV cache per token of published models', zh: '已发表模型每个词元的 KV 缓存' },
	kinds: ['process'],
	describe: (p: StageProps) => {
		const b = barsOf(configAt(p));
		return {
			en: 'KV cache per token: ' + b.map((x) => `${x.label.en} (${KIND_SHORT[kindOf(x.c)]}) ${bytesText(perToken(x.c))}`).join('; ') + '.',
			zh: '每个词元的 KV 缓存：' + b.map((x) => `${x.label.zh}（${KIND_SHORT[kindOf(x.c)]}）${bytesText(perToken(x.c))}`).join('；') + '。'
		};
	},
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const rt = fakeRuntime({ slug: 'attention-variants', input: { preset: 'custom', layers: 24, heads: 16, kvHeads: 4, dHead: 64, tokens: 4096, bytes: 2, latent: 0 }, tapes: {} });
		return { rt, source: 'cache', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
