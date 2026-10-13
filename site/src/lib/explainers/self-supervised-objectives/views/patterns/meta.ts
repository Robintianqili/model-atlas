// self-supervised-objectives view 'patterns': objectives drawn as mask patterns (objectives.ts) on the learner's text and
// on one drawing's 4 x 4 patch grid. For each objective: what the model sees, what is hidden, what it must predict.
//   props.show: ['nsp'] | ['span', 'denoise'] | ...   the objectives to draw (default: all)
//   props.pick: true    one objective at a time, chosen with buttons (the toy)
//   props.brief: true   leave out the "sees / predicts" lines (a long list)
// When the step's source is the 'hide' process, the masked-LM row shows that run's 80/10/10 draw.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import type { MaskedRaw } from '../../adapters/masked-text/adapter.ts';
import { objectives } from '../../objectives.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

export function patternsOf(p: P) {
	const st = (p.source === 'hide' && p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : null) as { fate?: Tensor; shown?: Tensor } | null;
	const all = objectives(p.rt.input as MaskedRaw, st);
	const show = Array.isArray(p.props.show) ? (p.props.show as string[]) : null;
	return { all, list: show ? all.filter((o) => show.includes(o.id)) : all, pick: !!p.props.pick, brief: !!p.props.brief };
}

const ROLE: Record<string, I18n> = { seen: { en: 'seen', zh: '看得见' }, gone: { en: 'hidden', zh: '藏起' }, target: { en: 'to predict', zh: '要预测' }, mark: { en: 'marker', zh: '标记' } };

const meta = {
	key: 'self-supervised-objectives/patterns',
	title: { en: 'Objectives as mask patterns', zh: '把训练目标画成遮罩图案' },
	kinds: ['trace', 'process'],
	describe: (p: P): I18n => {
		const { list } = patternsOf(p);
		const one = (o: (typeof list)[number], l: 'en' | 'zh') =>
			`${o.name[l]} (${o.example[l]}): ${l === 'en' ? 'sees' : '看到'} ${o.sees[l]}; ${l === 'en' ? 'predicts' : '预测'} ${o.predicts[l]}. ` +
			o.lines.map((x) => `${x.label[l]}: ${x.chips.map((c) => `${c.text} (${ROLE[c.role][l]})`).join(' ')}`).join('. ');
		return { en: list.map((o) => one(o, 'en')).join(' | '), zh: list.map((o) => one(o, 'zh')).join(' | ') };
	},
	demo: async () => demoProps({ show: ['span', 'mae'] }),
	minHeight: 200
} satisfies ViewMeta;

export default meta;
