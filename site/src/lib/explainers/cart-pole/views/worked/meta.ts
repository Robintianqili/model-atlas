// cart-pole view 'worked': another view (props.view, with props.inner as its props) and under it the numbers the
// narration works out: props.note (a line) and props.tex filled with props.values (a formula, with props.where as its
// legend), so every number the step narrates is on the stage. The lesson builds the props from the step's frame;
// props.say is the stage's text form.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import scene from '../scene/meta.ts';

const meta = {
	key: 'cart-pole/worked',
	title: { en: 'A view with its worked numbers', zh: '视图和它的算式' },
	kinds: ['trace', 'train', 'process'],
	describe: (p: Omit<StageProps, 'ctx'>): I18n => (p.props.say ?? p.props.note ?? { en: 'A view with its worked numbers.', zh: '一个视图和它的算式。' }) as I18n,
	async demo() {
		const d = await scene.demo();
		return { ...d, props: { view: 'cart-pole/scene', note: { en: 'A line under the view.', zh: '视图下面的一行字。' }, tex: 'y = {{a}} + {{b}} = {{c}}', values: { a: 1, b: 2, c: 3 } } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
