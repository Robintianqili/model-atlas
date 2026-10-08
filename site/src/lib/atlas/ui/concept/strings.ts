// UI strings of the concept page (AutoConceptPage, ConceptPanel). The shared ones (lesson end card, side panel, quiz)
// come from ../strings.ts; these are the concept-only ones. say() is reactive like ../strings.ts say().
import { t } from '$lib/i18n/i18n.svelte';
import type { I18n } from '$lib/i18n/types';

export const CSTR = {
	loading: { en: 'Loading the live parts of this page…', zh: '正在加载本页的实时部分…' },
	error: { en: 'The live parts of this page could not be loaded: {e}', zh: '本页的实时部分加载失败：{e}' },
	dials: { en: 'Your input', zh: '你的输入' },
	inLesson: { en: 'See it in the lesson', zh: '在课程中查看' },
	further: { en: 'Further reading', zh: '延伸阅读' },
	lessonFoot: {
		en: 'Every number in the lesson is computed in your browser from your input. Change it in the side panel or in the toy below.',
		zh: '课程里的每个数都由你的浏览器根据你的输入算出。可以在侧栏或下面的玩具实验里修改输入。'
	},
	lessonEndConcept: { en: 'Change your input and play it again, or try the toy and the other runs below.', zh: '换个输入再播放一遍，或者试试下面的玩具实验和其他运行。' },
	exploreIntro: {
		en: 'The same runs as the lesson, without the steps: turn the dials, compare runs and see the idea on a bigger model.',
		zh: '与课程相同的运行，只是不分步骤：转动旋钮、对比不同的运行，并在更大的模型上看同一个想法。'
	}
} satisfies Record<string, I18n>;
export type CStrKey = keyof typeof CSTR;

export function csay(key: CStrKey, vars?: Record<string, string | number>): string {
	let s = t(CSTR[key]);
	if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
	return s;
}
