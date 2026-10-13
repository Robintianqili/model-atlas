/**
 * The catalog pages' own strings (redesign wave W1). They live here, not in i18n/strings.ts, because a table exported
 * from strings.ts lands in the strings chunk every page loads as soon as any page uses it (Rollup keeps a module whole):
 * importing CATALOG_UI from the catalog pages put about 0.7 KB gzip into every shell, the planned pages and the lesson
 * pages included. This module is imported only by the catalog pages and their components. The values are the ones wave F
 * wrote in CATALOG_UI.
 *   ct('allN', { n: 6 })   -> "All 6" / "全部 6"
 */
import { t } from '$lib/i18n/i18n.svelte';

const CAT = {
	liveAndPlanned: { en: '{n} live · {m} planned', zh: '{n} 节可学 · {m} 节计划中' },
	liveN: { en: '{n} live', zh: '{n} 节可学' },
	plannedN: { en: 'Planned · {n}', zh: '计划中 · {n}' },
	notLiveYet: { en: 'Not live yet', zh: '尚未上线' },
	planned: { en: 'Planned', zh: '计划中' },
	coming: { en: 'coming', zh: '即将上线' },
	comingN: { en: '{n} coming', zh: '{n} 节即将上线' },
	allN: { en: 'All {n}', zh: '全部 {n}' },
	andMore: { en: 'and {n} more', zh: '另外 {n} 个' },
	pathMeta: { en: '{n} lessons · ≈ {m} min', zh: '{n} 节课 · 约 {m} 分钟' },
	startWith: { en: 'Start with {name}', zh: '从 {name} 开始' },
	browseAll: { en: 'Browse all {n} lessons', zh: '浏览全部 {n} 节课' },
	byInput: { en: 'Or start from your kind of input', zh: '或者从你的输入类型开始' },
	startHere: { en: 'Start here', zh: '从这里开始' },
	paths: { en: 'Learning paths', zh: '学习路径' },
	byArea: { en: 'Browse by area', zh: '按领域浏览' },
	methods: { en: 'Training methods', zh: '训练方法' },
	modelsByArea: { en: 'Models by area', zh: '各领域的模型' },
	allLessons: { en: 'All lessons', zh: '全部课程' },
	followIt: { en: 'Follow it step by step', zh: '一步一步跟着看' },
	kind: { en: 'Kind', zh: '类型' },
	kindAll: { en: 'All', zh: '全部' },
	kindModels: { en: 'Models', zh: '模型' },
	kindMethods: { en: 'Training methods', zh: '训练方法' },
	layout: { en: 'Layout', zh: '布局' },
	rows: { en: 'Rows', zh: '列表' },
	pictures: { en: 'Pictures', zh: '图卡' },
	jumpTo: { en: 'Jump to', zh: '跳到' },
	noMatch: { en: 'Nothing matches “{q}”.', zh: '没有与“{q}”匹配的内容。' },
	clearFilters: { en: 'Clear filters', zh: '清除筛选' },
	usedIn: { en: 'Used in', zh: '用在' },
	related: { en: 'Related terms', zh: '相关术语' },
	willShow: { en: 'What this lesson will show', zh: '这节课将展示什么' },
	liveFirst: { en: 'Before this, try', zh: '可以先学' },
	notFoundBody: { en: 'This page is not here. Search, or start from the lessons.', zh: '找不到这个页面。可以搜索，或者从课程开始。' }
} as const;

export type CatKey = keyof typeof CAT;

/** A catalog string in the current language, with {placeholders} filled (reactive like t()). */
export function ct(key: CatKey, vars?: Record<string, string | number>): string {
	let s = t(CAT[key]);
	if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
	return s;
}
