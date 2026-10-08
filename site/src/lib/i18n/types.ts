/** Language codes used on the site. English is the default and must always be complete. */
export type Lang = 'en' | 'zh';

/** A bilingual string. `zh` may be missing only for long article text (then English is shown). */
export interface I18n {
	en: string;
	zh?: string;
}

/** Anything a component accepts as a label: a plain string (language-neutral) or a bilingual pair. */
export type Text = string | I18n;

/** Pick the string for a language (falls back to English). Pure function, no reactivity. */
export function pick(text: Text | undefined | null, lang: Lang): string {
	if (text == null) return '';
	if (typeof text === 'string') return text;
	return (lang === 'zh' ? text.zh : undefined) ?? text.en;
}
