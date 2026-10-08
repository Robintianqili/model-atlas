/** Input presets for the TinyGPT page: English story openings (the model's world) plus two
 *  "out of distribution" inputs that show byte-level fallback. */
export const PRESETS: { text: string; label?: { en: string; zh: string } }[] = [
	{ text: 'Once upon a time, there was a little' },
	{ text: 'The cat sat on the' },
	{ text: 'Tom and his dog went to the park. They' },
	{ text: 'Lily was sad because she lost her' },
	{ text: 'One day, a girl named Mia found a big box. She opened it and saw' },
	{ text: '我爱自然语言处理', label: { en: 'Chinese: 我爱自然语言处理', zh: '中文：我爱自然语言处理' } },
	{ text: 'I love 🍎 and 🐶', label: { en: 'Emoji', zh: 'Emoji 表情' } }
];

export const DEFAULT_TEXT = 'Once upon a time, there was a little';
