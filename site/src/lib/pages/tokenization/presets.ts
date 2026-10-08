/**
 * Texts for the tokenization page. All written for this page (no third-party text), so they can be
 * shipped freely. The Chinese story is a translation of the English one, so the two can be compared.
 */
import type { I18n } from '../../i18n/types.ts';

export type TrainPreset = { id: string; label: I18n; text: string };
export type LangSentence = { code: string; name: I18n; text: string };

const STORY_EN = `Once upon a time, there was a little cat named Tom. Tom liked to sit in the sun and watch the birds. One day, a small bird landed on the fence. "Hello," said the bird. "Do you want to play?" Tom was happy. They played in the garden all day. They ran around the tree, they jumped over the flowers, and they laughed at the clouds. When the sun went down, the bird said, "I have to go home now." Tom was sad, but the bird said, "I will come back tomorrow." The next day, the bird came back, and the two friends played again. Tom learned that a good friend always comes back.`;

const STORY_ZH = `从前，有一只小猫叫汤姆。汤姆喜欢坐在太阳下看小鸟。有一天，一只小鸟落在篱笆上。小鸟说：“你好！你想和我一起玩吗？”汤姆很高兴。它们在花园里玩了一整天。它们绕着大树跑，跳过花丛，对着白云笑。太阳下山的时候，小鸟说：“我要回家了。”汤姆很难过，可是小鸟说：“我明天还会回来。”第二天，小鸟真的回来了，两个好朋友又一起玩。汤姆明白了：好朋友总会回来的。`;

const CODE = `def count_pairs(words):
    counts = {}
    for word in words:
        for a, b in zip(word, word[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def merge(word, pair, new_token):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(new_token)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out
`;

/** The worked example from Sennrich, Haddow & Birch (2016): low ×5, lower ×2, newest ×6, widest ×3. */
const CLASSIC = 'low low low low low lower lower newest newest newest newest newest newest widest widest widest';

export const PRESETS: { train: TrainPreset[]; sentence: { text: string; label?: I18n }[]; languages: LangSentence[] } = {
	train: [
		{ id: 'story-en', label: { en: 'English story', zh: '英文故事' }, text: STORY_EN },
		{ id: 'story-zh', label: { en: 'Same story in Chinese', zh: '同一个故事（中文）' }, text: STORY_ZH },
		{ id: 'code', label: { en: 'Python code', zh: 'Python 代码' }, text: CODE },
		{ id: 'classic', label: { en: 'low / lower / newest / widest', zh: 'low / lower / newest / widest' }, text: CLASSIC }
	],
	sentence: [
		{ text: '我爱自然语言处理 unbelievable!', label: { en: '我爱自然语言处理 unbelievable!', zh: '我爱自然语言处理 unbelievable!' } },
		{ text: "The weather is nice today, so let's go to the park.", label: { en: 'English', zh: '英文' } },
		{ text: '今天天气很好，我们去公园吧。', label: { en: 'Chinese', zh: '中文' } },
		{ text: 'Once upon a time, there was a little cat named Tom.', label: { en: 'Story (TinyGPT style)', zh: '故事（TinyGPT 风格）' } },
		{ text: 'In 2024, 12345678 + 3.14159 = ?', label: { en: 'Numbers', zh: '数字' } },
		{ text: 'I ❤️ 🦙🦙 llamas! 👩‍👩‍👧', label: { en: 'Emoji', zh: '表情符号' } },
		{ text: 'the The THE  the', label: { en: 'Case and spaces', zh: '大小写与空格' } },
		{ text: 'for i in range(10): print(i ** 2)', label: { en: 'Code', zh: '代码' } }
	],
	languages: [
		{ code: 'en', name: { en: 'English', zh: '英语' }, text: "The weather is nice today, so let's go to the park." },
		{ code: 'zh', name: { en: 'Chinese', zh: '中文' }, text: '今天天气很好，我们去公园吧。' },
		{ code: 'ja', name: { en: 'Japanese', zh: '日语' }, text: '今日は天気がいいので、公園に行きましょう。' },
		{ code: 'ko', name: { en: 'Korean', zh: '韩语' }, text: '오늘은 날씨가 좋으니 공원에 가자.' },
		{ code: 'es', name: { en: 'Spanish', zh: '西班牙语' }, text: 'Hoy hace buen tiempo, así que vamos al parque.' },
		{ code: 'fr', name: { en: 'French', zh: '法语' }, text: "Il fait beau aujourd'hui, alors allons au parc." },
		{ code: 'de', name: { en: 'German', zh: '德语' }, text: 'Heute ist schönes Wetter, also lass uns in den Park gehen.' },
		{ code: 'ru', name: { en: 'Russian', zh: '俄语' }, text: 'Сегодня хорошая погода, давай пойдём в парк.' },
		{ code: 'ar', name: { en: 'Arabic', zh: '阿拉伯语' }, text: 'الطقس جميل اليوم، فلنذهب إلى الحديقة.' },
		{ code: 'hi', name: { en: 'Hindi', zh: '印地语' }, text: 'आज मौसम अच्छा है, चलो पार्क चलते हैं।' }
	]
};

export const DEFAULT_SENTENCE = PRESETS.sentence[0].text;
