"""Reference token ids for the tokenization page's JS tokenizers (pipeline/tokenizers/parity.ts).

References:
  gpt2         tiktoken "gpt2" encode_ordinary   (also compared with HF openai-community/gpt2)
  cl100k_base  tiktoken "cl100k_base" encode_ordinary
  qwen2.5      HF tokenizers, Qwen/Qwen2.5-0.5B tokenizer.json, add_special_tokens=False
  bert         HF tokenizers, google-bert/bert-base-uncased tokenizer.json (adds [CLS] ... [SEP])
Writes pipeline/tokenizers/parity_cases.json.
"""
import os, json, random
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
import tiktoken
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'raw')
C = chr  # build unusual characters from code points (tool layers rewrite backslash-u escapes)

cases = [
    # the page's own presets and examples
    '我爱自然语言处理 unbelievable!', '我爱自然语言处理', '我爱你', 'unbelievable', ' unbelievable', 'Unbelievable!!!',
    'The weather is nice today, so let\'s go to the park.', '今天天气很好，我们去公园吧。',
    '今日は天気がいいので、公園に行きましょう。', '오늘은 날씨가 좋으니 공원에 가자.',
    'Hoy hace buen tiempo, así que vamos al parque.', "Il fait beau aujourd'hui, alors allons au parc.",
    'Heute ist schönes Wetter, also lass uns in den Park gehen.', 'Сегодня хорошая погода, давай пойдём в парк.',
    'الطقس جميل اليوم، فلنذهب إلى الحديقة.', 'आज मौसम अच्छा है, चलो पार्क चलते हैं।',
    'the The THE tHe', ' the', 'the', 'hello world', 'Hello, World!', 'HELLO WORLD',
    # numbers
    '12345678 + 3.14159 = ?', 'In 2024, 1,000,000 people paid $3.50.', '0.000001', '9999999999', '٣٤٥ ٣٤٥', '①②③ ⅫⅬ',
    # contractions and apostrophes
    "I'm sure it's fine, isn't it? We'll see; they'd've known.", "IT'S DON'T WE'LL I'M", "it’s curly",
    "it'" + C(0x17F) + " long-s", "rock 'n' roll", "''''", "'s's 't",
    # whitespace
    'a  b   c    d', '  leading and trailing  ', 'tab\there', 'line1\nline2\n\nline4', 'win\r\nline', 'end with newline\n',
    'end with spaces   ', 'newline then spaces\n   ', '\n\n\n', '   ', ' ', 'x' + C(0x3000) + 'y' + C(0xA0) + 'z',
    'a' + C(0x2028) + 'b' + C(0x2029) + 'c', 'nel' + C(0x85) + 'here', 'bom' + C(0xFEFF) + 'mid', 'zw' + C(0x200B) + 'sp',
    'vt' + C(0x0B) + 'ff' + C(0x0C) + 'x', 'fs' + C(0x1C) + 'gs' + C(0x1D) + 'x', 'tabs\t\t\tthen text', '\t\n \n x',
    # emoji and symbols
    'Hello world 🙂', 'I ❤️ 🦙🦙 llamas!', '👩‍👩‍👧‍👦 family', '🇨🇳🇺🇸🇯🇵', '👍🏽 ok', '🥲🫠🪿', '∑ x² ≤ ∞ → ∀ε>0', '©®™ ✓ ✗',
    # accents and normalization
    'naïve café résumé', 'nai' + C(0x308) + 've cafe' + C(0x301), 'İstanbul ΟΔΥΣΣΕΥΣ straße', 'Ångström Æsir Øresund',
    'ﬁne ﬂow', 'Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ', 'ｶﾀｶﾅ',
    # code
    'def add(a, b):\n    return a + b\n', 'for (let i = 0; i < n; i++) { sum += x[i]; }', '<div class="x">&nbsp;</div>',
    'https://example.com/path?q=1&r=two#frag', 'snake_case camelCase PascalCase kebab-case', '{"key": [1, 2, 3]}',
    '    indented\n        more', '#include <stdio.h>', 'x=1;y=2;z=x+y', 'SELECT * FROM t WHERE id = 42;',
    # mixed scripts
    'GPT-4和Qwen2.5都能读中文。', 'BERT的词表里有“我”吗？', 'Transformer模型（2017）', '繁體中文與简体中文',
    '𠀀𠀁 rare CJK ext B', 'ㄅㄆㄇ 注音', 'ภาษาไทย', 'עברית', 'Ελληνικά', 'ქართული', 'தமிழ்', 'አማርኛ',
    # BERT specific
    'supercalifragilisticexpialidocious', 'antidisestablishmentarianism', 'tokenization tokenizer tokenized',
    'x' * 101, 'a' * 100, 'Mr. O\'Neil-Smith (the 3rd)', '...?!...', '¿Qué? ¡Sí!', '«quotes» „more“', '—dash–dash-',
    'ID: 100,101,102', 'café' * 30,
    # repeats (long pre-tokens)
    ' ' * 50 + 'x', '-' * 70, '=' * 200, 'ha' * 60, '\n' * 20 + 'x', '0123456789' * 5,
    # misc
    '', 'a', '.', 'é', '中', '😀', 'abc\x00def', 'replacement ' + C(0xFFFD) + ' char',
]

# deterministic fuzz: random code points from a few blocks
random.seed(20261006)
blocks = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x4E00, 0x9FFF), (0x3040, 0x30FF),
          (0xAC00, 0xD7A3), (0x1F300, 0x1F64F), (0x2000, 0x206F), (0x0600, 0x06FF), (0x0900, 0x097F)]
for _ in range(400):
    n = random.randint(1, 40)
    s = ''
    for _ in range(n):
        a, b = random.choice(blocks)
        s += C(random.randint(a, b)) if random.random() < 0.8 else random.choice([' ', '  ', '\n', "'s", '123', '\t'])
    cases.append(s)

# English paragraphs (TinyStories validation split, never shipped)
story = open(os.path.join(HERE, '..', 'data', 'tinystories', 'TinyStoriesV2-GPT4-valid.txt'), encoding='utf8').read()
for _ in range(60):
    i = random.randint(0, len(story) - 1000)
    cases.append(story[i:i + random.randint(100, 900)])

enc_g = tiktoken.get_encoding('gpt2')
enc_c = tiktoken.get_encoding('cl100k_base')
hf_g = Tokenizer.from_file(f'{R}/gpt2/tokenizer.json')
hf_q = Tokenizer.from_file(f'{R}/qwen2.5/tokenizer.json')
hf_b = Tokenizer.from_file(f'{R}/bert-base-uncased-tokenizer.json')
# The page treats typed text as ordinary text: "<|endoftext|>" or "[SEP]" typed by the learner are
# split like any other characters (same as tiktoken's encode_ordinary).
for t in (hf_g, hf_q, hf_b):
    t.encode_special_tokens = True

out = []
gpt2_ref_disagree = []
for s in cases:
    g = enc_g.encode_ordinary(s)
    hg = hf_g.encode(s, add_special_tokens=False).ids
    if g != hg:
        gpt2_ref_disagree.append(s)
    out.append({
        'text': s,
        'gpt2': g,
        'gpt2_hf': hg,
        'cl100k_base': enc_c.encode_ordinary(s),
        'qwen2.5': hf_q.encode(s, add_special_tokens=False).ids,
        'bert-base-uncased': hf_b.encode(s).ids,
    })
json.dump({'cases': out, 'gpt2_tiktoken_vs_hf_disagree': gpt2_ref_disagree,
           'versions': {'tiktoken': tiktoken.__version__, 'tokenizers': __import__('tokenizers').__version__}},
          open(os.path.join(HERE, 'parity_cases.json'), 'w', encoding='utf8'), ensure_ascii=False)
print('cases', len(out), 'gpt2 tiktoken vs HF disagreements:', len(gpt2_ref_disagree), [repr(x) for x in gpt2_ref_disagree][:5])
