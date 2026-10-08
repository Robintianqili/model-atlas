"""Build the BPE parity cases: 300 test strings with the ids the Python tokenizer gives, plus
token displays, decode cases and small training cases. pipeline/parity/bpe.parity.ts checks the
TypeScript tokenizer against this file.

Run: python pipeline/parity/make_bpe_cases.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tinygpt"))
import bpe  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
TOK = os.path.join(ROOT, "site", "static", "models", "tinygpt", "tokenizer.json")
VAL = os.path.join(ROOT, "pipeline", "data", "tinystories", "TinyStoriesV2-GPT4-valid.txt")

ZWJ, VS16, NBSP, IDSP, COMB_ACUTE = chr(0x200D), chr(0xFE0F), chr(0xA0), chr(0x3000), chr(0x301)


def cases():
    rnd = random.Random(20261005)
    out = []
    # 1) TinyStories snippets (random cuts, also mid-word) ------------------------------- 100
    with open(VAL, encoding="utf-8") as f:
        text = f.read(3_000_000)
    for _ in range(100):
        a = rnd.randrange(0, len(text) - 400)
        out.append(text[a:a + rnd.randrange(1, 300)])
    # 2) Chinese and Chinese + English ------------------------------------------------------ 40
    zh = [
        "我爱自然语言处理 unbelievable!", "从前有一只小猫。", "今天天气很好，我们去公园玩吧！",
        "小明说：“你好！”", "模型图鉴", "机器学习和深度学习", "我有3个苹果和12个橘子。",
        "这是一个测试" + IDSP + "全角空格", "中文，标点。符号！问号？", "「引号」『双引号』",
        "价格：￥99.9元", "１２３４５全角数字", "Once upon a time 从前", "Lily 和 Tom 是好朋友",
        "你好世界", "一二三四五六七八九十", "龍飛鳳舞", "繁體中文測試", "简体中文测试", "人工智能",
        "注意力机制（attention）", "词元化 tokenization", "下一个词预测", "微调与预训练",
        "我爱北京天安门", "小猫坐在垫子上。", "她非常开心。", "太阳是红色的", "一天，汤姆去了公园。",
        "他们一起玩耍。\n第二天", "大家好！\t欢迎", "1984年", "第10章", "百分之50%", "温度：-5°C",
        "我。", "猫", "  前导空格中文", "结尾空格  ", "混合mixed文字text",
    ]
    out += zh
    # 3) Emoji -------------------------------------------------------------------------- 30
    fam = chr(0x1F468) + ZWJ + chr(0x1F469) + ZWJ + chr(0x1F467)
    emo = [
        chr(0x1F600), "Hi " + chr(0x1F600) + "!", fam, "family: " + fam + " ok",
        chr(0x1F44D) + chr(0x1F3FD), "thumbs " + chr(0x1F44D) + chr(0x1F3FB) + chr(0x1F44D) + chr(0x1F3FF),
        chr(0x1F1E8) + chr(0x1F1F3) + chr(0x1F1FA) + chr(0x1F1F8), "1" + VS16 + chr(0x20E3),
        chr(0x2764) + VS16, "I " + chr(0x2764) + VS16 + " cats " + chr(0x1F431),
        chr(0x1F431) * 5, chr(0x1F680) + " launch " + chr(0x1F680) + chr(0x1F680),
        "smile" + chr(0x1F642) + "smile", chr(0x1F9D1) + ZWJ + chr(0x1F4BB), chr(0x1F3F3) + VS16 + ZWJ + chr(0x1F308),
        "end." + chr(0x1F389), chr(0x2600) + chr(0x2601) + chr(0x2602), chr(0x1F525) + " fire",
        chr(0x1F4AF) + "%", "a" + chr(0x1F600) + "b" + chr(0x1F600) + "c", chr(0x1F43C) + "熊猫",
        chr(0x1F600) + "\n" + chr(0x1F600), " " + chr(0x1F600), chr(0x1F600) + "  ",
        chr(0x2B50) + chr(0x2B50) + chr(0x2B50), "(" + chr(0x1F600) + ")", "#" + VS16 + chr(0x20E3),
        chr(0x1FAE0), chr(0x1F468) + chr(0x1F3FE) + ZWJ + chr(0x1F680), "ok " + chr(0x2705),
    ]
    out += emo
    # 4) Numbers ------------------------------------------------------------------------ 30
    nums = [
        "3.14159", "1,000,000", "2026-10-05", "$12.50", "100%", "-42", "+7", "1e-5", "0x1F",
        "007", "12345678901234567890", "3 apples and 12 oranges", "v2.0.1", "1/2", "½ cup",
        "x² + y³", "Ⅻ o'clock", "١٢٣٤", "٣ apples", "१२३", "page 5 of 10", "1st 2nd 3rd 4th",
        "9:30 am", "50km/h", "(555) 123-4567", "A1B2C3", "  42  ", "1.5e+10", "π ≈ 3.14", "∞",
    ]
    out += nums
    # 5) Punctuation and contractions --------------------------------------------------- 30
    punct = [
        "!!!???", "...", "Wait... what?!", "—", "“quoted”", "‘single’", "'s 'll 've 're 'd 'm 't",
        "can't won't I'm you're we've they'd she'll", "I'M YOU'RE CAN'T", "((a))", "[x]{y}<z>",
        "a,b;c:d", "Tom's dog's ball", "rock 'n' roll", "'hello'", "\"Hi,\" she said.",
        "<|endoftext|>", "The end.<|endoftext|>Once upon", "<|endoftext|><|endoftext|>",
        "<|endoftext", "endoftext|>", "@user #tag", "a&b|c", "~/path/to/file.txt",
        "foo_bar-baz", "*bold* _it_", "50% off!!!", "¿Qué? ¡Sí!", "«guillemets»", "…",
    ]
    out += punct
    # 6) Whitespace runs ---------------------------------------------------------------- 30
    ws = [
        "  multiple   spaces  ", "\t\ttabs\there", "\n\n\nnewlines", "trailing   ", "   leading",
        "line1\r\nline2", "a" + IDSP + "b", "a" + NBSP + "b", "     ", "", " ", "\n", "\t", " \n ",
        "word \n word", "a  b   c    d", "\n \n \n", "x y".replace("\\u0020", " "), "tab\t \tmix",
        "end\n", "\r", "  \t  ", "a\fb", "a\vb", "hello" + chr(0x2028) + "world", "one" + chr(0x2003) + "two",
        "zero" + chr(0x200B) + "width", "bom" + chr(0xFEFF) + "x", "nbsp" + NBSP * 3 + "x", "Lily said,\n\n\"Hi!\"",
    ]
    out += ws
    # 7) Other scripts, combining marks, code ------------------------------------------- 30
    other = [
        "こんにちは世界", "カタカナ", "안녕하세요 세계", "مرحبا بالعالم", "नमस्ते दुनिया", "Привет, мир!",
        "Γειά σου Κόσμε", "สวัสดีชาวโลก", "שלום עולם", "e" + COMB_ACUTE + "cole", "café naïve",
        "Straße", "ǅungla", "İstanbul", "∑∫√∂", "for (i=0;i<10;i++) {x+=1;}", "https://example.com/a?b=c",
        "name@example.com", "def f(x):\n    return x*2", "<div class=\"a\">hi</div>", "ﬁ ligature",
        "Ａ全角Ｂ", "ｶﾀｶﾅ", "Ⓐ Ⓑ", "ⅷ", "a" + chr(0x0308) + chr(0x0301), "ʻokina", "Ελληνικά", "ქართული",
        "Tiếng Việt",
    ]
    out += other
    # 8) Random code points from many ranges -------------------------------------------- 10
    ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x4E00, 0x9FFF),
              (0x3040, 0x30FF), (0xAC00, 0xD7A3), (0x1F300, 0x1F6FF), (0x2000, 0x206F), (0x0E00, 0x0E7F)]
    for _ in range(10):
        s = []
        for _ in range(rnd.randrange(5, 40)):
            lo, hi = rnd.choice(ranges)
            s.append(chr(rnd.randrange(lo, hi + 1)))
        out.append("".join(s))
    return out


def main():
    tok = bpe.BPETokenizer.load(TOK)
    strings = cases()
    assert len(strings) == 300, len(strings)
    rows = []
    for s in strings:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, repr(s)  # byte-level BPE is lossless
        pts = []
        for piece in bpe.split_special(s):
            pts += [piece] if piece in tok.special else bpe.pretokenize(piece)
        rows.append({"text": s, "ids": ids, "pretokens": pts})
    steps = tok.encode_with_steps("Once upon a time, 我爱 " + chr(0x1F600) + " cats!<|endoftext|>")
    rnd = random.Random(7)
    dec = []
    for _ in range(100):
        ids = [rnd.randrange(0, tok.vocab_size) for _ in range(rnd.randrange(1, 12))]
        dec.append({"ids": ids, "text": tok.decode(ids)})
    train_cases = []
    for text in [strings[0] + strings[1] + strings[2], "low lower lowest newer newest wider " * 5,
                 "我爱自然语言处理。我爱机器学习。" * 3 + "aaa aaaa aaaaa"]:
        merges = bpe.train_bpe(__import__("collections").Counter(bpe.pretokenize(text)), 40)
        train_cases.append({"text": text, "numMerges": 40, "merges": [[a, b] for a, b, _ in merges],
                            "counts": [c for _, _, c in merges]})
    out = {
        "tokenizer": "site/static/models/tinygpt/tokenizer.json",
        "cases": rows,
        "display": [tok.token_to_display(i) for i in range(tok.vocab_size)],
        "encodeWithSteps": steps,
        "decode": dec,
        "train": train_cases,
    }
    path = os.path.join(HERE, "bpe_cases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n_tok = sum(len(r["ids"]) for r in rows)
    print(f"wrote {path}: {len(rows)} strings, {n_tok} tokens, {len(dec)} decode cases, {len(train_cases)} train cases")


if __name__ == "__main__":
    main()
