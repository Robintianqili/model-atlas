"""Which characters does JavaScript (V8) classify differently from HF tokenizers' BERT normalizer /
pre-tokenizer? Writes bert_unicode_exceptions.json (used by realtok.ts)."""
import json
from tokenizers import Tokenizer
b = Tokenizer.from_file('raw/bert-base-uncased-tokenizer.json')
js = json.load(open('js_unicode_sets.json'))
pre = b.pre_tokenizer; norm = b.normalizer
not_punct = [c for c in js['P'] if len(pre.pre_tokenize_str('a' + chr(c) + 'b')) != 3]
# Mn: stripped by HF (normalize 'a'+mark -> 'a')?
not_mn = [c for c in js['Mn'] if norm.normalize_str('a' + chr(c)) != 'a']
# C (Cc/Cf/Co/Cs except private use planes): removed by HF clean_text?
not_c = [c for c in js['C'] if norm.normalize_str('a' + chr(c) + 'b') not in ('ab', 'a b')]
# and the other direction for punctuation: HF isolates but JS does not (scan all assigned BMP+SMP)
import unicodedata
extra_punct = []
jsP = set(js['P'])
for c in range(0x20, 0x30000):
    if 0xD800 <= c <= 0xDFFF or c in jsP: continue
    if 33 <= c <= 47 or 58 <= c <= 64 or 91 <= c <= 96 or 123 <= c <= 126: continue
    if len(pre.pre_tokenize_str('a' + chr(c) + 'b')) == 3 and not chr(c).isspace():
        extra_punct.append(c)
res = {'notPunct': not_punct, 'notMn': not_mn, 'notControl': not_c, 'extraPunct': extra_punct}
json.dump(res, open('bert_unicode_exceptions.json', 'w'))
for k, v in res.items(): print(k, len(v), [hex(x) for x in v[:40]])
