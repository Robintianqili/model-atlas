"""Turn the research workflow's design brief (viz_brief.md) into one readable HTML page for the Artifact."""
import json
import os
import re

import markdown

SP = os.path.dirname(os.path.abspath(__file__))
md_text = open(os.path.join(SP, "design_brief.md"), encoding="utf-8").read()
research = json.load(open(os.path.join(SP, "research_raw.json"), encoding="utf-8"))

# drop the brief's own H1 and date lines; the page header replaces them
body_md = md_text.split("\n## 1. Landscape", 1)
intro_md, rest_md = body_md[0], "## 1. Landscape" + body_md[1]
summary_md = intro_md.split("**Summary**", 1)[1].split("\n---", 1)[0]

conv = lambda s: markdown.markdown(s, extensions=["tables", "fenced_code", "sane_lists", "toc"],
                                   extension_configs={"toc": {"slugify": lambda v, sep: re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")}})
summary_html = conv(summary_md)
main_html = conv(rest_md)

# wrap every table so wide ones scroll inside their own box
main_html = main_html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
main_html = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', main_html)
# give the landscape table an id so the filter can find it
main_html = main_html.replace('<div class="tw"><table>', '<div class="tw" id="landscape"><table>', 1)

toc = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', main_html)
toc_html = "".join(f'<li><a href="#{i}">{re.sub("<[^>]+>", "", t)}</a></li>' for i, t in toc)

n_cand = research["counts"]["candidates"]
n_deep = research["counts"]["deep_read"]

page = f"""<title>Model Atlas Research Brief</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;600&display=swap">
<style>
/* Layout: a field guide. Sticky contents rail on wide screens, one reading column, wide tables scroll in their own boxes. */
:root {{
  --paper: #f6f7f9; --panel: #ffffff; --ink: #18202b; --muted: #5b6573; --rule: #dde2e8;
  --neg: #2f6fbf; --pos: #d9772b; --seq: #6b4fb3; --mark: #fff3d6;
  --display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --body: "Source Serif 4", Georgia, "Noto Sans SC", serif;
  --zh: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #11151b; --panel: #171c24; --ink: #e6e9ee; --muted: #9aa4b2; --rule: #2a313c;
  --neg: #6aa6ec; --pos: #f0a060; --seq: #a68ae6; --mark: #3a2f16; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --paper: #11151b; --panel: #171c24; --ink: #e6e9ee; --muted: #9aa4b2; --rule: #2a313c;
  --neg: #6aa6ec; --pos: #f0a060; --seq: #a68ae6; --mark: #3a2f16; color-scheme: dark }}
html {{ scroll-behavior: smooth }}
@media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto }} }}
body {{ background: var(--paper); color: var(--ink); font: 16px/1.6 var(--body); }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 80px;
  display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 40px }}
@media (max-width: 1000px) {{ .wrap {{ grid-template-columns: minmax(0, 1fr); gap: 20px; padding-inline: 16px }} }}
nav.toc {{ position: sticky; top: calc(env(safe-area-inset-top, 0px) + 20px); align-self: start;
  font: 500 13px/1.4 var(--display); max-height: calc(100vh - 40px); overflow: auto }}
nav.toc ol {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 2px }}
nav.toc a {{ display: block; padding: 6px 10px; color: var(--muted); text-decoration: none; border-radius: 6px }}
nav.toc a:hover, nav.toc a:focus-visible {{ color: var(--ink); background: var(--panel) }}
nav.toc .lbl {{ font: 600 11px/1 var(--display); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 0 10px 10px }}
@media (max-width: 1000px) {{ nav.toc {{ position: static; max-height: none }} nav.toc details {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 10px }} }}
main {{ min-width: 0 }}
header.top {{ display: grid; gap: 14px; margin-bottom: 28px }}
.eyebrow {{ font: 600 12px/1 var(--display); letter-spacing: .14em; text-transform: uppercase; color: var(--muted) }}
h1 {{ font: 800 clamp(30px, 5vw, 46px)/1.05 var(--display); font-stretch: 80%; margin: 0; text-wrap: balance }}
.dek {{ font-size: 18px; color: var(--muted); max-width: 64ch; margin: 0 }}
.scale {{ display: flex; align-items: center; gap: 10px; font: 12px var(--mono); color: var(--muted); flex-wrap: wrap }}
.scale .bar {{ width: 180px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg, var(--neg), color-mix(in oklab, var(--neg) 20%, var(--paper)) 45%, color-mix(in oklab, var(--pos) 20%, var(--paper)) 55%, var(--pos)) }}
.scale .seq {{ width: 90px; height: 10px; border-radius: 2px; background: linear-gradient(90deg, color-mix(in oklab, var(--seq) 8%, var(--paper)), var(--seq)) }}
.facts {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; margin-top: 6px }}
@media (max-width: 640px) {{ .facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)) }} }}
.facts div {{ background: var(--panel); padding: 12px 14px }}
.facts b {{ display: block; font: 700 26px/1.1 var(--display); font-variant-numeric: tabular-nums }}
.facts span {{ font: 13px/1.3 var(--display); color: var(--muted) }}
section.zh {{ font-family: var(--zh); background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 18px 20px; margin-bottom: 18px }}
section.zh h2 {{ font: 700 18px/1.3 var(--zh); margin: 0 0 8px }}
section.zh ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px }}
section.summary {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 18px 20px; margin-bottom: 36px }}
section.summary h2 {{ font: 700 18px/1.3 var(--display); margin: 0 0 8px }}
section.summary ul {{ margin: 0; padding-left: 1.2em }}
.content {{ max-width: 100% }}
.content p, .content li {{ max-width: 76ch }}
.content h2 {{ font: 800 28px/1.15 var(--display); font-stretch: 85%; margin: 56px 0 14px; padding-top: 18px; border-top: 2px solid var(--ink); text-wrap: balance; scroll-margin-top: 16px }}
.content h3 {{ font: 700 19px/1.3 var(--display); margin: 32px 0 10px; text-wrap: balance; scroll-margin-top: 16px }}
.content a {{ color: var(--neg); text-underline-offset: 2px }}
.content code {{ font: 0.86em var(--mono); background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 0 4px }}
.content pre {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; overflow-x: auto; font: 13px/1.5 var(--mono) }}
.content pre code {{ border: 0; background: none; padding: 0 }}
.content strong {{ font-weight: 600 }}
.tw {{ overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--panel); margin: 14px 0 22px }}
.tw table {{ border-collapse: collapse; font: 13.5px/1.45 var(--body); min-width: 760px; width: 100% }}
.tw th {{ position: sticky; top: 0; background: var(--panel); text-align: left; font: 600 12px/1.3 var(--display); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--rule); padding: 9px 10px; vertical-align: bottom }}
.tw td {{ border-top: 1px solid var(--rule); padding: 8px 10px; vertical-align: top }}
.tw tr.hit td {{ background: var(--mark) }}
#landscape table {{ min-width: 1200px }}
.filter {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 6px 0 -4px; font: 14px var(--display) }}
.filter input {{ font: 15px var(--display); padding: 7px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); color: var(--ink); width: min(340px, 100%) }}
.filter input:focus-visible, a:focus-visible {{ outline: 2px solid var(--seq); outline-offset: 2px }}
.filter output {{ color: var(--muted); font-variant-numeric: tabular-nums }}
footer {{ margin-top: 60px; color: var(--muted); font: 13px/1.5 var(--display) }}
</style>

<div class="wrap">
  <nav class="toc" aria-label="Contents">
    <details open id="tocbox"><summary class="lbl">Contents</summary>
      <ol><li><a href="#zh">中文摘要</a></li><li><a href="#summary">Summary</a></li>{toc_html}</ol>
    </details>
  </nav>
  <main>
    <header class="top">
      <div class="eyebrow">Research brief · 2026-10-05 · revision 2</div>
      <h1>Model Atlas: one site to play with every kind of model</h1>
      <p class="dek">What already exists, what the best explainers do, what nobody does yet, and a plan for one integrated site: models from 11 application areas, each animated layer by layer on your own input, plus a track on pre-training and fine-tuning methods.</p>
      <div class="scale" aria-label="The planned color grammar"><span>negative</span><span class="bar"></span><span>positive</span><span style="margin-left:14px">0</span><span class="seq"></span><span>probability</span></div>
      <div class="facts">
        <div><b>{n_cand}</b><span>projects found by 10 searches</span></div>
        <div><b>{n_deep}</b><span>read in depth (code, demo, license)</span></div>
        <div><b>11</b><span>application areas in the plan</span></div>
        <div><b>4 + 3</b><span>models + concept pages in the first version</span></div>
      </div>
    </header>

    <section class="zh" id="zh" lang="zh-Hans">
      <h2>中文摘要</h2>
      <ul>
        <li><b>现状：</b>没有一个网站同时做到“覆盖所有领域 + 一套统一画法 + 用你自己的输入 + 训练方法讲解”。Polo Club 的 CNN / Transformer / Diffusion Explainer 和 GAN Lab 质量最高，但它们是各自独立的网站；Google 的 LIT 最接近“一个界面、很多模型、自己的数据”，但需要 Python 服务器，面向专业人员，也没有逐层动画。</li>
        <li><b>我们的不同：</b>11 个应用领域用同一套视觉语言；可以玩的“训练概念”板块（预训练、微调、LoRA、对齐、量化……）；每个画面标明数字是“浏览器实时算的 / 离线预计算的 / 示意图”；中英双语；纯静态网站，不依赖服务器。</li>
        <li><b>技术路线：</b>自己用一块 GPU 训练很小的模型（≤1 MB），在浏览器里真实计算每一个数字；GPT-2、BERT、CLIP 这类大模型离线算好结果再展示。模型文件不放进 git。GitHub Pages 整站上限 1 GB。</li>
        <li><b>第一版（MVP）：</b>4 个模型：TinyCNN（画图或上传图片识别）、TinyGPT（每个数字都实时计算）、语音关键词识别（用你自己的声音）、2D flow matching 生成模型（画一个分布，现场训练）；3 个概念页：分词、下一个词预测、迁移学习和微调。</li>
        <li><b>v1：</b>11 个领域各一个旗舰模型，加约 11 个概念页。其余列为“计划中”，等共用组件成熟后再做。我们的 BERT 复现（MNLI → RTE、学习率搜索等）可以直接做成“多任务训练”和“微调”概念页。</li>
      </ul>
    </section>

    <section class="summary" id="summary">
      <h2>Summary</h2>
      {summary_html}
    </section>

    <div class="content">
      {main_html}
    </div>
    <footer>Made from a research run with 19 agents: 10 web sweeps, 20 deep reads, a synthesis, a skeptical review and a revision. Facts marked “unverified” could not be confirmed; “design target” marks our own plan.</footer>
  </main>
</div>

<script>
(function () {{
  // landscape filter
  var box = document.getElementById('landscape');
  if (box) {{
    var bar = document.createElement('div'); bar.className = 'filter';
    bar.innerHTML = '<label for="lf">Filter the landscape</label><input id="lf" type="search" placeholder="e.g. audio, LoRA, in-browser, MIT"><output id="lfn" aria-live="polite"></output>';
    box.parentNode.insertBefore(bar, box);
    var rows = box.querySelectorAll('tbody tr'), inp = bar.querySelector('input'), out = bar.querySelector('output');
    var run = function () {{
      var q = inp.value.trim().toLowerCase(), n = 0;
      rows.forEach(function (r) {{ var ok = !q || r.textContent.toLowerCase().indexOf(q) >= 0; r.hidden = !ok; if (ok) n++; }});
      out.textContent = q ? n + ' of ' + rows.length + ' projects' : rows.length + ' projects';
    }};
    inp.addEventListener('input', run); run();
  }}
  // collapse the contents list on narrow screens
  var d = document.getElementById('tocbox');
  if (d && window.matchMedia('(max-width: 1000px)').matches) d.open = false;
}})();
</script>
"""
out = os.path.join(SP, "model_atlas_brief.html")
open(out, "w", encoding="utf-8", newline="").write(page)
print(out, len(page), "chars;", len(toc), "sections")
