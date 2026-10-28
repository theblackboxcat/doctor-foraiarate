#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the **register evaluation set** for the detector: modernist prose and poetry.

Why a separate set from ``calib/``
----------------------------------
``calib/human_zh/`` holds one human draft (alley prose) plus two everyday-register
fillers. Every slot in the 8-slot literary taxonomy was designed against
*that* register. Modernist prose and poetry are the registers where the design
assumptions break:

* modernist prose repeats nouns on purpose (「一株是枣树，还有一株也是枣树」),
  chops clauses on purpose, and uses ultra-short nominal beats on purpose —
  exactly the shapes the "AI taste" slots fire on;
* poetry has line breaks instead of sentences, so every sentence-length and
  comma-density based slot measures something else entirely.

So this set exists to *measure the false-positive rate on those two registers*,
not to make the detector like them.

Copyright
---------
Chinese copyright is life + 50 years. Only authors who are safely public domain
are fetched here (all from zh.wikisource.org, converted to zh-hans by the
``variant`` parameter):

    鲁迅 (d. 1936) · 萧红 (d. 1942) · 徐志摩 (d. 1931)
    闻一多 (d. 1946) · 戴望舒 (d. 1950)

Living or in-copyright authors (格非 / 残雪 / 余华 / 双雪涛 …) are deliberately
absent. That is the single hardest constraint in this file: a missing sample is
better than a pirated one.

Usage
-----
    python eval_set/fetch_registers.py            # download + chunk + manifest
    python eval_set/fetch_registers.py --force    # refetch even if cached
"""
from __future__ import annotations

import argparse
import hashlib
import html as htmllib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
API = "https://zh.wikisource.org/w/api.php"
UA = "cowrite-register-eval/1.0 (local research; contact: local user)"

# register -> list of (page title on wikisource, short id)
SOURCES = {
    # 散文诗 / 现代主义散文 —— 鲁迅《野草》全编，逐篇取（目录页只有链接，没有正文）
    "modernist_prose": [
        ("秋夜_(魯迅)", "luxun_qiuye"),
        ("影的告别", "luxun_yinggaobie"),
        ("求乞者", "luxun_qiuqizhe"),
        ("我的失戀", "luxun_wodeshilian"),
        ("復讎", "luxun_fuchou"),
        ("復讎（其二）", "luxun_fuchou2"),
        ("希望", "luxun_xiwang"),
        ("風筝", "luxun_fengzheng"),
        ("好的故事", "luxun_haodegushi"),
        ("過客", "luxun_guoke"),
        ("死火", "luxun_sihuo"),
        ("狗的駁詰", "luxun_gou"),
        ("失掉的好地獄", "luxun_diyu"),
        ("墓碣文", "luxun_mujiewen"),
        ("頹敗線的顫動", "luxun_tuibai"),
        ("立論", "luxun_lilun"),
        ("死後", "luxun_sihou"),
        ("這樣的戰士", "luxun_zhanshi"),
        ("聰明人和儍子和奴才", "luxun_congmingren"),
        ("臘葉", "luxun_laye"),
        ("淡淡的血痕中", "luxun_xuehen"),
        ("一覺", "luxun_yijue"),
    ],
    # 诗 —— 戴望舒 / 徐志摩 / 闻一多
    "poetry": [
        ("雨巷", "dai_yuxiang"),
        ("煩憂", "dai_fanyou"),
        ("我用殘損的手掌", "dai_canshou"),
        ("再別康橋", "xu_zaibie"),
        ("偶然", "xu_ouran"),
        ("死水", "wen_sishui"),
        ("一句話", "wen_yijuhua"),
        ("也許", "wen_yexu"),
    ],
    # 早期白话叙事（现实主义）—— 对照组：不能因为改判现代主义就把这一类搞坏
    "realist_prose": [
        ("故鄉", "luxun_guxiang"),
        ("孔乙己", "luxun_kongyiji"),
        ("社戲", "luxun_shexi"),
        ("呼蘭河傳/第一章", "xiaohong_hulanhe_1"),
        ("呼蘭河傳/第二章", "xiaohong_hulanhe_2"),
    ],
}

MIN_CHUNK = 260          # shorter than this and the detector says "insufficient"
MAX_CHUNK = 900
TARGET_CHUNK = 620


def _api(params: dict, tries: int = 3) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise last if last else RuntimeError("api failed")


def fetch_html(title: str, force: bool = False) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^\w\u4e00-\u9fff]+", "_", title) + ".html"
    p = CACHE / key
    if p.is_file() and not force and p.stat().st_size > 500:
        return p.read_text(encoding="utf-8")
    data = _api({
        "action": "parse", "format": "json", "formatversion": "2",
        "prop": "text", "redirects": "1", "variant": "zh-hans",
        "page": title,
    })
    if "error" in data:
        raise RuntimeError("parse error: %s" % data["error"].get("info"))
    html = (data.get("parse") or {}).get("text") or ""
    if len(html) < 200:
        raise RuntimeError("empty parse result")
    p.write_text(html, encoding="utf-8")
    time.sleep(0.5)                      # be polite to wikisource
    return html


# Nav / licence / sister-project furniture that must not enter the corpus:
# it is *not* the author's prose, and it is the same string in every file,
# which would teach the calibrator a fake "human" pattern.
DROP_LINE = re.compile(
    r"^(←|→|《野草》题辞|本作品收录于.*|姊妹计划.*|数据项|作者：.*|"
    r"检索自.*|\[编辑\]|.*维基百科.*|公有领域.*)$"
)
DROP_TAIL = ("维基文库", "本作品", "Public domain", "此页面最后编辑")
# 页面里内嵌的 CSS / 脚注样式会被 <p> 的贪婪匹配卷进正文（实测：整段中文的
# cjk_ratio 掉到 0.03，因为后面接了一大段 .mw-parser-output 规则）。那不是
# 作者的字，混进语料就是在教校准器一个假的人类特征。
CSS_JUNK = re.compile(r"(mw-parser-output|wst-|@media|font-size:|margin:|padding:|display:)")


def html_to_blocks(html: str) -> list[str]:
    """Paragraph and poem-line blocks only — never the page furniture."""
    # 先整块切掉 <style> / <script>，再找段落：顺序反了就会被卷进去
    html = re.sub(r"<(style|script)\b[^>]*>[\s\S]*?</\1>", " ", html, flags=re.I)
    blocks: list[str] = []
    # <p> … </p>, <poem>…</poem>, and <div class="poem">…</div>
    pat = re.compile(
        r"<(p|poem|div\s+class=\"poem[^\"]*\")\b[^>]*>(.*?)</\1>", re.S | re.I)
    for m in pat.finditer(html):
        seg = m.group(2)
        seg = re.sub(r"<br\s*/?>", "\n", seg)
        seg = re.sub(r"<[^>]+>", "", seg)
        seg = htmllib.unescape(seg).replace("\u200b", "")
        for line in seg.split("\n"):
            line = line.strip()
            if not line or len(line) <= 1 or DROP_LINE.match(line):
                continue
            if CSS_JUNK.search(line):
                continue
            # 一行里 ASCII 符号占比过半 → 排版残留，不是正文
            ascii_junk = sum(1 for c in line if c in "{};:@()[]#.")
            if ascii_junk > max(3, len(line) * 0.15):
                continue
            blocks.append(line)
    return blocks


def chunk_blocks(blocks: list[str]) -> list[str]:
    """Pack blocks into ~TARGET_CHUNK char chunks, never crossing a block."""
    out, buf = [], []
    n = 0
    for b in blocks:
        if n + len(b) > MAX_CHUNK and n >= MIN_CHUNK:
            out.append("\n".join(buf))
            buf, n = [], 0
        buf.append(b)
        n += len(b)
        if n >= TARGET_CHUNK:
            out.append("\n".join(buf))
            buf, n = [], 0
    if buf and n >= MIN_CHUNK:
        out.append("\n".join(buf))
    return [c for c in out if len(c) >= MIN_CHUNK]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = {"note": "public-domain Chinese texts, fetched from zh.wikisource.org",
                "license": "author died > 50 years ago (PRC life+50)",
                "items": []}
    seen: set[str] = set()
    for register, items in SOURCES.items():
        d = HERE / register
        d.mkdir(parents=True, exist_ok=True)
        for title, sid in items:
            try:
                html = fetch_html(title, force=args.force)
            except Exception as e:                      # noqa: BLE001
                print("  !! %-28s %s" % (title, e))
                continue
            blocks = html_to_blocks(html)
            chunks = chunk_blocks(blocks)
            if not chunks:
                print("  -- %-28s (no usable text)" % title)
                continue
            kept = 0
            for i, c in enumerate(chunks):
                h = hashlib.sha1(re.sub(r"\s+", "", c).encode("utf-8")).hexdigest()[:12]
                if h in seen:          # 野艸 与单篇重复收录时只留一份
                    continue
                seen.add(h)
                fn = "%s_%02d.txt" % (sid, i)
                (d / fn).write_text(c + "\n", encoding="utf-8")
                manifest["items"].append({
                    "file": "%s/%s" % (register, fn), "register": register,
                    "label": "human", "source": "zh.wikisource.org",
                    "page": title, "chars": len(c),
                })
                kept += 1
            print("  ok %-28s %2d chunks  %5d chars" % (title, kept, sum(map(len, chunks))))

    (HERE / "_manifest_human.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n = len(manifest["items"])
    print("\n%d chunks -> %s" % (n, HERE))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
