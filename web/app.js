/* Local AI-writing signal detector v3.0 — frontend (no build step, no framework) */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { result: null, weights: null, calib: null, defaults: null };

  const EXAMPLES = {
    ai_en: "Moreover, this innovative solution seamlessly leverages cutting-edge technology to optimize workflows, enhance productivity, and unlock unprecedented value for organizations navigating today's rapidly evolving digital landscape. Furthermore, it is important to note that stakeholders should delve into the myriad of opportunities that this holistic approach can foster in the foreseeable future. Additionally, this comprehensive framework strategically empowers teams to unlock synergies across the entire digital ecosystem in a seamless and robust manner.",
    human_en: "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war.",
    human_zh: "昨晚下了点小雨，巷子里的石板路还是湿的。我骑车去买醋，老板娘还是那副慢悠悠的样子，一边找零钱一边抱怨说今年的青菜贵得没道理。我没接话，只是点了点头。回到家才发现，口袋里那张五块钱不知道什么时候蹭湿了，黏在一起，撕开的时候还带出一点纸毛。窗外有人在晾被单，铁架子偶尔咔一声。我把醋瓶搁灶台上，听见楼道里电视的声音模模糊糊传过来。",
    ai_zh: "综上所述，在高质量发展的背景下，我们要进一步赋能基层治理，推动数字化转型落地见效。值得注意的是，必须坚持系统观念，全面统筹各方资源，持续提升核心竞争力。首先，要完善顶层设计；其次，要加强协同联动；最后，要强化闭环管理，确保各项举措落地生根、开花结果。与此同时，我们要以人民为中心，不断满足人民群众对美好生活的向往，为经济社会发展注入新动能。",
    lit_zh: "张了张嘴，想了想。它还是空的，直直的。他在那儿站了一会儿，回去了。巫医看了他一眼，没说什么。鸡拿到手里，鸡不叫。那是名，是娘叫的。魂若来，铃会响。铃舌是空的，响的不是风。端木赐后来试着跟别人描述过，描述不出来。不是风吹的响法——风吹是乱响，这一声是正的。他抬起手，像课堂上要人安静。看得久了些。烟升上去，直直的。他说：没有线。有。你看不见。冰还在换，苍蝇还在。土是新翻的。人不在。屋门开着，里面空了。还是空。",
  };

  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  async function jget(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  }
  async function jpost(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let msg = `${r.status}`;
      try { msg = (await r.json()).detail || msg; } catch (e) { /* ignore */ }
      throw new Error(msg);
    }
    return r.json();
  }

  const fmt = (x, n = 3) => (typeof x === "number" ? x.toFixed(n) : "—");

  /* ---------------- theme ---------------- */
  function initTheme() {
    const saved = localStorage.getItem("lai_theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    $("btnTheme").onclick = () => {
      const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", cur);
      localStorage.setItem("lai_theme", cur);
    };
  }

  /* ---------------- input ---------------- */
  function initInput() {
    const ta = $("textInput");
    const upd = () => {
      const t = ta.value || "";
      const cjk = (t.match(/[\u4e00-\u9fff]/g) || []).length / Math.max(1, t.length);
      $("charCount").textContent = `${t.length} 字`;
      const need = cjk >= 0.45 ? 55 : 80;
      $("langTag").textContent = `CJK ${(cjk * 100).toFixed(0)}% · 建议下限 ${need}`;
      $("langTag").className = t.length < need ? "badge warn" : "badge ok";
    };
    ta.addEventListener("input", upd);
    upd();

    document.querySelectorAll(".chip[data-ex]").forEach((b) => {
      b.onclick = () => { ta.value = EXAMPLES[b.dataset.ex] || ""; upd(); };
    });

    const dz = $("dropzone");
    ["dragenter", "dragover"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", async (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) await importFile(f);
    });
    $("fileInput").addEventListener("change", async (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) await importFile(f);
      e.target.value = "";
    });
  }

  async function importFile(f) {
    $("status").textContent = `读取 ${f.name} …`;
    try {
      const buf = await f.arrayBuffer();
      let bin = "";
      const bytes = new Uint8Array(buf);
      for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
      const data = await jpost("/api/import", {
        filename: f.name,
        content_base64: btoa(bin),
      });
      $("textInput").value = data.text || "";
      $("textInput").dispatchEvent(new Event("input"));
      $("status").textContent = `${f.name} · ${data.chars} 字`;
    } catch (err) {
      $("status").textContent = `导入失败：${err.message}`;
    }
  }

  /* ---------------- analyze ---------------- */
  async function runAnalyze() {
    const text = $("textInput").value || "";
    if (!text.trim()) { $("status").textContent = "先粘贴或导入文本。"; return; }
    $("btnAnalyze").disabled = true;
    $("status").textContent = "分析中…";
    try {
      const payload = {
        text,
        backend: $("backend").value,
        ollama_host: $("ollamaHost").value,
        ollama_model: $("ollamaModel").value,
        use_onnx: $("useOnnx").checked,
        use_whitebox: $("useWhitebox").checked,
        highlight_mode: $("highlightMode").value,
        weights: state.weights || undefined,
      };
      const data = await jpost("/api/analyze", payload);
      state.result = data;
      render(data);
      $("status").textContent = `完成 · ${data.char_count} 字`;
      loadHistory();
    } catch (err) {
      $("status").textContent = `分析失败：${err.message}`;
    } finally {
      $("btnAnalyze").disabled = false;
    }
  }

  /* ---------------- render ---------------- */
  function render(d) {
    $("scoreNum").textContent = d.human_score;
    $("bandZh").textContent = d.human_band_zh;
    $("bandEn").textContent = d.human_band_en;
    $("aiP").textContent = fmt(d.ai_probability, 3);

    const conf = typeof d.confidence === "number" ? d.confidence : 0;
    $("confVal").textContent = conf.toFixed(2);
    $("confFill").style.width = `${Math.round(conf * 100)}%`;
    const notes = [];
    if (d.insufficient_evidence) notes.push("⚠️ 证据不足：文本太短或各特征结论分散，分数仅作参考。");
    if (d.too_short) notes.push(`文本短于建议下限 ${d.meta.min_chars} 字。`);
    const cm = d.meta.calibration || {};
    if (cm.provisional) notes.push(`校准样本不足（human=${cm.n_human} / ai=${cm.n_ai}），中文/该语言锚点为临时先验，置信度已下调。`);
    if (cm.temperature && Math.abs(cm.temperature - 1) > 0.02) notes.push(`温度缩放 T=${Number(cm.temperature).toFixed(2)}（T>1 冷却过自信）。`);
    if (!cm.calibrated) notes.push("没有 calibration.json，正在使用 v2.4 回退曲线。");
    if ((d.meta.voter_disagreement || 0) > 0.3) notes.push(`投票分歧 ${fmt(d.meta.voter_disagreement, 2)}（启发式 / ONNX / DNA 意见不一致）。`);
    $("confNote").textContent = notes.join(" ") || "各特征结论一致。";

    const flags = [];
    if (d.meta.mixed_writing_hint) flags.push('<span class="badge warn">疑似人机混写（段落间差异大）</span>');
    if (d.too_short) flags.push('<span class="badge warn">文本过短</span>');
    if (cm.provisional) flags.push('<span class="badge warn">校准 provisional</span>');
    const wb = d.whitebox || {};
    if (wb.used) flags.push(`<span class="badge ok">白盒信号：${esc(wb.model || "")} · mean logprob ${fmt(wb.mean_logprob, 2)}</span>`);
    else if (wb.error) flags.push(`<span class="badge warn">白盒未启用：${esc(wb.error)}</span>`);
    const ox = d.meta.onnx || {};
    flags.push(ox.used ? `<span class="badge ok">ONNX 已混入 (${fmt(ox.blend_applied, 2)})</span>`
                       : `<span class="badge soft">ONNX 未混入：${esc(ox.status || "n/a")}</span>`);
    $("flags").innerHTML = flags.join(" ");

    renderHighlight(d);
    renderFeatures(d);
    renderSentences(d);
    renderParagraphs(d);
    renderAnnotations(d);
    $("jsonOut").textContent = JSON.stringify(d, null, 2);
    $("policyTag").textContent = `policy: ${d.highlight_policy}`;
  }

  function renderHighlight(d) {
    const box = $("highlight");
    if (!d.text) { box.textContent = "（无文本）"; return; }
    const sents = (d.sentences || []).slice().sort((a, b) => a.char_start - b.char_start);
    let out = "";
    let cursor = 0;
    sents.forEach((s, i) => {
      if (s.char_start > cursor) out += esc(d.text.slice(cursor, s.char_start));
      const cls = s.heat === "red" ? "s-red" : (s.heat === "yellow" ? "s-yellow" : "");
      const title = `AI≈${fmt(s.ai_probability, 2)} · 本文分位 ${fmt(s.percentile, 2)}`;
      const inner = esc(s.text);
      out += cls
        ? `<span class="${cls}" data-i="${i}" title="${esc(title)}">${inner}</span>`
        : `<span data-i="${i}" title="${esc(title)}">${inner}</span>`;
      cursor = s.char_start + s.length;
    });
    if (cursor < d.text.length) out += esc(d.text.slice(cursor));
    box.innerHTML = out;

    box.querySelectorAll("span[data-i]").forEach((el) => {
      el.onclick = () => {
        const s = sents[Number(el.dataset.i)];
        const tip = $("sentTip");
        const feats = (s.top_features || []).slice(0, 4)
          .map((f) => `${esc(f.name_zh)} <code>${esc(f.name)}</code> z=${fmt(f.z, 2)}`)
          .join("，") || "无显著特征";
        tip.innerHTML = `<b>AI≈${fmt(s.ai_probability, 2)}</b> · 分位 ${fmt(s.percentile, 2)} · ${esc(s.heat)}<br>${feats}`;
        tip.classList.remove("hidden");
      };
    });
  }


  function renderAnnotations(d) {
    const items = (d.annotations || []).filter((a) => a.action === "flag" || a.action === "review");
    const all = d.annotations || [];
    if (!all.length) {
      $("tab-annotations").innerHTML = '<p class="muted">无标注。特征未超过触发门槛，或句子未进入 flag/review 带。</p>';
      return;
    }
    const badge = (act) => act === "flag"
      ? '<span class="badge warn">flag</span>'
      : (act === "review" ? '<span class="badge soft">review</span>' : '<span class="badge">ignore</span>');
    const rows = items.map((a) => {
      const kind = a.kind === "feature" ? "特征" : (a.kind === "sentence" ? "句子" : "段落");
      const prev = a.preview ? `<div class="muted">${esc(a.preview)}</div>` : "";
      return `<tr>
        <td>${badge(a.action)} <span class="muted">${esc(kind)}</span></td>
        <td><b>${esc(a.name_zh || a.key)}</b><br><code>${esc(a.key)}</code>${prev}</td>
        <td class="num">${fmt(a.p_ai, 3)}</td>
        <td class="num">${fmt(a.confidence, 2)}</td>
      </tr>`;
    }).join("");
    const nFlag = all.filter((a) => a.action === "flag").length;
    const nReview = all.filter((a) => a.action === "review").length;
    $("tab-annotations").innerHTML = `
      <p class="muted">Jev 式并行标注：每个 lit/zh/marketing 槽与句段是独立问题（AI概率 + 置信度 + 行动带）。
      仅列出 <b>flag / review</b>（共 ${items.length}；全量 flag=${nFlag} review=${nReview}）。
      阈值示例：flag 若 p≥0.65 且 conf≥0.5；review 若 p≥0.4。</p>
      <table><thead><tr><th>行动</th><th>标注</th><th class="num">AI概率</th><th class="num">置信度</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4" class="muted">没有 flag/review 项</td></tr>'}</tbody></table>`;
  }

  function renderFeatures(d) {
    const rows = (d.features || []).map((f) => {
      const w = Math.min(1, Math.abs(f.effective_weight || 0) * 8);
      const neg = (f.gain || 0) < 0;
      return `<tr>
        <td><b>${esc(f.name_zh)}</b><br><code>${esc(f.name)}</code>${f.flipped ? ' <span class="flip">FLIP</span>' : ""}</td>
        <td class="num">${fmt(f.value, 3)}</td>
        <td class="num">${fmt(f.z, 2)}</td>
        <td class="num">${fmt(f.anchor_lo, 2)} → ${fmt(f.anchor_hi, 2)}</td>
        <td class="num">${fmt(f.disc, 2)}</td>
        <td class="num">${fmt(f.gain, 2)}</td>
        <td><div class="bar"><i class="${neg ? "neg" : ""}" style="width:${(w * 100).toFixed(0)}%"></i></div></td>
        <td class="num">${fmt(f.contribution, 4)}</td>
        <td>${esc(f.explanation_zh)}<br><span class="muted">${esc(f.anchor_source)}</span></td>
      </tr>`;
    }).join("");
    $("tab-features").innerHTML = `<table>
      <thead><tr>
        <th>特征</th><th class="num">raw</th><th class="num">z</th><th class="num">锚点 H→A</th>
        <th class="num">判别力</th><th class="num">增益</th><th>有效权重</th>
        <th class="num">贡献</th><th>说明</th>
      </tr></thead><tbody>${rows || '<tr><td colspan="9" class="muted">无</td></tr>'}</tbody></table>
      <p class="muted" style="margin-top:8px">z：0 = 校准集人类中位数，1 = AI 侧偏高段。判别力/增益由 calib/ 实测得出；
      <span class="flip">FLIP</span> 表示数据里这个特征反而偏向人类（小样本下可能是语料现象，不是规律）。</p>`;
  }

  function renderSentences(d) {
    const rows = (d.sentences || []).map((s, i) => {
      const cls = s.heat === "red" ? "s-red" : (s.heat === "yellow" ? "s-yellow" : "");
      return `<tr>
        <td class="num">${i + 1}</td>
        <td><span class="${cls}">&nbsp;${esc(s.heat)}&nbsp;</span></td>
        <td class="num">${fmt(s.ai_probability, 3)}</td>
        <td class="num">${fmt(s.percentile, 2)}</td>
        <td>${esc(s.text.length > 90 ? s.text.slice(0, 88) + "…" : s.text)}</td>
        <td>${(s.top_features || []).slice(0, 3).map((f) => `${esc(f.name_zh)}(${fmt(f.value, 2)})`).join("；") || "—"}</td>
      </tr>`;
    }).join("");
    $("tab-sentences").innerHTML = `<table>
      <thead><tr><th class="num">#</th><th>档</th><th class="num">P(AI)</th><th class="num">分位</th><th>句子</th><th>顶层特征</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="6" class="muted">无</td></tr>'}</tbody></table>`;
  }

  function renderParagraphs(d) {
    const ps = d.paragraphs || [];
    if (!ps.length) {
      $("tab-paragraphs").innerHTML = '<p class="muted">文本太短，未做段落窗口扫描（需要 ≥240 字）。</p>';
      return;
    }
    const max = Math.max(...ps.map((p) => p.ai_probability));
    const rows = ps.map((p) => `
      <div class="para-row">
        <span class="num" style="min-width:52px">${fmt(p.ai_probability, 2)}</span>
        <div class="para-bar"><i style="width:${(p.ai_probability * 100).toFixed(0)}%"></i></div>
        <span class="muted" style="min-width:190px">${esc(p.preview)}</span>
        <span class="muted">${p.chars}字</span>
      </div>`).join("");
    $("tab-paragraphs").innerHTML =
      `<p class="muted">段落窗口 ≈320 字；最高 ${fmt(max, 2)}，最低 ${fmt(Math.min(...ps.map((p) => p.ai_probability)), 2)}。
       ${d.meta.mixed_writing_hint ? '<b class="flip">段落间差异 ≥0.40 → 可能是人机混写。</b>' : "段落间差异不大。"}</p>${rows}`;
  }

  function renderWeights() {
    const w = state.weights || state.defaults || {};
    const g = {};
    if (state.calib && state.calib.present) {
      const lang = (state.result && state.result.meta.calibration.lang) === "zh" ? "zh" : "en";
      const lc = state.calib.langs[lang] || state.calib.langs.en || {};
      (lc.anchors || []).forEach((a) => { g[a.name] = a.g; });
    }
    const keys = Object.keys(w).sort((a, b) => Math.abs(g[b] || 0) - Math.abs(g[a] || 0));
    const rows = keys.map((k) => `
      <div class="weight-row">
        <span title="${esc(k)}">${esc(k)} <span class="muted">g=${fmt(g[k], 2)}</span></span>
        <input type="range" min="0" max="0.4" step="0.005" value="${w[k]}" data-k="${esc(k)}">
        <span class="num">${fmt(w[k], 3)}</span>
      </div>`).join("");
    $("tab-weights").innerHTML =
      `<p class="muted">拖动后点「分析」生效；g 是该特征在校准集上的实测增益（0 = 无判别力，调它也无效）。
       <button id="btnSaveW" class="ghost small">保存为默认权重</button>
       <button id="btnResetW" class="ghost small">恢复默认</button></p>${rows}`;
    $("tab-weights").querySelectorAll("input[type=range]").forEach((el) => {
      el.oninput = () => {
        state.weights = state.weights || { ...(state.defaults || {}) };
        state.weights[el.dataset.k] = parseFloat(el.value);
        el.nextElementSibling.textContent = parseFloat(el.value).toFixed(3);
      };
    });
    $("btnSaveW").onclick = async () => {
      try {
        await jpost("/api/weights", { weights: state.weights || state.defaults });
        $("status").textContent = "权重已保存到 weights.json";
      } catch (e) { $("status").textContent = `保存失败：${e.message}`; }
    };
    $("btnResetW").onclick = () => { state.weights = { ...(state.defaults || {}) }; renderWeights(); };
  }

  async function renderCalib() {
    const c = state.calib;
    if (!c || !c.present) {
      $("tab-calib").innerHTML = '<p class="muted">没有 calibration.json。运行 <code>python calibrate.py</code>，或在下面上传样本后点「重算校准」。</p>';
      return;
    }
    const blocks = Object.keys(c.langs || {}).map((lang) => {
      const lc = c.langs[lang];
      const rows = (lc.anchors || []).slice(0, 16).map((a) => `
        <tr><td><code>${esc(a.name)}</code></td><td class="num">${fmt(a.lo, 3)}</td>
        <td class="num">${fmt(a.hi, 3)}</td><td class="num">${fmt(a.disc, 2)}</td>
        <td class="num">${fmt(a.g, 2)}</td><td>${esc(a.source)}${a.flipped ? ' <span class="flip">FLIP</span>' : ""}</td></tr>`).join("");
      const tLine = (typeof lc.temperature === "number")
        ? `<p class="muted">温度 T=${fmt(lc.temperature, 3)}`
          + (lc.ece_before != null ? ` · ECE ${fmt(lc.ece_before, 3)}→${fmt(lc.ece_after, 3)}` : "")
          + (lc.brier_before != null ? ` · Brier ${fmt(lc.brier_before, 3)}→${fmt(lc.brier_after, 3)}` : "")
          + ` <span title="T>1 冷却过自信">（T>1 cools）</span></p>`
        : "";
      return `<h3 style="margin:10px 0 4px">${esc(lang)} · human=${lc.n_human} ai=${lc.n_ai}
        ${lc.provisional ? '<span class="badge warn">provisional</span>' : '<span class="badge ok">ok</span>'}</h3>
        ${tLine}
        ${(lc.notes || []).map((n) => `<p class="muted">${esc(n)}</p>`).join("")}
        <table><thead><tr><th>特征</th><th class="num">H 中位</th><th class="num">AI 上四分位</th>
        <th class="num">判别力</th><th class="num">增益</th><th>来源</th></tr></thead><tbody>${rows}</tbody></table>`;
    }).join("");
    $("tab-calib").innerHTML = `
      <p class="muted">built ${esc(c.built_at)} ·
        <button id="btnRecal" class="ghost small">重算校准</button>
        <button id="btnValidate" class="ghost small">留出法验证</button>
        <span id="calibMsg"></span></p>
      ${blocks}
      <h3 style="margin:14px 0 4px">加入你自己的样本</h3>
      <p class="muted">把你自己的手稿放进人类侧（中文尤其缺），AI 文本放进 AI 侧，然后「重算校准」。
      每侧 ≥5 条后该语言的锚点才不再是 provisional。</p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <select id="calibGroup">
          <option value="human_zh">human_zh（你的人类中文稿）</option>
          <option value="ai_zh">ai_zh</option>
          <option value="human_en">human_en</option>
          <option value="ai_en">ai_en</option>
        </select>
        <input id="calibName" placeholder="my_draft.txt" style="width:160px">
        <button id="btnUploadCalib" class="ghost small">上传当前输入框文本</button>
      </div>
      <div id="validateOut" style="margin-top:10px"></div>`;
    $("btnRecal").onclick = async () => {
      $("calibMsg").textContent = "重算中…";
      try {
        const r = await jpost("/api/calibrate", {});
        $("calibMsg").textContent = `已重算 ${r.built_at}`;
        await loadCalib();
      } catch (e) { $("calibMsg").textContent = `失败：${e.message}`; }
    };
    $("btnValidate").onclick = async () => {
      $("validateOut").textContent = "留出法验证中…";
      try {
        const r = await jget("/api/calibration/validate");
        const out = Object.keys(r.loo || {}).map((k) => {
          const o = r.loo[k];
          const bins = (o.reliability_bins || []).slice(0, 5).map((b) =>
            `n=${b.n} conf=${fmt(b.conf_mean, 2)} acc=${fmt(b.acc_mean, 2)}`).join(" · ");
          return `<p><b>${esc(k)}</b> LOO AUC=${fmt(o.auc, 3)} · T=${fmt(o.temperature, 3)}<br>
            ECE ${fmt(o.ece_before, 3)}→${fmt(o.ece_after, 3)} ·
            Brier ${fmt(o.brier_before, 3)}→${fmt(o.brier_after, 3)} ·
            LOO ECE=${fmt(o.ece_loo, 3)} Brier=${fmt(o.brier_loo, 3)}<br>
            人类 P(AI)∈[${fmt(o.human_min_p, 2)}, ${fmt(o.human_max_p, 2)}] ·
            AI P(AI)∈[${fmt(o.ai_min_p, 2)}, ${fmt(o.ai_max_p, 2)}]
            ${bins ? `<br><span class="muted">reliability(eq-mass): ${esc(bins)}</span>` : ""}</p>`;
        }).join("");
        $("validateOut").innerHTML = out + '<p class="muted">AUC 越接近 1 越好；ECE/Brier 越低越好。AI概率≠置信度。</p>';
      } catch (e) { $("validateOut").textContent = `失败：${e.message}`; }
    };
    $("btnUploadCalib").onclick = async () => {
      const text = $("textInput").value || "";
      if (text.trim().length < 40) { $("calibMsg").textContent = "输入框文本太短（≥40 字）"; return; }
      try {
        const r = await jpost("/api/calib/upload", {
          group: $("calibGroup").value,
          filename: $("calibName").value || "my_sample.txt",
          text,
        });
        $("calibMsg").textContent = `已写入 ${r.path}`;
      } catch (e) { $("calibMsg").textContent = `失败：${e.message}`; }
    };
  }

  /* ---------------- misc ---------------- */
  async function loadCalib() {
    try {
      state.calib = await jget("/api/calibration");
      const en = state.calib.langs && state.calib.langs.en;
      $("calibTag").textContent = state.calib.present
        ? `校准 ${state.calib.built_at} · en ${en ? en.n_human : "?"}H/${en ? en.n_ai : "?"}A`
        : "校准：缺失";
      $("calibTag").className = state.calib.present ? "badge ok" : "badge warn";
    } catch (e) {
      $("calibTag").textContent = "校准：读取失败";
    }
    renderCalib();
    renderWeights();
  }

  async function loadHistory() {
    try {
      const r = await jget("/api/history");
      const items = (r.items || []).slice().reverse();
      $("historyList").innerHTML = items.length
        ? items.map((h) => `<li>${esc(h.at)} · <b>${h.human_score}</b> · conf ${fmt(h.confidence, 2)} · ${esc(h.preview)}…</li>`).join("")
        : '<li class="muted">暂无</li>';
    } catch (e) { /* ignore */ }
  }

  async function probe() {
    $("probeBadge").textContent = "探测中…";
    try {
      const r = await jget("/api/probe");
      const o = r.ollama || {};
      const ox = r.onnx || {};
      const txt = `Ollama ${o.available ? "在线" : "离线"} · ONNX EN ${ox.en && ox.en.available ? "有" : "无"} / ZH ${ox.zh && ox.zh.available ? "有" : "无"}`;
      $("probeBadge").textContent = txt;
      $("probeBadge").className = o.available ? "badge ok" : "badge soft";
    } catch (e) {
      $("probeBadge").textContent = "探测失败";
    }
  }

  function initTabs() {
    document.querySelectorAll(".tab").forEach((t) => {
      t.onclick = () => {
        document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
        document.querySelectorAll(".tab-page").forEach((x) => x.classList.remove("active"));
        t.classList.add("active");
        $(`tab-${t.dataset.tab}`).classList.add("active");
      };
    });
  }

  function initButtons() {
    $("btnAnalyze").onclick = runAnalyze;
    $("btnProbe").onclick = probe;
    $("btnExport").onclick = async () => {
      if (!state.result) { $("status").textContent = "先分析一次。"; return; }
      const r = await jpost("/api/export", state.result);
      $("status").textContent = `已导出：${r.path}`;
    };
    $("btnCompare").onclick = () => $("compareBox").classList.toggle("hidden");
    $("btnRunCompare").onclick = async () => {
      $("compareOut").textContent = "对比中…";
      try {
        const r = await jpost("/api/compare", {
          before: $("beforeText").value || "",
          after: $("afterText").value || "",
          use_onnx: $("useOnnx").checked,
          weights: state.weights || undefined,
        });
        const d = r.delta_human_score;
        $("compareOut").innerHTML =
          `<b>${r.before.human_score} → ${r.after.human_score}</b>（Δ ${d > 0 ? "+" : ""}${d}）<br>` +
          (r.feature_deltas || []).slice(0, 6).map(
            (f) => `${esc(f.name)}: ${fmt(f.before, 3)} → ${fmt(f.after, 3)}`).join("<br>");
      } catch (e) { $("compareOut").textContent = `失败：${e.message}`; }
    };
    $("btnClearHistory").onclick = async () => { await fetch("/api/history", { method: "DELETE" }); loadHistory(); };
  }

  /* 版本号只有一个真相来源：/api/meta 的 engine（它又来自 detector._ENGINE）。
     HTML 里那个 "v3.1" 是 v3.1 时代的字面量——发 v3.2 时没人记得改它，
     于是界面写着 v3.1、Cowrite 那边写着 v3.2，报 bug 的人第一句就报错版本。
     取不到就保留 HTML 里的值，不要让一个装饰性数字挡住启动。 */
  async function syncVersion() {
    try {
      const m = await jget("/api/meta");
      const v = String(m.engine || "").split("_v").pop();
      if (!v) return;
      const el = document.querySelector(".brand .ver");
      if (el) el.textContent = "v" + v;
      document.title = `本地 AI 写作信号检测器 v${v}`;
    } catch (e) { /* 装饰性，失败不影响任何功能 */ }
  }

  async function boot() {
    initTheme();
    initInput();
    initTabs();
    initButtons();
    syncVersion();
    try {
      const w = await jget("/api/weights");
      state.defaults = w.defaults;
      state.weights = { ...w.weights };
    } catch (e) { /* ignore */ }
    await loadCalib();
    loadHistory();
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
