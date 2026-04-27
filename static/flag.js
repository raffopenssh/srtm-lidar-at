/* srtm-lidar-at flag widget
 *
 * Embeds anywhere. When the user selects text on the page, a discreet
 * floating chip appears above the selection. Clicking it calls
 * /api/v1/flags/match with the selected text (and optional context lon/lat
 * + kg). If the match is plausible, a small popover opens with details and
 * a one-click "Report" form that POSTs /api/v1/feedback.
 *
 * Usage (minimal):
 *     <script src="/flag.js" defer></script>
 *     <script> SrtmFlag.install(); </script>
 *
 * Optional contextual hints — sites can scope matching by setting
 * data-srtm-kg, data-srtm-lon, data-srtm-lat on any ancestor element. The
 * widget walks up the DOM from the selection anchor to find them.
 *
 * Public API:
 *     SrtmFlag.install(opts?)            — enable selection watcher
 *     SrtmFlag.uninstall()
 *     SrtmFlag.openFor({obj_ref, kg_code, point, hint})  — open programmatically
 *     SrtmFlag.matchText(text, ctx?)      — pure API call, returns Promise
 *     SrtmFlag.attachClickHandler(el)     — explicit click target (icon)
 *     SrtmFlag.on('reported', cb)        — fires on successful POST
 *
 * Designed to be subtle: never hijacks normal selection. Chip auto-hides
 * after 4s of no interaction.
 */
(function () {
  'use strict';
  if (window.SrtmFlag) return;

  const API_BASE = (typeof SRTM_API_BASE !== 'undefined' && SRTM_API_BASE) || '';
  const TYPES = ['tree','shrub','hedge','grass','crop','road','path','parking',
    'roof','greenhouse','solar_panel','wall','fence','mast','wind_turbine',
    'water','orchard','vineyard','garden','bare_soil','rock','excavation',
    'fill','tree_loss','construction','bridge','substation',
    'forest','woodland','hedgerow','waterbody','building','not_a_feature'];

  const listeners = {};
  function emit(name, data){ (listeners[name]||[]).forEach(f => { try { f(data); } catch(e){} }); }
  function on(name, cb){ (listeners[name]=listeners[name]||[]).push(cb); }

  // ----- DOM helpers ---------------------------------------------------
  function el(tag, attrs, ...children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'style') Object.assign(n.style, attrs[k]);
      else if (k === 'on') for (const ev in attrs.on) n.addEventListener(ev, attrs.on[ev]);
      else if (k === 'class') n.className = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    }
    for (const c of children) if (c != null) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    return n;
  }

  function ensureStyles() {
    if (document.getElementById('srtm-flag-css')) return;
    const css = `
      .srtm-chip { position: fixed; z-index: 2147483646; background: #1f6feb;
        color: #fff; font: 11px 'SF Mono', monospace; padding: 6px 10px;
        border-radius: 14px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,.4);
        transition: opacity .15s; user-select: none;
        -webkit-user-select: none; -webkit-touch-callout: none;
        -webkit-tap-highlight-color: rgba(0,0,0,0); touch-action: manipulation; }
      /* Mobile chip: dock to bottom-center of viewport so it never fights
         the OS text-selection toolbar or covers the selected content. */
      .srtm-chip.srtm-chip-touch { padding: 12px 22px; font-size: 14px;
        border-radius: 26px; min-height: 44px; box-shadow: 0 6px 20px rgba(0,0,0,.55);
        bottom: calc(env(safe-area-inset-bottom, 0px) + 18px);
        top: auto !important; left: 50% !important;
        transform: translateX(-50%); }
      .srtm-chip:hover { background: #388bfd; }
      .srtm-chip:before { content: '🚩'; margin-right: 4px; font-size: 12px; }
      .srtm-pop { position: fixed; z-index: 2147483647; background: #161b22; color: #c9d1d9;
        border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px;
        font: 12px 'SF Mono', monospace; min-width: 320px; max-width: 460px;
        box-shadow: 0 6px 24px rgba(0,0,0,.5); max-height: 80vh; overflow-y: auto;
        overscroll-behavior: contain; -webkit-overflow-scrolling: touch; }
      @media (max-width: 600px) {
        /* Sheet-style popover: full-width, anchored to bottom of viewport,
           independent scroll, never affected by document scroll position. */
        .srtm-pop { left: 8px !important; right: 8px !important;
          width: auto !important; max-width: none !important;
          min-width: 0 !important;
          top: auto !important;
          bottom: calc(env(safe-area-inset-bottom, 0px) + 8px) !important;
          max-height: 75vh; }
        .srtm-pop select, .srtm-pop input, .srtm-pop textarea,
        .srtm-pop button:not(.srtm-pop-close) { font-size: 14px !important;
          padding: 8px 10px !important; min-height: 38px; }
        .srtm-pop .cands { max-height: 28vh; }
        .srtm-pop button.srtm-pop-close { display: inline-flex !important; }
      }
      .srtm-pop button.srtm-pop-close { display: none; position: absolute;
        top: 4px; right: 4px; background: transparent; border: 0;
        color: #8b949e; font-size: 24px; line-height: 1; padding: 6px 10px;
        cursor: pointer; box-shadow: none; min-height: 0; font-family: inherit; }
      .srtm-pop button.srtm-pop-close:hover { background: transparent;
        color: #c9d1d9; filter: none; }
      .srtm-pop h4 { font-size: 12px; color: #58a6ff; margin: 0 0 8px; }
      .srtm-pop .row { display: flex; gap: 6px; margin: 4px 0; align-items: baseline; flex-wrap: wrap; }
      .srtm-pop .lb { color: #8b949e; min-width: 64px; font-size: 10px; text-transform: uppercase; }
      .srtm-pop .vl { color: #c9d1d9; }
      .srtm-pop .cands { max-height: 130px; overflow-y: auto; margin: 6px 0;
        border: 1px solid #30363d; border-radius: 4px; }
      .srtm-pop .cand { padding: 4px 8px; cursor: pointer; border-bottom: 1px solid #21262d; }
      .srtm-pop .cand:hover, .srtm-pop .cand.sel { background: #1f6feb33; }
      .srtm-pop .cand .meta { color: #8b949e; font-size: 10px; }
      .srtm-pop select, .srtm-pop input, .srtm-pop textarea {
        background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
        border-radius: 4px; padding: 4px 6px; font: 11px 'SF Mono', monospace; width: 100%; }
      .srtm-pop textarea { min-height: 36px; resize: vertical; }
      .srtm-pop .btns { display: flex; gap: 6px; justify-content: flex-end; margin-top: 8px; }
      .srtm-pop button { background: #238636; color: #fff; border: 0; border-radius: 4px;
        padding: 5px 12px; font: 11px 'SF Mono', monospace; cursor: pointer; }
      .srtm-pop button.alt { background: #30363d; }
      .srtm-pop button.danger { background: #da3633; }
      .srtm-pop button:hover { filter: brightness(1.15); }
      .srtm-pop .flag-badge { display: inline-block; padding: 1px 5px; border-radius: 3px;
        font-size: 10px; margin-right: 4px; }
      .srtm-pop .flag-critical { background: #da3633; color: #fff; }
      .srtm-pop .flag-high { background: #b62324; color: #fff; }
      .srtm-pop .flag-medium { background: #d29922; color: #000; }
      .srtm-pop .flag-low { background: #586069; color: #fff; }
      .srtm-pop .pred { background: #0d1117; border: 1px solid #30363d;
        border-radius: 4px; padding: 6px 8px; margin: 4px 0;
        font-size: 10.5px; color: #8b949e; line-height: 1.4; }
      .srtm-pop .pred b { color: #c9d1d9; font-weight: 600; }
      .srtm-pop .pred .flip { color: #d29922; }
      .srtm-pop .pred .verify { color: #3fb950; }
      .srtm-pop .agg { color: #d29922; font-size: 10px; margin-left: 4px; }
      .srtm-pop .err { color: #f85149; font-size: 11px; }
      .srtm-pop .ok { color: #3fb950; font-size: 11px; }
      .srtm-hl { background: rgba(212, 153, 34, .35); outline: 1px solid #d29922;
                 transition: background .3s, outline-color .3s; }`;
    const s = el('style', { id: 'srtm-flag-css' }); s.textContent = css;
    document.head.appendChild(s);
  }

  // ----- Context discovery from selection ------------------------------
  function findContext(node) {
    const ctx = { kg_code: null, lon: null, lat: null };
    let n = node;
    while (n && n.nodeType === 3) n = n.parentNode;
    while (n && n.getAttribute) {
      if (!ctx.kg_code) {
        const k = n.getAttribute('data-srtm-kg');
        if (k) ctx.kg_code = k;
      }
      if (!ctx.lon) {
        const lo = n.getAttribute('data-srtm-lon');
        if (lo) ctx.lon = parseFloat(lo);
      }
      if (!ctx.lat) {
        const la = n.getAttribute('data-srtm-lat');
        if (la) ctx.lat = parseFloat(la);
      }
      n = n.parentNode;
    }
    return ctx;
  }

  // ----- Selection tracking --------------------------------------------
  let chipEl = null, chipTimer = null, lastSel = null;
  // Touch devices (Android/iOS): the OS text-selection toolbar overlaps
  // any small in-page chip. We avoid that by (a) only showing the chip
  // *after* the selection has settled (i.e. the user has finished
  // dragging the handles), (b) anchoring it well below the selection so
  // it doesn't fight the native menu, and (c) making it large enough to
  // tap. We also key off touchend / pointerup rather than mousedown.
  const IS_TOUCH = (typeof window !== 'undefined') && (
    (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) ||
    ('ontouchstart' in window));
  let _touchSettleTimer = null;

  function placeChip(rect, sel) {
    if (!chipEl) {
      chipEl = el('div', {
        class: 'srtm-chip' + (IS_TOUCH ? ' srtm-chip-touch' : ''),
        title: 'Flag this object',
      }, IS_TOUCH ? 'Flag selection' : 'Flag');
      // Use click (and pointerup as fallback) instead of mousedown so we
      // don't preventDefault'ing the native selection menu. On touch the
      // first tap dismisses native menu; the second tap on our chip wins.
      const fire = (e) => { e.preventDefault(); e.stopPropagation();
                            openPopForSelection(lastSel || sel); };
      chipEl.addEventListener('click', fire);
      chipEl.addEventListener('touchend', fire, { passive: false });
      document.body.appendChild(chipEl);
    } else {
      // refresh handler reference (lastSel may have been updated)
    }
    // Touch: chip is bottom-docked via CSS (safe-area aware). Don't write
    // top/left here — leave them blank so CSS rules win. Desktop: place
    // chip just above the selection.
    if (IS_TOUCH) {
      chipEl.style.top = ''; chipEl.style.left = '';
    } else {
      const chipH = 28, chipW = 70;
      const top = Math.max(8, rect.top - chipH - 4);
      const left = Math.min(window.innerWidth - chipW - 8,
                            Math.max(8, rect.left + rect.width/2 - chipW/2));
      chipEl.style.top = top + 'px';
      chipEl.style.left = left + 'px';
    }
    chipEl.style.display = 'block';
    chipEl.style.opacity = '1';
    if (chipTimer) clearTimeout(chipTimer);
    chipTimer = setTimeout(hideChip, IS_TOUCH ? 12000 : 4500);
  }
  function hideChip() { if (chipEl) chipEl.style.display = 'none'; }

  function _evaluateSelection() {
    const sel = document.getSelection();
    if (!sel || sel.isCollapsed || !sel.toString().trim()) { hideChip(); return; }
    const text = sel.toString().trim();
    if (text.length < 2 || text.length > 200) { hideChip(); return; }
    const hasNum = /\d/.test(text);
    const hasType = TYPES.some(t => new RegExp('\\b' + t + '\\b','i').test(text));
    const ctx = findContext(sel.anchorNode);
    const hasCtx = !!(ctx.kg_code || ctx.lon);
    if (!hasNum && !hasType && !hasCtx) { hideChip(); return; }
    let rect; try { rect = sel.getRangeAt(0).getBoundingClientRect(); } catch (e) { return; }
    if (!rect || (rect.width === 0 && rect.height === 0)) return;
    lastSel = { text, ctx, rect };
    placeChip(rect, lastSel);
  }

  function onSelectionChange() {
    if (!IS_TOUCH) { _evaluateSelection(); return; }
    // On touch: hide while the user is actively dragging selection
    // handles. We re-show only after a short settle period to avoid
    // flashing under the native menu.
    hideChip();
    if (_touchSettleTimer) clearTimeout(_touchSettleTimer);
    _touchSettleTimer = setTimeout(_evaluateSelection, 700);
  }
  function onSelectionSettle() {
    // Called on touchend / pointerup: re-evaluate after a small delay
    // so the OS has finalised the selection rectangle.
    if (!IS_TOUCH) return;
    if (_touchSettleTimer) clearTimeout(_touchSettleTimer);
    _touchSettleTimer = setTimeout(_evaluateSelection, 350);
  }

  // ----- Popover -------------------------------------------------------
  let popEl = null;

  function closePop() {
    if (popEl) { popEl.remove(); popEl = null; }
    document.removeEventListener('mousedown', popOutsideHandler, true);
    document.removeEventListener('touchstart', popOutsideHandler, true);
    document.removeEventListener('keydown', popKeyHandler, true);
  }
  function popOutsideHandler(e) {
    if (popEl && !popEl.contains(e.target)) closePop();
  }
  function popKeyHandler(e) { if (e.key === 'Escape') closePop(); }

  function showPop(anchorRect, content) {
    closePop();
    popEl = el('div', { class: 'srtm-pop' });
    // Mobile/touch users get a visible close button in addition to
    // tap-outside / Escape. Desktop hides it via CSS.
    const closeBtn = el('button', {
      class: 'srtm-pop-close', type: 'button',
      title: 'Close', 'aria-label': 'Close',
    }, '×');
    const fireClose = (e) => { e.preventDefault(); e.stopPropagation(); closePop(); };
    closeBtn.addEventListener('click', fireClose);
    closeBtn.addEventListener('touchend', fireClose, { passive: false });
    popEl.appendChild(closeBtn);
    const body = el('div', { class: 'srtm-pop-body' });
    body.appendChild(content);
    popEl.appendChild(body);
    document.body.appendChild(popEl);
    // On mobile (≤600px) the CSS pins the popover to the viewport bottom
    // as a sheet — don't override top/left.
    const isMobile = window.matchMedia('(max-width: 600px)').matches;
    if (!isMobile) {
      const r = popEl.getBoundingClientRect();
      const top = Math.min(window.innerHeight - r.height - 8,
                  Math.max(8, anchorRect.bottom + 6));
      const left = Math.min(window.innerWidth - r.width - 8,
                   Math.max(8, anchorRect.left));
      popEl.style.top = top + 'px';
      popEl.style.left = left + 'px';
    }
    setTimeout(() => {
      document.addEventListener('mousedown', popOutsideHandler, true);
      document.addEventListener('touchstart', popOutsideHandler, true);
      document.addEventListener('keydown', popKeyHandler, true);
    }, 0);
  }

  function _popBody() {
    return popEl && popEl.querySelector('.srtm-pop-body');
  }

  function openPopForSelection(sel) {
    hideChip();
    const { text, ctx, rect } = sel;
    const loading = el('div', null, 'Matching “' + text + '”…');
    showPop(rect, loading);
    matchText(text, ctx).then(res => {
      const body = _popBody(); if (!body) return;
      body.replaceChildren(renderMatch(res, { selectedText: text, ctx }));
    }).catch(e => {
      const body = _popBody(); if (!body) return;
      body.replaceChildren(el('div', { class: 'err' }, 'Match failed: ' + e));
    });
  }

  // ----- Match / render ------------------------------------------------
  function matchText(text, ctx = {}) {
    const params = new URLSearchParams();
    params.set('text', text);
    if (ctx.kg_code) params.set('kg', ctx.kg_code);
    if (ctx.lon != null) params.set('lon', ctx.lon);
    if (ctx.lat != null) params.set('lat', ctx.lat);
    return fetch(API_BASE + '/api/v1/flags/match?' + params).then(r => r.json());
  }

  function fmtCoord(v) { return v == null ? '—' : (Math.round(v*1e5)/1e5).toFixed(5); }

  function renderCandidate(c) {
    const meta = [];
    if (c.height_max_m != null) meta.push(c.height_max_m.toFixed(1) + 'm');
    if (c.area_sqm != null) meta.push(Math.round(c.area_sqm) + 'm²');
    if (c.distance_m != null) meta.push(c.distance_m.toFixed(0) + 'm away');
    if (c.rf_confidence != null) meta.push('rf=' + c.rf_confidence.toFixed(2));
    const aliasN = (c.aliases || []).filter(Boolean).length;
    const aliasNote = aliasN ? ' · +' + aliasN + ' alias' + (aliasN===1?'':'es') : '';
    let agg = null;
    if (c.agg && (c.agg.total_weight || c.agg.codes && c.agg.codes.length)) {
      agg = el('span', { class: 'agg' },
        '⚠ weight=' + (c.agg.total_weight||0).toFixed(1)
        + ' · ' + (c.agg.codes||[]).join(', '));
    }
    return el('div', { class: 'cand', 'data-ref': c.obj_ref },
      el('div', null, c.obj_type + ' · ',
        el('span', { class: 'meta' }, c.kind + ' · ' + (c.kg_code || '—') + aliasNote), agg),
      el('div', { class: 'meta' }, meta.join(' · ') + ' · ' + fmtCoord(c.centroid_lon) + ',' + fmtCoord(c.centroid_lat))
    );
  }

  function renderPrediction(act, pred) {
    if (!pred) return null;
    const lines = [];
    if (act === 'confirm') {
      lines.push(el('div', null,
        'Adds your weight to the prediction (',
        el('b', null, pred.predicted_type || '?'), '). ',
        'Current confirms: ', el('b', null, String(pred.current.n_confirms||0)),
        ' → after submit: ', el('b', null, String(pred.n_confirms_after||1)), '.'));
      if ((pred.n_confirms_after||0) >= 2) lines.push(el('div', { class: 'verify' },
        '→ community-verified after this confirm.'));
    } else if (act === 'reject') {
      lines.push(el('div', null,
        'Records rejection. Current rejections: ', el('b', null, String(pred.current.n_rejects||0)),
        ' → after submit: ', el('b', null, String(pred.n_rejects_after||1)), '.'));
      if (pred.flips_outcome) lines.push(el('div', { class: 'flip' },
        '→ enough rejections to flag as low-quality + queue for resampling.'));
      else lines.push(el('div', null, 'Will not yet flip outcome (need 2+ unless trusted).'));
    } else if (act === 'correct_type') {
      const eff = pred.projected_effective_type;
      lines.push(el('div', null,
        'Suggests ‘', el('b', null, (pred.kind === 'correct_type' ? (pred.corrected_type || '?') : '?')),
        '’ instead of ‘', el('b', null, pred.predicted_type || '?'), '’.'));
      if (eff && eff !== pred.predicted_type) lines.push(el('div', { class: 'flip' },
        '→ community-effective type would become ', el('b', null, eff), '.'));
      else lines.push(el('div', null, 'Need 2+ students agreeing OR 1 trusted reviewer to override.'));
    }
    if (pred.n_flags) lines.push(el('div', { class: 'meta' },
      pred.n_flags + ' rule flag(s) already weight ' + (pred.flag_weight||0).toFixed(1) + '.'));
    const wrap = el('div', { class: 'pred' });
    lines.forEach(l => wrap.appendChild(l));
    return wrap;
  }

  function renderMatch(res, info) {
    const wrap = el('div');
    const title = res.status === 'no_object'
      ? 'No matching object found'
      : (res.status === 'ambiguous' ? 'Multiple candidates' : 'Matched object');
    const titleRow = el('div', { class: 'row',
      style: { justifyContent: 'space-between', alignItems: 'center', margin: '0 0 6px' } },
      el('h4', { style: { margin: 0 } }, title));
    // Build a query-explorer link for the top candidate, if any.
    const top = (res.candidates && res.candidates[0]) || null;
    if (top && (top.obj_ref || (top.centroid_lon != null && top.centroid_lat != null))) {
      const params = new URLSearchParams();
      if (top.obj_ref) params.set('obj_ref', top.obj_ref);
      if (top.kg_code) params.set('kg', top.kg_code);
      if (top.centroid_lon != null) params.set('lon', top.centroid_lon);
      if (top.centroid_lat != null) params.set('lat', top.centroid_lat);
      if (top.obj_type) params.set('type', top.obj_type);
      const href = '/query.html?' + params.toString();
      const link = el('a', { href: href, target: '_blank', rel: 'noopener',
        title: 'Open this object in Query Explorer (filtered, centred + zoomed on map)',
        style: { color: '#58a6ff', fontSize: '14px', textDecoration: 'none',
                 padding: '2px 6px', borderRadius: '4px',
                 border: '1px solid #30363d', background: '#0d1117' } },
        '↗');
      titleRow.appendChild(link);
    }
    wrap.appendChild(titleRow);
    wrap.appendChild(el('div', { class: 'row' },
      el('span', { class: 'lb' }, 'Selected:'),
      el('span', { class: 'vl' }, '“' + (info.selectedText || '') + '”')));
    if (res.hint && Object.keys(res.hint).length) {
      const h = res.hint;
      const parts = [];
      if (h.predicted_type) parts.push('type=' + h.predicted_type);
      if (h.height_max_m != null) parts.push('h≈' + h.height_max_m + 'm');
      if (h.area_sqm != null) parts.push('A≈' + Math.round(h.area_sqm) + 'm²');
      wrap.appendChild(el('div', { class: 'row' },
        el('span', { class: 'lb' }, 'Parsed:'),
        el('span', { class: 'vl' }, parts.join(' · ') || '—')));
    }
    if (!res.candidates || !res.candidates.length) {
      wrap.appendChild(el('div', { class: 'err' }, 'Nothing in our index matches that snippet.'));
      return wrap;
    }
    const cands = el('div', { class: 'cands' });
    res.candidates.forEach((c, i) => {
      const node = renderCandidate(c);
      if (i === 0) node.classList.add('sel');
      node.addEventListener('click', () => {
        cands.querySelectorAll('.cand').forEach(n => n.classList.remove('sel'));
        node.classList.add('sel');
      });
      cands.appendChild(node);
    });
    wrap.appendChild(cands);

    // Flags on the top candidate (dedup by code, keep highest severity + count)
    if (res.flags && res.flags.length) {
      const SEV_RANK = { error: 3, warning: 2, info: 1 };
      const byCode = new Map();
      res.flags.forEach(f => {
        const cur = byCode.get(f.flag_code);
        if (!cur || (SEV_RANK[f.severity]||0) > (SEV_RANK[cur.severity]||0)) {
          byCode.set(f.flag_code, { severity: f.severity, count: (cur ? cur.count : 0) + 1 });
        } else {
          cur.count += 1;
        }
      });
      const fb = el('div', { class: 'row' }, el('span', { class: 'lb' }, 'Flags:'));
      byCode.forEach((v, code) => {
        const label = v.count > 1 ? code + ' ×' + v.count : code;
        fb.appendChild(el('span', { class: 'flag-badge flag-' + v.severity }, label));
      });
      wrap.appendChild(fb);
    }

    // Form
    const sel = el('select');
    sel.appendChild(el('option', { value: 'reject' }, 'reject prediction'));
    sel.appendChild(el('option', { value: 'confirm' }, 'confirm prediction'));
    sel.appendChild(el('option', { value: 'correct_type' }, 'correct type →'));
    const cor = el('select');
    TYPES.forEach(t => cor.appendChild(el('option', { value: t }, t)));
    cor.style.display = 'none';
    const predBox = el('div');
    function refreshPrediction() {
      const action = sel.value;
      // Server-rendered prediction provided pre-bake; for correct_type, also
      // re-fetch with the chosen target so the projection reflects this user's vote.
      let pred = (res.action_predictions || {})[action];
      if (action === 'correct_type') {
        const ct = cor.value;
        const ref = (cands.querySelector('.cand.sel') || {}).getAttribute
          ? cands.querySelector('.cand.sel').getAttribute('data-ref') : null;
        if (ref) {
          fetch(API_BASE + '/api/v1/flags/predict?obj_ref=' + encodeURIComponent(ref)
              + '&kind=correct_type&corrected_type=' + encodeURIComponent(ct))
            .then(r => r.json()).then(p => {
              p.kind = 'correct_type'; p.corrected_type = ct;
              const node = renderPrediction('correct_type', p);
              predBox.replaceChildren(); if (node) predBox.appendChild(node);
            }).catch(()=>{});
          return;
        }
      }
      const node = renderPrediction(action, pred);
      predBox.replaceChildren();
      if (node) predBox.appendChild(node);
    }
    sel.addEventListener('change', () => {
      cor.style.display = (sel.value === 'correct_type') ? 'block' : 'none';
      refreshPrediction();
    });
    cor.addEventListener('change', refreshPrediction);
    const notes = el('textarea', { placeholder: 'Notes (optional)' });
    const status = el('div', { class: 'row' });
    const submit = el('button', null, 'Submit');
    const cancel = el('button', { class: 'alt' }, 'Cancel');
    cancel.addEventListener('click', closePop);
    submit.addEventListener('click', () => {
      const chosenRef = (cands.querySelector('.cand.sel') || {}).getAttribute
        ? cands.querySelector('.cand.sel').getAttribute('data-ref') : null;
      const payload = {
        obj_ref: chosenRef,
        kg_code: res.kg_code,
        kind: sel.value,
        corrected_type: sel.value === 'correct_type' ? cor.value : null,
        notes: notes.value,
        context_text: info.selectedText,
        source_app: 'flag.js',
      };
      submit.disabled = true; status.textContent = 'Submitting…'; status.className = 'row';
      fetch(API_BASE + '/api/v1/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).then(r => r.json()).then(j => {
        if (j.ok) {
          status.innerHTML = '<span class="ok">Recorded — thank you. id=' + j.id + '</span>';
          emit('reported', { payload, response: j });
          setTimeout(closePop, 1200);
        } else {
          status.innerHTML = '<span class="err">' + (j.error || 'Failed') + '</span>';
          submit.disabled = false;
        }
      }).catch(e => {
        status.innerHTML = '<span class="err">' + e + '</span>'; submit.disabled = false;
      });
    });
    wrap.appendChild(el('div', { class: 'row' },
      el('span', { class: 'lb' }, 'Action:'), sel));
    wrap.appendChild(cor);
    wrap.appendChild(predBox);
    refreshPrediction();
    wrap.appendChild(notes);
    wrap.appendChild(el('div', { class: 'btns' }, cancel, submit));
    wrap.appendChild(status);
    return wrap;
  }

  // ----- Programmatic open ---------------------------------------------
  function openFor(opts) {
    ensureStyles();
    const text = opts.text || '';
    const ctx = { kg_code: opts.kg_code, lon: (opts.point||{}).lon, lat: (opts.point||{}).lat };
    const rect = opts.anchorRect || { left: window.innerWidth/2 - 200, top: window.innerHeight/2 - 100,
                                      bottom: window.innerHeight/2 + 100, right: window.innerWidth/2 + 200,
                                      width: 400, height: 0 };
    if (opts.obj_ref) {
      // Direct: skip matching
      fetch(API_BASE + '/api/v1/flags/object/' + encodeURIComponent(opts.obj_ref))
        .then(r => r.json()).then(j => {
          if (j.error) {
            showPop(rect, el('div', { class: 'err' }, j.error)); return;
          }
          const fakeRes = {
            status: 'resolved', obj_ref: j.object.obj_ref, kg_code: j.object.kg_code,
            candidates: [{
              obj_ref: j.object.obj_ref, kg_code: j.object.kg_code, kind: j.object.kind,
              obj_type: j.object.obj_type, centroid_lon: j.object.centroid_lon,
              centroid_lat: j.object.centroid_lat, height_max_m: j.object.height_max_m,
              area_sqm: j.object.area_sqm, rf_confidence: j.object.rf_confidence,
              agg: j.aggregate,
            }],
            flags: j.flags, hint: {},
            action_predictions: j.predictions,
          };
          showPop(rect, renderMatch(fakeRes, { selectedText: text, ctx }));
        });
    } else {
      const loading = el('div', null, 'Matching…');
      showPop(rect, loading);
      matchText(text, ctx).then(res => {
        const body = _popBody(); if (!body) return;
        body.replaceChildren(renderMatch(res, { selectedText: text, ctx }));
      });
    }
  }

  // ----- Install --------------------------------------------------------
  let installed = false;
  function install(opts) {
    if (installed) return; installed = true;
    ensureStyles();
    let _selDebounce = null;
    document.addEventListener('selectionchange', () => {
      if (_selDebounce) clearTimeout(_selDebounce);
      _selDebounce = setTimeout(onSelectionChange, 120);
    });
    // Touch finalisation triggers a fresh evaluation after the OS has
    // settled the selection rectangle (Android sometimes fires
    // selectionchange while the user is still dragging the handles).
    ['touchend', 'pointerup'].forEach(ev =>
      document.addEventListener(ev, onSelectionSettle, { passive: true }));
    // Hide the chip when the user begins a new touch elsewhere —
    // avoids the chip ghosting on top of the native menu while the
    // user is still adjusting the selection.
    document.addEventListener('touchstart', (e) => {
      if (chipEl && chipEl.contains(e.target)) return;
      // Only hide if the touch starts outside any active selection range.
      hideChip();
    }, { passive: true });
    // Visual viewport changes (keyboard, OS toolbar) reposition popover.
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => {
        if (popEl) hideChip();
      });
    }
  }
  function uninstall() { hideChip(); closePop(); installed = false; }

  window.SrtmFlag = {
    install, uninstall, openFor, matchText, on,
    attachClickHandler(elem, opts={}) {
      elem.addEventListener('click', (e) => {
        e.preventDefault();
        const r = elem.getBoundingClientRect();
        openFor({ ...opts, anchorRect: r });
      });
    },
  };
})();
