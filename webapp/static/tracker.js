// ShopLens clickstream tracker — same event schema as the synthetic dataset.
// Explicit:  track("event_type", {props})
// Declarative: <button data-track="reviews_opened" data-props='{"k":1}'>
(function () {
  const ENDPOINT = "/collect";
  const body = document.body.dataset;
  const ctx = { page_type: body.page || null, product_id: body.product || null, category: body.category || null };
  const queue = [];
  const startedAt = Date.now();

  function cookie(name) {
    const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
    return m ? decodeURIComponent(m[1]) : null;
  }
  function refreshSession() {
    const sid = cookie("sid");
    if (sid) document.cookie = "sid=" + sid + "; max-age=1800; path=/; samesite=lax";
  }
  function uid() {
    return (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now()).replace(/-/g, "");
  }

  function track(event_type, props = {}, opts = {}) {
    queue.push({ event_id: uid(), event_type, ts: new Date().toISOString(), ...ctx, props });
    refreshSession();
    if (opts.beacon) flushBeacon();
    else if (opts.now || queue.length >= 10) flush();
  }

  async function flush() {
    if (!queue.length) return;
    const batch = queue.splice(0);
    try {
      const r = await fetch(ENDPOINT, {
        method: "POST", body: JSON.stringify(batch), keepalive: true,
        headers: { "Content-Type": "application/json" }, credentials: "same-origin",
      });
      const data = await r.json().catch(() => ({}));
      if (data && data.intervention) showNudge(data.intervention);
    } catch (_) { /* offline: drop */ }
  }
  function flushBeacon() {
    if (!queue.length) return;
    const blob = new Blob([JSON.stringify(queue.splice(0))], { type: "application/json" });
    navigator.sendBeacon(ENDPOINT, blob);
  }
  setInterval(flush, 3000);
  addEventListener("pagehide", () => {
    track("page_exit", { dwell_ms: Date.now() - startedAt });
    flushBeacon();
  });

  // ---- automatic capture ----
  track("page_view", Object.assign({ referrer: document.referrer ? new URL(document.referrer).pathname : "direct" },
    JSON.parse(body.viewProps || "{}")), { now: true });

  let hiddenAt = null;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { hiddenAt = Date.now(); track("tab_hidden", {}, { beacon: true }); }
    else { track("tab_visible", { away_ms: hiddenAt ? Date.now() - hiddenAt : 0 }, { now: true }); }
  });

  const marks = new Set();
  addEventListener("scroll", () => {
    const h = document.documentElement;
    const pct = Math.round(((h.scrollTop + innerHeight) / h.scrollHeight) * 100);
    for (const m of [25, 50, 75, 100]) {
      if (pct >= m && !marks.has(m)) { marks.add(m); track("scroll_depth", { pct: m }); }
    }
  }, { passive: true });

  let clicks = [];
  document.addEventListener("click", (e) => {
    const t = Date.now();
    clicks = clicks.filter((c) => t - c.t < 1000 && Math.hypot(c.x - e.clientX, c.y - e.clientY) < 30);
    clicks.push({ t, x: e.clientX, y: e.clientY });
    if (clicks.length >= 3) { track("rage_click", { target: (e.target.closest("[data-track]") || e.target).tagName }); clicks = []; }

    const el = e.target.closest("[data-track]");
    if (!el) return;
    let props = {};
    try { props = JSON.parse(el.dataset.props || "{}"); } catch (_) {}
    const navigates = el.tagName === "A" && el.getAttribute("href") && !el.getAttribute("href").startsWith("#");
    track(el.dataset.track, props, navigates ? { beacon: true } : {});
  }, true);

  document.addEventListener("change", (e) => {
    const el = e.target.closest("[data-track-change]");
    if (el) track(el.dataset.trackChange, { [el.name || "value"]: el.value });
  });

  // time spent with reviews in view -> review_dwell
  const reviews = document.getElementById("reviews");
  if (reviews && "IntersectionObserver" in window) {
    let since = null;
    new IntersectionObserver(([en]) => {
      if (en.isIntersecting) since = Date.now();
      else if (since) { const ms = Date.now() - since; since = null; if (ms > 2000) track("review_dwell", { ms }); }
    }, { threshold: 0.4 }).observe(reviews);
  }

  // ---- recovery assistant nudge ----
  function showNudge(n) {
    const box = document.getElementById("nudge");
    if (!box) return;
    box.querySelector("[data-n=title]").textContent = n.title;
    box.querySelector("[data-n=body]").textContent = n.body;
    const why = box.querySelector("[data-n=why]");
    why.textContent = n.evidence && n.evidence.length ? "Why you're seeing this: " + n.evidence.join("; ") : "";
    const a = box.querySelector("[data-n=link]");
    if (n.link) { a.href = n.link; a.classList.remove("hidden"); } else a.classList.add("hidden");
    box.dataset.cause = n.cause;
    box.classList.remove("hidden");
  }
  document.addEventListener("click", (e) => {
    const close = e.target.closest("[data-n=close]");
    if (close) {
      const box = document.getElementById("nudge");
      track("intervention_dismissed", { cause: box.dataset.cause });
      box.classList.add("hidden");
    }
    const link = e.target.closest("[data-n=link]");
    if (link) track("intervention_clicked", { cause: document.getElementById("nudge").dataset.cause }, { beacon: true });
  });

  window.track = track;
})();
