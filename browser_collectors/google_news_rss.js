// Google News RSS search (US English edition) — weekly windows via the after:/before: operators.
// Run on a news.google.com page after bridge.js. news.google.com enforces a CSP that forbids
// blob workers, so this collector runs on the main thread (keep the tab in the foreground).
// Each request returns up to 100 items (title, publisher, pubDate, link).
// Python equivalent: src/midterm/collect/google_news.py::collect
(() => {
  const Q = { midterms: 'midterms', voters_poll: 'voters poll', campaign: 'midterm campaign',
              senate_race: 'Senate race', house_race: 'House race', voters_say: 'voters say' };  // issue-neutral queries
  // (the reported run executed the first three and the last three as two consecutive passes)
  const day = (ms) => new Date(ms).toISOString().slice(0, 10);
  const start = Date.parse('2025-11-03T00:00:00Z'), end = Date.parse('2026-10-08T00:00:00Z');
  const wins = [];
  for (let s = start; s < end; s += 7 * 864e5) wins.push([s, Math.min(s + 7 * 864e5, end)]);
  const g = (s, tag) => { const m = s.match(new RegExp('<' + tag + '[^>]*>([\\s\\S]*?)</' + tag + '>')); return m ? m[1] : null; };
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  J.gn = { status: 'running', progress: null, res: null };
  const st = { i: 0, n: wins.length * Object.keys(Q).length, errors: 0, items: [], errlog: [] };
  (async () => {
    for (const [s, e] of wins) for (const [qk, q] of Object.entries(Q)) {
      const url = '/rss/search?q=' + encodeURIComponent(q + ' after:' + day(s) + ' before:' + day(e)) + '&hl=en-US&gl=US&ceid=US:en';
      for (let a = 0; a < 6; a++) {
        try {
          const r = await fetch(url); if (r.status !== 200) throw new Error(r.status);
          const t = await r.text();
          for (const m of t.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
            const it = m[1];
            st.items.push({ q: qk, win: day(s), title: g(it, 'title'), pubDate: g(it, 'pubDate'), source: g(it, 'source'), link: g(it, 'link') });
          }
          break;
        } catch (err) { st.errors++; st.errlog.push(String(err.message)); await sleep(15000); }
      }
      st.i++; J.gn.progress = { i: st.i, n: st.n, items: st.items.length, errors: st.errors };
      await sleep(800);
    }
    J.gn.res = { meta: { source: 'news.google.com/rss/search hl=en-US gl=US', queries: Q, window_days: 7, retrieved: new Date().toISOString(), errors: st.errlog }, items: st.items };
    J.gn.status = 'done';
  })();
  return 'started ' + st.n;
})();
