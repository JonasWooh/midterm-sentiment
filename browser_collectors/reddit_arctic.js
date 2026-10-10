// Reddit posts via the Arctic Shift archive API. Run on an arctic-shift.photon-reddit.com
// page after bridge.js and worker_runner.js:  __runWorker('rd', BODY).
// Sampling design (respects the archive's per-minute compute budget, ~180 requests):
//   * r/politics (~350 posts/day): the 100 most recent posts before 00:00 UTC, one day per week,
//   * r/Conservative: all posts, 10-day windows,
//   * r/AskConservatives, r/AskALiberal: all posts, 14-day windows,
//   * r/democrats, r/Republican, r/moderatepolitics, r/PoliticalDiscussion: all posts, 28-day windows.
// Jobs are interleaved in time (round-robin over communities) so that any prefix of
// the run is balanced. Each post keeps n_win (= posts returned for its window).
// Windows that hit the 1000-post cap (6 r/Conservative windows) were completed afterwards by
// paging backwards (before = earliest returned post), see BACKFILL at the end of this file.
// Python equivalent: src/midterm/collect/reddit_arctic.py::collect_posts
const BODY = `
  const SUBS = { politics: [7, 'last100'], Conservative: [10, 'all'], AskConservatives: [14, 'all'], AskALiberal: [14, 'all'],
    democrats: [28, 'all'], Republican: [28, 'all'], moderatepolitics: [28, 'all'], PoliticalDiscussion: [28, 'all'] };
  const FIELDS = 'id,created_utc,title,selftext,score,num_comments,subreddit,link_flair_text';
  const start = Date.parse('2025-11-03T00:00:00Z'), end = Date.parse('2026-10-08T00:00:00Z');
  let jobs = [];
  for (const [sub, [wd, mode]] of Object.entries(SUBS)) for (let s = start; s < end; s += wd * 864e5)
    jobs.push([sub, mode, s, mode === 'last100' ? s + 864e5 : Math.min(s + wd * 864e5, end)]);
  jobs.sort((a, b) => a[2] - b[2]);
  const st = { i: 0, n: jobs.length, waits: 0, errors: [], truncated: [], posts: [] };
  for (const [sub, mode, s, e] of jobs) {
    const url = base + '/api/posts/search?subreddit=' + sub + '&after=' + new Date(s).toISOString() + '&before=' + new Date(e).toISOString()
      + '&limit=' + (mode === 'all' ? 'auto' : '100') + '&sort=desc&fields=' + FIELDS;
    let ok = false;
    for (let a = 0; a < 20 && !ok; a++) {
      try {
        const r = await fetchT(url, 60000); const j = await r.json();
        if (!j.data) { st.waits++; await sleep((Number(r.headers.get('x-ratelimit-reset') || 15) + 1) * 1000); continue; }
        if (mode === 'all' && j.data.length >= 1000) st.truncated.push([sub, s]);
        for (const p of j.data) st.posts.push({ id: p.id, t: p.created_utc, sub: p.subreddit, title: p.title, text: (p.selftext || '').slice(0, 400),
          score: p.score, nc: p.num_comments, flair: p.link_flair_text, win: s / 1000, n_win: j.data.length, mode });
        ok = true;
      } catch (err) { st.errors.push([sub, s, a, String(err).slice(0, 60)]); await sleep(5000); }
    }
    if (!ok) st.errors.push([sub, s, 'gave_up']);
    st.i++; emit({ i: st.i, n: st.n, posts: st.posts.length, waits: st.waits, errors: st.errors.length, truncated: st.truncated.length });
    await sleep(1000);
  }
  done({ meta: { source: 'arctic-shift.photon-reddit.com/api/posts/search', subs: SUBS, selftext_max_chars: 400, retrieved: new Date().toISOString(),
    errors: st.errors, truncated: st.truncated }, posts: st.posts });
`;

// BACKFILL for capped windows: for each [sub, windowStart] in meta.truncated, request
// [windowStart, earliest post already returned) with limit=auto, paging again if 1000 are returned.
