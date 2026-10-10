// Run a collector inside a dedicated Web Worker so that timers/fetches are not
// throttled when the tab is in the background (Chrome intensive throttling
// aligns main-thread timers of hidden tabs to 1-minute wake-ups).
// Usage: __runWorker('rd', `<collector body using: emit(progress), done(result), sleep(ms), fetchT(url)>`)
window.__runWorker = (name, body) => {
  const prelude = `
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    const fetchT = async (url, ms = 30000) => { const c = new AbortController(); const t = setTimeout(() => c.abort(), ms);
      try { return await fetch(url, { signal: c.signal }); } finally { clearTimeout(t); } };
    const emit = (p) => postMessage({ type: 'progress', p });
    const done = (res) => postMessage({ type: 'done', res });
    const base = ${JSON.stringify(location.origin)};
  `;
  const blob = new Blob([prelude + '\n(async () => {' + body + '\n})().catch(e => postMessage({type: "error", e: String(e)}));'], { type: 'text/javascript' });
  const w = new Worker(URL.createObjectURL(blob));
  window.J = window.J || {};
  J[name] = { status: 'running', progress: null, res: null, worker: w };
  w.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'progress') J[name].progress = m.p;
    else if (m.type === 'done') { J[name].res = m.res; J[name].status = 'done'; }
    else if (m.type === 'error') { J[name].status = 'error: ' + m.e; }
  };
  return 'worker started: ' + name;
};
