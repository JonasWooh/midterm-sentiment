// Browser-side half of the bridge protocol (see src/midterm/bridge/ingest.py).
// Paste once per tab, then: __stage(obj) -> [length, fnv1a32, nChunks];
// __chunk(i) -> '#CHUNK<i>#' + fixed-width slice.
window.__C = 160000;
window.__fnv = (s) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h;
};
window.__stage = (obj) => {
  window.__buf = JSON.stringify(obj);
  return [window.__buf.length, window.__fnv(window.__buf), Math.ceil(window.__buf.length / window.__C)];
};
window.__chunk = (i) => '#CHUNK' + i + '#' + window.__buf.slice(i * window.__C, (i + 1) * window.__C).padEnd(window.__C, ' ');
// Long-running jobs: run detached and poll window.J[name].status to avoid the
// 45 s tool timeout.
window.J = window.J || {};
window.__sleep = (ms) => new Promise((r) => setTimeout(r, ms));
