// GDELT DOC 2.0 daily average-tone timelines for each topic query (config/topic_queries.yaml).
// Run on an api.gdeltproject.org page after bridge.js; QUERIES is pasted from the YAML file.
// Each full-period timeline request takes ~25 s server-side; requests are serialised with a
// >= 6 s pause (GDELT asks for <= 1 request / 5 s) and retried on the rate-limit message.
// Python equivalent: src/midterm/collect/gdelt.py::collect_tone
const runTone = (QUERIES) => {
  J.tone2 = { status: 'running', i: 0, errs: [], out: { meta: { source: 'api.gdeltproject.org/api/v2/doc/doc mode=timelinetone/timelinevolraw', start: '20251103000000', end: '20261008000000', retrieved: new Date().toISOString() }, series: {} } };
  (async () => {
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    for (const [key, q] of Object.entries(QUERIES)) {
      const full = q + ' sourcecountry:US sourcelang:english';
      J.tone2.out.series[key] = { query: full };
      for (const mode of ['timelinetone']) {   // (the first topic also has timelinevolraw from an earlier pass)
        const url = '/api/v2/doc/doc?query=' + encodeURIComponent(full) + '&mode=' + mode + '&format=json&startdatetime=20251103000000&enddatetime=20261008000000';
        for (let a = 0; a < 12; a++) {
          try {
            const tx = await (await fetch(url)).text();
            if (!tx.trim().startsWith('{')) throw new Error(tx.slice(0, 50));
            const j = JSON.parse(tx);
            J.tone2.out.series[key][mode] = j.timeline[0].data;
            J.tone2.out.series[key][mode + '_resolution'] = j.query_details && j.query_details.date_resolution;
            break;
          } catch (e) { J.tone2.errs.push([key, mode, String(e.message).slice(0, 50)]); await sleep(10000); }
        }
        J.tone2.i++;
        await sleep(6000);
      }
    }
    J.tone2.status = 'done';
  })();
};
