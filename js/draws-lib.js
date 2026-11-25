// The shared draw cache of the GPU server (pure parts; main.js calls them from sampleDraws).
// Active only for our GPU server (id server or server:<key>) and only while its GET /status says `drawCache: true`.
// Then a request for draws carries `offset`, the number of draws the page's buffer already took from the server's stream
// (precomputed draws at the start of a buffer are not in the stream), and the answer is those stream positions. Without the field
// the server samples with the request's seed, as before, and never stores the draws.

/** True when the status body offers the draw cache and the backend is the GPU server. */
export const drawCacheOn = (backendId, status) => /^server(:|$)/.test(String(backendId ?? '')) && status?.drawCache === true;

/** The /sample body with `offset` (and without `seed`, which the server ignores for stream requests) when the cache is on. */
export function withOffset(body, { on, position }) {
  if (!on || !Number.isInteger(position) || position < 0) return body;
  const { seed, ...rest } = body;   // eslint-disable-line no-unused-vars
  return { ...rest, offset: position };
}

/** True when an answer to a request that carried `offset` does not say it holds those positions. */
export const offsetMismatch = (out, { on, position }) => on && Number.isInteger(position) && out?.offset !== position;
