const MATCH = 2;
const WS_MATCH = 0.5;
const MISMATCH = -4;
const GAP_NONWS = -2;
const GAP_WS = 0;
const BEAM = 20;

export interface AlignmentResult {
  tStart: number;
  tEnd: number;
  score: number;
}

function isWs(ch: string): boolean {
  return ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r';
}

function decodeHtml(text: string): string {
  return text
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&apos;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&nbsp;/g, ' ');
}

function collapseWs(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function viterbiAlign(S: string, T: string): AlignmentResult | null {
  type State = { score: number; si: number; ti: number; parentKey: string | null; op: string; hi: number };

  let beams = new Map<string, State>();
  beams.set('0,0', { score: 0, si: 0, ti: 0, parentKey: null, op: '', hi: 0 });
  const history: Map<string, State>[] = [new Map(beams)];

  const maxIter = (S.length + T.length) * 2;
  for (let iter = 1; iter < maxIter; iter++) {
    const next = new Map<string, State>();

    for (const [key, st] of beams) {
      const [si, ti] = key.split(',').map(Number);

      if (si < S.length && ti < T.length) {
        const sc =
          S[si] === T[ti]
            ? isWs(S[si])
              ? WS_MATCH
              : MATCH
            : isWs(S[si]) && isWs(T[ti])
              ? WS_MATCH
              : MISMATCH;
        const nk = `${si + 1},${ti + 1}`;
        if (!next.has(nk) || next.get(nk)!.score < st.score + sc) {
          next.set(nk, { score: st.score + sc, si: si + 1, ti: ti + 1, parentKey: key, op: 'M', hi: iter });
        }
      }

      if (ti < T.length) {
        const penalty = isWs(T[ti]) ? GAP_WS : GAP_NONWS;
        const nk = `${si},${ti + 1}`;
        if (!next.has(nk) || next.get(nk)!.score < st.score + penalty) {
          next.set(nk, { score: st.score + penalty, si, ti: ti + 1, parentKey: key, op: 'DT', hi: iter });
        }
      }

      if (si < S.length) {
        const penalty = isWs(S[si]) ? GAP_WS : GAP_NONWS;
        const nk = `${si + 1},${ti}`;
        if (!next.has(nk) || next.get(nk)!.score < st.score + penalty) {
          next.set(nk, { score: st.score + penalty, si: si + 1, ti, parentKey: key, op: 'DS', hi: iter });
        }
      }
    }

    if (next.size === 0) break;
    history.push(next);
    const sorted = [...next.entries()].sort((a, b) => b[1].score - a[1].score);
    beams = new Map(sorted.slice(0, BEAM));
  }

  let bestKey = '';
  let bestSt: State | null = null;
  let bestScore = -Infinity;
  let bestHi = 0;
  for (let hi = 0; hi < history.length; hi++) {
    for (const [key, st] of history[hi]) {
      const [si] = key.split(',').map(Number);
      if (si === S.length && st.score > bestScore) {
        bestScore = st.score;
        bestKey = key;
        bestSt = st;
        bestHi = hi;
      }
    }
  }

  if (!bestSt) return null;

  const tPositions: number[] = [];
  let curKey = bestKey;
  let chi = bestHi;
  while (true) {
    let st: State | undefined;
    if (chi === 0) st = history[0].get(curKey);
    else st = history[chi].get(curKey);
    if (!st || !st.parentKey) break;
    if (st.op === 'M' && st.ti > 0) tPositions.push(st.ti - 1);
    curKey = st.parentKey;
    chi--;
  }

  if (tPositions.length === 0) return null;
  const tStart = Math.min(...tPositions);
  const tEnd = Math.max(...tPositions) + 1;
  if (tEnd <= tStart || tEnd > T.length) return null;
  return { tStart, tEnd, score: bestScore };
}

export function alignChunkToSutta(
  chunkText: string,
  suttaText: string,
  debug = false
): AlignmentResult | null {
  if (!chunkText || !suttaText) return null;

  const chunkDec = decodeHtml(chunkText);
  const suttaDec = decodeHtml(suttaText);

  if (debug) {
    console.log('[align] chunkDec len:', chunkDec.length, JSON.stringify(chunkDec.slice(0, 80)));
    console.log('[align] suttaDec len:', suttaDec.length, JSON.stringify(suttaDec.slice(0, 80)));
  }

  // Fast path: exact substring search on decoded text
  const directIdx = suttaDec.indexOf(chunkDec);
  if (directIdx !== -1) {
    if (debug) console.log('[align] direct match at', directIdx);
    return { tStart: directIdx, tEnd: directIdx + chunkDec.length, score: 999 };
  }

  // Viterbi with candidate pre-search: find approximate match regions first,
  // then run focused Viterbi to find precise boundaries
  const PREFIX_LEN = 40;
  const prefix = chunkDec.slice(0, PREFIX_LEN);
  const candidatePositions: number[] = [];
  let searchFrom = 0;
  while (true) {
    const pos = suttaDec.indexOf(prefix, searchFrom);
    if (pos === -1) break;
    candidatePositions.push(pos);
    searchFrom = pos + 1;
  }

  if (debug) console.log('[align] found', candidatePositions.length, 'candidate positions for prefix of len', PREFIX_LEN);

  let bestResult: AlignmentResult | null = null;
  if (candidatePositions.length > 0) {
    // Run Viterbi in a window around each candidate
    const WINDOW = Math.max(chunkDec.length * 2, 600);
    for (const cand of candidatePositions) {
      const winStart = Math.max(0, cand - WINDOW);
      const winEnd = Math.min(suttaDec.length, cand + WINDOW);
      const T_window = suttaDec.slice(winStart, winEnd);
      const result = viterbiAlign(chunkDec, T_window);
      if (result && (bestResult === null || result.score > bestResult.score)) {
        bestResult = { ...result, tStart: result.tStart + winStart, tEnd: result.tEnd + winStart };
        if (debug) console.log('[align] candidate at', cand, 'gave score', result.score);
      }
    }
  }

  if (bestResult) {
    if (debug) console.log('[align] best windowed result:', bestResult);
    return bestResult;
  }

  // Full Viterbi as last resort
  const result = viterbiAlign(chunkDec, suttaDec);
  if (debug && result) {
    console.log('[align] viterbi:', result, '→', JSON.stringify(suttaDec.slice(result.tStart, result.tEnd).slice(0, 80)));
  }
  return result;
}

export function wrapRange(text: string, start: number, end: number): string {
  if (start >= end || start < 0 || end > text.length) return text;
  return text.slice(0, start) + '<mark id="matching-chunk">' + text.slice(start, end) + '</mark>' + text.slice(end);
}

export function highlightChunk(chunkText: string, suttaText: string, debug = false): string {
  if (!chunkText || !suttaText) return suttaText;

  const suttaDec = decodeHtml(suttaText);
  const result = alignChunkToSutta(chunkText, suttaText, debug);
  if (!result) return suttaDec;

  return wrapRange(suttaDec, result.tStart, result.tEnd);
}
