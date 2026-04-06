import { test, expect } from "bun:test";
import { readFileSync } from "fs";
import { alignChunkToSutta, highlightChunk, wrapRange } from "./highlight";

test("wrapRange wraps correct slice", () => {
  const text = "hello world test";
  expect(wrapRange(text, 0, 5)).toBe('<mark id="matching-chunk">hello</mark> world test');
  expect(wrapRange(text, 6, 11)).toBe('hello <mark id="matching-chunk">world</mark> test');
  expect(wrapRange(text, 0, 0)).toBe(text);
});

test("alignChunkToSutta returns null for empty inputs", () => {
  expect(alignChunkToSutta("", "hello")).toBeNull();
  expect(alignChunkToSutta("hello", "")).toBeNull();
  expect(alignChunkToSutta("", "")).toBeNull();
});

test("alignChunkToSutta finds exact match", () => {
  const result = alignChunkToSutta("hello world", "say hello world here");
  expect(result).not.toBeNull();
  expect(result!.tStart).toBe(4);
  expect(result!.tEnd).toBe(15);
});

test("alignChunkToSutta handles whitespace differences", () => {
  const result = alignChunkToSutta("hello   world", "hello world");
  expect(result).not.toBeNull();
  expect(result!.tStart).toBe(0);
  expect(result!.tEnd).toBe(11);
});

test("alignChunkToSutta handles newlines vs spaces", () => {
  const result = alignChunkToSutta("hello\nworld", "hello world");
  expect(result).not.toBeNull();
  expect(result!.tStart).toBe(0);
  expect(result!.tEnd).toBe(11);
});

test("alignChunkToSutta decodes HTML entities in chunk", () => {
  const result = alignChunkToSutta("&lt;em&gt;hello&lt;/em&gt;", "<em>hello</em>");
  expect(result).not.toBeNull();
  expect(result!.score).toBe(999);
});

test("alignChunkToSutta decodes HTML entities in sutta", () => {
  const result = alignChunkToSutta("<em>hello</em>", "&lt;em&gt;hello&lt;/em&gt;");
  expect(result).not.toBeNull();
  expect(result!.score).toBe(999);
});

test("alignChunkToSutta handles partial entity match", () => {
  const result = alignChunkToSutta(
    "Let it be so &amp; let it not be so",
    "Let it be so & let it not be so. More text here."
  );
  expect(result).not.toBeNull();
  expect(result!.tStart).toBe(0);
  expect(result!.tEnd).toBeLessThan(60);
});

test("alignChunkToSutta with em tags in chunk and sutta", () => {
  const chunk = '<em>Feeling, perception, volition, and consciousness are likewise</em>';
  const sutta = `Some text before
<em>Feeling, perception, volition, and consciousness are likewise</em>
Some text after`;
  const result = alignChunkToSutta(chunk, sutta);
  expect(result).not.toBeNull();
  expect(result!.score).toBeGreaterThan(0);
  expect(result!.tEnd).toBeGreaterThan(result!.tStart);
});

test("highlightChunk returns suttaText unchanged on failure", () => {
  const result = highlightChunk("", "hello world");
  expect(result).toBe("hello world");
});

test("highlightChunk wraps the highlighted region", () => {
  const result = highlightChunk("hello world", "say hello world here");
  expect(result).toContain('<mark id="matching-chunk">hello world</mark>');
  expect(result).toContain('say ');
  expect(result).toContain(' here');
});

test("highlightChunk handles mixed HTML entities", () => {
  const chunk = '&lt;em&gt;Feeling, perception, volition, and consciousness are likewise&lt;/em&gt;';
  const sutta = `text <em>Feeling, perception, volition, and consciousness are likewise</em> more text`;
  const result = highlightChunk(chunk, sutta);
  expect(result).toContain('<mark');
  expect(result).toContain('</mark>');
});

test("alignChunkToSutta handles the sa34 real-world case", () => {
  const sa34Data = JSON.parse(readFileSync("data/texts/sa34.json", "utf8"));
  const suttaText = sa34Data.text;

  const chunkStart = 450;
  const chunkText = suttaText.slice(chunkStart, chunkStart + 200);

  const result = alignChunkToSutta(chunkText, suttaText);
  expect(result).not.toBeNull();
  expect(result!.score).toBeGreaterThan(0);
  expect(result!.tEnd).toBeGreaterThan(result!.tStart);
  expect(result!.tStart).toBe(chunkStart);
  expect(result!.tEnd).toBe(chunkStart + 200);
});
