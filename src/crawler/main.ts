import { crawl } from "./crawl.ts";

const outputDir = process.argv[2] ?? "./data";
const concurrency = parseInt(process.argv[3] ?? "3", 10);

console.log(`Crawling SuttaCentral → ${outputDir} (concurrency: ${concurrency})\n`);

const startTime = Date.now();
let lastLog = 0;

const results = await crawl(
  { outputDir, concurrency },
  (progress) => {
    const now = Date.now();
    if (now - lastLog > 2000) {
      const pct = ((progress.completed / progress.total) * 100).toFixed(1);
      console.log(
        `[${pct}%] ${progress.completed}/${progress.total} done, ${progress.failed} failed — ${progress.currentUid}`,
      );
      lastLog = now;
    }
  },
);

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(
  `\nDone. ${results.length} texts crawled in ${elapsed}s. Output: ${outputDir}/`,
);
