import pLimit from "p-limit";
import { SuttaCentralClient } from "../api/client.ts";
import type {
  BilaraResponse,
  PwaTextEntry,
  SuttaResponse,
} from "../types/suttacentral.ts";

export interface CrawlOptions {
  languages?: string[];
  concurrency?: number;
  outputDir?: string;
}

export interface CrawledText {
  uid: string;
  authorUid: string;
  lang: string;
  title: string;
  segmented: boolean;
  text: string;
}

function bilaraToPlainText(bilara: BilaraResponse): string {
  return bilara.keys_order
    .map((key) => bilara.translation_text[key])
    .filter(Boolean)
    .join("\n");
}

function htmlToPlainText(html: string): string {
  return html
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n\n")
    .replace(/<\/h[1-6]>/gi, "\n\n")
    .replace(/<\/li>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function fetchSuttaText(
  client: SuttaCentralClient,
  uid: string,
  authorUid: string,
): Promise<CrawledText | null> {
  const suttaResp: SuttaResponse = await client.getSutta(uid, authorUid);

  if (suttaResp.segmented) {
    const bilara = await client.getBilaraSutta(uid, authorUid);
    return {
      uid,
      authorUid,
      lang: suttaResp.translation.lang,
      title: suttaResp.translation.title,
      segmented: true,
      text: bilaraToPlainText(bilara),
    };
  }

  const html = suttaResp.translation.text;
  if (!html) return null;

  return {
    uid,
    authorUid,
    lang: suttaResp.translation.lang,
    title: suttaResp.translation.title,
    segmented: false,
    text: htmlToPlainText(html),
  };
}

export interface CrawlProgress {
  total: number;
  completed: number;
  failed: number;
  currentUid?: string;
}

export type ProgressCallback = (progress: CrawlProgress) => void;

export async function crawl(
  options: CrawlOptions = {},
  onProgress?: ProgressCallback,
): Promise<CrawledText[]> {
  const {
    languages = ["en"],
    concurrency = 3,
    outputDir = "./data",
  } = options;

  const client = new SuttaCentralClient();
  const limit = pLimit(concurrency);

  console.log("Fetching sutta collection manifest...");
  const collection = await client.getCollection("sutta", languages);
  const texts = collection.texts;
  console.log(`Found ${texts.length} texts to crawl.`);

  await Bun.write(`${outputDir}/manifest.json`, JSON.stringify(collection, null, 2));

  const progress: CrawlProgress = {
    total: texts.length,
    completed: 0,
    failed: 0,
  };

  const results: CrawledText[] = [];
  const errors: Array<{ uid: string; error: string }> = [];

  const tasks = texts.map((entry: PwaTextEntry) =>
    limit(async () => {
      const lang = languages[0] ?? "en";
      const authorUid =
        entry.translations.find((t) => t.lang === lang)?.authors[0];

      if (!authorUid) {
        progress.failed++;
        errors.push({ uid: entry.uid, error: "no author found" });
        return;
      }

      progress.currentUid = entry.uid;

      try {
        const result = await fetchSuttaText(client, entry.uid, authorUid);
        if (result) {
          results.push(result);
          await Bun.write(
            `${outputDir}/texts/${result.uid}.json`,
            JSON.stringify(result, null, 2),
          );
        } else {
          errors.push({ uid: entry.uid, error: "no text content" });
          progress.failed++;
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        errors.push({ uid: entry.uid, error: msg });
        progress.failed++;
      }

      progress.completed++;
      onProgress?.(progress);
    }),
  );

  await Promise.all(tasks);

  if (errors.length > 0) {
    await Bun.write(`${outputDir}/errors.json`, JSON.stringify(errors, null, 2));
    console.log(`${errors.length} errors saved to ${outputDir}/errors.json`);
  }

  return results;
}
