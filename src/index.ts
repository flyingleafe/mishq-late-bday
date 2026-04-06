import { SuttaCentralClient } from "./api/client.ts";

async function main() {
  const client = new SuttaCentralClient();

  console.log("=== Smoke Test: SuttaCentral API ===\n");

  console.log("1. Fetching languages...");
  const languages = await client.getLanguages();
  console.log(`   ${languages.length} languages available`);
  const english = languages.find((l) => l.iso_code === "en");
  console.log(`   English: ${english?.name} (root: ${english?.is_root})\n`);

  console.log("2. Fetching sutta collection manifest...");
  const collection = await client.getCollection("sutta", ["en"]);
  console.log(`   ${collection.texts.length} texts`);
  console.log(`   ${collection.suttaplex.length} suttaplex entries`);
  console.log(`   ${collection.menu.length} menu entries`);
  const sample = collection.texts[0];
  if (sample) {
    console.log(
      `   First entry: ${sample.uid} (${sample.translations.length} translation(s))\n`,
    );
  }

  console.log("3. Fetching DN1 (Brahmajāla Sutta)...");
  const sutta = await client.getSutta("dn1", "sujato");
  console.log(`   Title: ${sutta.translation.title}`);
  console.log(`   Author: ${sutta.translation.author}`);
  console.log(`   Root lang: ${sutta.suttaplex.root_lang_name}`);
  console.log(`   Translations: ${sutta.suttaplex.translations.length}`);
  const textLen = sutta.translation.text?.length ?? 0;
  console.log(`   Text length: ${textLen} chars\n`);

  console.log("4. Fetching suttaplex for DN...");
  const suttaplex = await client.getSuttaplex("dn");
  console.log(`   ${suttaplex.length} entries in DN collection\n`);

  console.log("5. Fetching parallels for DN1...");
  const parallels = await client.getParallels("dn1");
  const groups = Object.keys(parallels);
  console.log(`   ${groups.length} parallel group(s)\n`);

  console.log("=== All checks passed ===");
}

main().catch(console.error);
