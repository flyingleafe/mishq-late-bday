/**
 * Type definitions for SuttaCentral API responses.
 * Based on the Swagger spec at https://suttacentral.net/api/spec
 */

// -- /api/menu --

export interface MenuItem {
  uid: string;
  root_name?: string;
  translated_name?: string;
  node_type?: string;
  blurb?: string;
  acronym?: string;
  root_lang_iso?: string;
  root_lang_name?: string;
  child_range?: string;
  yellow_brick_road?: boolean;
  children?: MenuItem[];
}

// -- /api/languages --

export interface Language {
  uid: string;
  name: string;
  iso_code: string;
  is_root: boolean;
  localized: boolean;
  localized_percent: number;
}

// -- /api/suttaplex/{uid} --

export interface TranslationListing {
  author: string;
  author_short?: string;
  author_uid?: string;
  id: string;
  lang: string;
  lang_name?: string;
  title?: string;
  is_root?: boolean;
  segmented?: boolean;
  publication_date?: string | null;
  volpage?: string | null;
  has_comment?: boolean;
}

export interface Suttaplex {
  uid: string;
  blurb: string;
  difficulty?: number | null;
  original_title: string;
  type: string;
  acronym?: string;
  volpages?: string;
  alt_volpages?: string | null;
  root_lang?: string;
  root_lang_name?: string;
  translated_title?: string;
  translations: TranslationListing[];
  parallel_count?: number;
  biblio?: unknown;
  priority_author_uid?: string;
}

// -- /api/suttas/{uid}/{author_uid} --

export interface SuttaNeighbour {
  author?: string;
  author_uid?: string;
  lang?: string;
  title?: string | null;
  name?: string | null;
  uid: string;
}

export interface RootText {
  uid: string;
  lang: string;
  is_root?: boolean;
  title: string;
  author: string;
  author_uid: string;
  author_short?: string;
  text?: string;
  previous?: SuttaNeighbour;
  next?: SuttaNeighbour;
}

export interface Translation {
  uid: string;
  lang: string;
  title: string;
  author: string;
  author_uid: string;
  author_short?: string;
  text?: string;
  previous?: SuttaNeighbour;
  next?: SuttaNeighbour;
}

export interface SuttaResponse {
  root_text: RootText;
  translation: Translation;
  segmented: boolean;
  suttaplex: Suttaplex;
  bilara_root_text?: RootText;
  bilara_translated_text?: Translation;
  candidate_authors?: string[];
}

// -- /api/pwa/collection/{collection}?languages=... --

export interface PwaTextEntry {
  uid: string;
  translations: Array<{
    lang: string;
    authors: string[];
  }>;
}

export interface PwaCollectionResponse {
  menu: string[];
  suttaplex: Suttaplex[];
  texts: PwaTextEntry[];
}

// -- /api/parallels/{uid} --

export interface Parallel {
  type: string;
  partial: boolean;
  to: Suttaplex[];
}

export type ParallelsResponse = Record<string, Parallel[]>;

// -- /api/bilarasuttas/{uid}/{author_uid} --

export type SegmentMap = Record<string, string>;

export interface BilaraResponse {
  translation_text: SegmentMap;
  root_text: SegmentMap;
  html_text: SegmentMap;
  keys_order: string[];
  comment_text?: SegmentMap;
  variant_text?: SegmentMap;
  reference_text?: SegmentMap;
}
