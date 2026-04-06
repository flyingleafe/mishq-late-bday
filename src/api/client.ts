import type {
  MenuItem,
  SuttaResponse,
  BilaraResponse,
  Suttaplex,
  Language,
  PwaCollectionResponse,
  ParallelsResponse,
} from "../types/suttacentral.ts";
import { fetchWithRetry } from "../utils/rate-limit.ts";

const BASE_URL = "https://suttacentral.net";

export interface ClientOptions {
  baseUrl?: string;
}

export class SuttaCentralClient {
  private baseUrl: string;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? BASE_URL;
  }

  private async fetchJson<T>(path: string): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetchWithRetry(url);

    if (!response.ok) {
      throw new Error(
        `API error: ${response.status} ${response.statusText} for ${url}`,
      );
    }

    return response.json() as Promise<T>;
  }

  async getMenu(): Promise<MenuItem[]> {
    return this.fetchJson<MenuItem[]>("/api/menu");
  }

  async getLanguages(): Promise<Language[]> {
    return this.fetchJson<Language[]>("/api/languages");
  }

  async getSutta(uid: string, authorUid: string): Promise<SuttaResponse> {
    return this.fetchJson<SuttaResponse>(
      `/api/suttas/${uid}/${authorUid}`,
    );
  }

  async getSuttaplex(uid: string): Promise<Suttaplex[]> {
    return this.fetchJson<Suttaplex[]>(`/api/suttaplex/${uid}`);
  }

  async getCollection(
    collection: string,
    languages: string[] = ["en"],
  ): Promise<PwaCollectionResponse> {
    const langs = languages.join(",");
    return this.fetchJson<PwaCollectionResponse>(
      `/api/pwa/collection/${collection}?languages=${langs}`,
    );
  }

  async getParallels(uid: string): Promise<ParallelsResponse> {
    return this.fetchJson<ParallelsResponse>(`/api/parallels/${uid}`);
  }

  async getBilaraSutta(uid: string, authorUid: string): Promise<BilaraResponse> {
    return this.fetchJson<BilaraResponse>(
      `/api/bilarasuttas/${uid}/${authorUid}`,
    );
  }
}
