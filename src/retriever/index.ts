export interface RetrievedSutta {
  uid: string;
  title: string;
  collection: string;
  summary: string;
  text: string;
  source: string;
}

export interface SuttaRetriever {
  retrieve(userQuery: string): Promise<RetrievedSutta>;
}

class PlaceholderSuttaRetriever implements SuttaRetriever {
  async retrieve(userQuery: string): Promise<RetrievedSutta> {
    const normalizedQuery = userQuery.trim();

    if (!normalizedQuery) {
      throw new Error("Retriever received an empty user_query.");
    }

    return {
      uid: "placeholder",
      title: "Retriever not implemented yet",
      collection: "Pali Canon",
      summary:
        "This is a placeholder response. Dima should replace PlaceholderSuttaRetriever with real retrieval logic.",
      text: `Retriever placeholder for query: "${normalizedQuery}"`,
      source: "local-placeholder",
    };
  }
}

export function createSuttaRetriever(): SuttaRetriever {
  return new PlaceholderSuttaRetriever();
}
