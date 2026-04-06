import pLimit from "p-limit";
import pRetry from "p-retry";

const DEFAULT_CONCURRENCY = 3;
const DEFAULT_DELAY_MS = 200;

export function createLimiter(concurrency = DEFAULT_CONCURRENCY) {
  return pLimit(concurrency);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchWithRetry(
  url: string,
  options?: RequestInit,
  retries = 3,
): Promise<Response> {
  return pRetry(
    async () => {
      const response = await fetch(url, options);

      if (response.status >= 500) {
        throw new Error(`Server error ${response.status} for ${url}`);
      }

      return response;
    },
    {
      retries,
      minTimeout: 1000,
      maxTimeout: 10_000,
      onFailedAttempt: (error) => {
        console.warn(
          `  Retry ${error.attemptNumber}/${retries} for ${url}: ${error.error.message}`,
        );
      },
    },
  );
}

export async function throttledFetch(
  url: string,
  options?: RequestInit,
  delayMs = DEFAULT_DELAY_MS,
): Promise<Response> {
  const response = await fetchWithRetry(url, options);
  await sleep(delayMs);
  return response;
}
