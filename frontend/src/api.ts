import type { DataOverview, SearchPayload, SearchResponse, SimilarityResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:18000";

async function postSearch(path: string, payload: SearchPayload): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Search request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchOverview(): Promise<DataOverview> {
  const response = await fetch(`${API_BASE}/api/overview`);
  if (!response.ok) {
    throw new Error(`Overview request failed with status ${response.status}`);
  }
  return response.json();
}

export function searchText(payload: SearchPayload): Promise<SearchResponse> {
  return postSearch("/api/search/text", payload);
}

export function searchVector(payload: SearchPayload): Promise<SearchResponse> {
  return postSearch("/api/search/vector", payload);
}

export function searchHybrid(payload: SearchPayload): Promise<SearchResponse> {
  return postSearch("/api/search/hybrid", payload);
}

export function searchAdvancedRrf(payload: SearchPayload): Promise<SearchResponse> {
  return postSearch("/api/search/advanced/rrf", payload);
}

export async function compareSimilarity(sentenceA: string, sentenceB: string): Promise<SimilarityResponse> {
  const response = await fetch(`${API_BASE}/api/similarity`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence_a: sentenceA, sentence_b: sentenceB }),
  });
  if (!response.ok) {
    throw new Error(`Similarity request failed with status ${response.status}`);
  }
  return response.json();
}
