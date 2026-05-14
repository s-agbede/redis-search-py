export type SearchMode = "text" | "vector" | "hybrid" | "rrf" | "rerank";

export type SearchResultItem = {
  id: string;
  title: string;
  year: number;
  plot: string;
  rating?: number | null;
  genres: string[];
  actors: string[];
  release_date?: string | null;
  rank?: number | null;
  image_url?: string | null;
  running_time_secs?: number | null;
  score?: number | null;
  explanation?: string | null;
};

export type SearchResponse = {
  mode: SearchMode;
  query: string;
  results: SearchResultItem[];
  timings: Record<string, number>;
  lesson_takeaway?: string | null;
};

export type SearchPayload = {
  query: string;
  limit: number;
  filters: {
    genres: string[];
    min_rating?: number;
  };
  hybrid?: {
    alpha: number;
  };
  advanced?: {
    rrf_k?: number;
    rrf_weights?: [number, number];
    rerank_top_n?: number;
  };
};

export type IndexFieldDescriptor = {
  name: string;
  type: string;
  attrs: Record<string, unknown>;
};

export type DataOverview = {
  dataset_source: string;
  total_documents: number;
  raw_shape_example: Record<string, unknown>;
  normalized_shape_example: Record<string, unknown>;
  index_name: string;
  index_fields: IndexFieldDescriptor[];
  redis_index_info: Record<string, unknown>;
};

export type SimilarityResponse = {
  sentence_a: string;
  sentence_b: string;
  cosine_similarity: number;
  interpretation: string;
  embedding_preview: Record<string, number[]>;
};
