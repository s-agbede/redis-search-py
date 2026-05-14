export type StoryStepId = "fulltext-baseline" | "fulltext-failure" | "semantic-rescue" | "hybrid-synthesis";

export type StoryTab = "fulltext" | "semantic" | "hybrid" | "compare";
export type StoryHybridStrategy = "weighted" | "rrf";

export type StoryCodeStop = {
  label: string;
  file: string;
  symbols: string[];
};

export type StoryStep = {
  id: StoryStepId;
  title: string;
  summary: string;
  tab: StoryTab;
  query: string;
  limit: number;
  genres: string[];
  minRating?: number;
  alpha?: number;
  hybridStrategy?: StoryHybridStrategy;
  whatToNotice: string;
  decisionTakeaway: string;
  schemaNote?: string;
  nextSteps?: string;
  codeStops: StoryCodeStop[];
};

export const STORY_STEPS: StoryStep[] = [
  {
    id: "fulltext-baseline",
    title: "Step 1: Full-text baseline",
    summary: "Start with the familiar win case: a keyword-rich query where lexical matching is fast, precise, and easy to explain.",
    tab: "fulltext",
    query: "criminal mastermind",
    limit: 5,
    genres: [],
    whatToNotice: "The top results should visibly share the important query terms, which makes the ranking easy to justify live.",
    decisionTakeaway: "Use full-text first when users already know the right words and you want the most explainable baseline.",
    schemaNote: "At this stage, you only need indexed text fields like title and plot.",
    codeStops: [
      {
        label: "Query builder",
        file: "backend/app/search/modes/full_text.py",
        symbols: ["build_text_query(...)"],
      },
      {
        label: "Service orchestration",
        file: "backend/app/search/redis_service.py",
        symbols: ["search_text(...)"],
      },
    ],
  },
  {
    id: "fulltext-failure",
    title: "Step 2: Full-text failure",
    summary: "Keep the query natural-language and paraphrase-heavy so the audience can feel where exact-word matching starts to bend.",
    tab: "compare",
    query: "a teacher inspires troubled students",
    limit: 5,
    genres: [],
    whatToNotice: "Look at the text column first. The lexical ranking is less intuitive once the user stops typing the exact words stored in plot text.",
    decisionTakeaway: "Full-text is not bad here; it is solving a different problem. This is the moment to introduce semantic retrieval, not to over-tune keywords.",
    schemaNote: "Nothing in the full-text schema understands meaning yet; it still only matches tokens.",
    codeStops: [
      {
        label: "Text retrieval path",
        file: "backend/app/search/modes/full_text.py",
        symbols: ["build_text_query(...)", "query_text_rows(...)"],
      },
      {
        label: "API handoff",
        file: "backend/app/main.py",
        symbols: ["POST /api/search/text"],
      },
    ],
  },
  {
    id: "semantic-rescue",
    title: "Step 3: Semantic rescue",
    summary: "Run the same paraphrase-heavy query through the vector path so the audience sees meaning-based recovery instead of new wording rules.",
    tab: "semantic",
    query: "a teacher inspires troubled students",
    limit: 5,
    genres: [],
    whatToNotice: "The same query now retrieves results that feel semantically aligned even when the visible words differ from the input.",
    decisionTakeaway: "Use semantic search when user language varies and you care more about intent than exact overlap.",
    schemaNote: "This is where the conceptual schema jump happens: the index still has text fields, but now it also stores plot_embedding vectors.",
    codeStops: [
      {
        label: "Embedding and vector query",
        file: "backend/app/search/modes/semantic.py",
        symbols: ["embed_query(...)", "build_vector_query(...)", "query_vector_rows(...)"],
      },
      {
        label: "Service orchestration",
        file: "backend/app/search/redis_service.py",
        symbols: ["_get_vectorizer(...)", "search_vector(...)"],
      },
    ],
  },
  {
    id: "hybrid-synthesis",
    title: "Step 4: Hybrid synthesis",
    summary: "Finish with a query that benefits from both lexical precision and semantic intent so the audience sees what they would likely ship.",
    tab: "hybrid",
    query: "spy thriller with emotional depth",
    limit: 5,
    genres: ["Thriller"],
    minRating: 6.5,
    alpha: 0.7,
    hybridStrategy: "weighted",
    whatToNotice: "Hybrid should reward the explicit thriller phrasing while still honoring the softer intent phrase about emotional depth.",
    decisionTakeaway: "Use hybrid when real users mix exact terms and fuzzy intent, and you want one production path that handles both gracefully.",
    schemaNote: "Hybrid works because the same index can consult both text fields and the embedding field in one teaching story.",
    nextSteps: "Next steps after this demo: mention RRF and reranking as follow-on quality work, not as required pieces of the main mental model.",
    codeStops: [
      {
        label: "Hybrid query builder",
        file: "backend/app/search/modes/hybrid.py",
        symbols: ["build_hybrid_query(...)", "query_hybrid_rows(...)"],
      },
      {
        label: "Service orchestration",
        file: "backend/app/search/redis_service.py",
        symbols: ["search_hybrid(...)"],
      },
    ],
  },
];
