import { useEffect, useMemo, useState } from "react";

import {
  compareSimilarity,
  fetchOverview,
  searchAdvancedRrf,
  searchHybrid,
  searchText,
  searchVector,
} from "./api";
import type { DataOverview, SearchPayload, SearchResponse, SimilarityResponse } from "./types";

const CORE_TABS = ["fulltext", "semantic", "hybrid", "compare"] as const;
const LEARN_SECTIONS = ["data", "similarity", "explainer"] as const;
const GENRE_OPTIONS = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Sci-Fi", "Thriller", "Animation"];

type CoreTab = (typeof CORE_TABS)[number];
type LearnSection = (typeof LEARN_SECTIONS)[number];
type CoreMode = "text" | "vector" | "hybrid";
type HybridStrategy = "weighted" | "rrf";

const TAB_LABELS: Record<CoreTab, string> = {
  fulltext: "Full-Text",
  semantic: "Semantic",
  hybrid: "Hybrid",
  compare: "Compare",
};

const PRESETS: Record<CoreTab, string[]> = {
  fulltext: ["criminal mastermind", "space mission", "family comedy"],
  semantic: ["movie about redemption after war", "a teacher inspires troubled students", "coming-of-age emotional journey"],
  hybrid: ["magic school adventure with danger", "spy thriller with emotional depth", "underdog sports drama"],
  compare: ["hero saves world from alien invasion", "heist with brilliant planner", "future dystopia with resistance"],
};

const LEARN_LABELS: Record<LearnSection, string> = {
  data: "Data & Indexes",
  similarity: "Similarity Lab",
  explainer: "Quick Explainer",
};

function isCoreTab(value: string | null): value is CoreTab {
  return value !== null && CORE_TABS.includes(value as CoreTab);
}

function isLearnSection(value: string | null): value is LearnSection {
  return value !== null && LEARN_SECTIONS.includes(value as LearnSection);
}

function formatTimings(timings: Record<string, number>): string {
  return Object.entries(timings)
    .map(([k, v]) => `${k}: ${v}ms`)
    .join(" | ");
}

function tokenizeQueryTerms(query: string): string[] {
  return Array.from(new Set(query.toLowerCase().split(/\s+/).map((term) => term.trim()).filter((term) => term.length >= 3)));
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function HighlightedText({ text, query }: { text: string; query: string }) {
  const terms = tokenizeQueryTerms(query);
  if (!terms.length) return <>{text}</>;
  const pattern = new RegExp(`(${terms.map(escapeRegex).join("|")})`, "gi");
  const pieces = text.split(pattern);
  return (
    <>
      {pieces.map((piece, idx) => {
        const isMatch = terms.some((t) => t.toLowerCase() === piece.toLowerCase());
        return isMatch ? (
          <mark key={`${piece}-${idx}`} className="hit">
            {piece}
          </mark>
        ) : (
          <span key={`${piece}-${idx}`}>{piece}</span>
        );
      })}
    </>
  );
}

function CompactResults({
  result,
  query,
  title,
}: {
  result: SearchResponse | null;
  query: string;
  title: string;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [showAll, setShowAll] = useState(false);
  const queryTerms = tokenizeQueryTerms(query);

  if (!result) return <p className="empty">Run a query to load top movie results.</p>;
  const visible = showAll ? result.results : result.results.slice(0, 5);

  return (
    <div className="result-block">
      <h3>{title}</h3>
      <p className="timings">{formatTimings(result.timings)}</p>
      <div className="result-list">
        {visible.map((row, idx) => {
          const rowKey = `${result.mode}-${row.id}`;
          const isOpen = expanded[rowKey] ?? false;
          const text = `${row.title} ${row.plot}`.toLowerCase();
          const matchedTerms = queryTerms.filter((term) => text.includes(term)).slice(0, 4);
          const matchLine = matchedTerms.length
            ? `Matched terms: ${matchedTerms.join(", ")}`
            : row.explanation ?? "Matched via semantic/hybrid relevance";
          return (
            <article key={rowKey} className="result-row">
              <div className="row-head">
                <p className="row-rank">{idx + 1}.</p>
                <div className="row-main">
                  <p className="row-title">{row.title}</p>
                  <p className="row-meta">
                    {row.year} {row.rating != null ? `| rating ${row.rating.toFixed(1)}` : ""} {row.score != null ? `| score ${row.score.toFixed(2)}` : ""}
                  </p>
                </div>
                <button
                  className="mini ghost"
                  onClick={() => setExpanded((prev) => ({ ...prev, [rowKey]: !isOpen }))}
                  aria-expanded={isOpen}
                >
                  {isOpen ? "Hide" : "Details"}
                </button>
              </div>
              <p className="row-snippet">
                <HighlightedText text={isOpen ? row.plot : `${row.plot.slice(0, 140)}${row.plot.length > 140 ? "..." : ""}`} query={query} />
              </p>
              <p className="row-rationale">{matchLine}</p>
              {isOpen && <p className="row-extra">{row.explanation ?? "Relevance match"} | Genres: {row.genres.join(", ")}</p>}
            </article>
          );
        })}
      </div>
      {result.results.length > 5 && (
        <button className="mini ghost" onClick={() => setShowAll((prev) => !prev)}>
          {showAll ? "Show top 5" : `Show all (${result.results.length})`}
        </button>
      )}
    </div>
  );
}

function CompareTable({ compareResults }: { compareResults: Partial<Record<CoreMode, SearchResponse>> }) {
  const modes: CoreMode[] = ["text", "vector", "hybrid"];
  const lengths = modes.map((m) => compareResults[m]?.results.length ?? 0);
  const rowCount = Math.min(5, Math.max(0, ...lengths));

  const timingSummary = modes
    .map((mode) => {
      const timings = compareResults[mode]?.timings;
      if (!timings) return `${mode}: no run`;
      return `${mode}: ${formatTimings(timings)}`;
    })
    .join(" | ");

  if (!rowCount) return <p className="empty">Run compare to populate side-by-side rankings.</p>;

  return (
    <div className="compare-table-wrap">
      <p className="timings">{timingSummary}</p>
      <table className="compare-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Text</th>
            <th>Vector</th>
            <th>Hybrid</th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }).map((_, idx) => {
            const t = compareResults.text?.results[idx];
            const v = compareResults.vector?.results[idx];
            const h = compareResults.hybrid?.results[idx];
            return (
              <tr key={`row-${idx}`}>
                <td>{idx + 1}</td>
                <td>{t ? `${t.title} (${t.score?.toFixed(3) ?? "-"})` : "-"}</td>
                <td>{v ? `${v.title} (${v.score?.toFixed(3) ?? "-"})` : "-"}</td>
                <td>{h ? `${h.title} (${h.score?.toFixed(3) ?? "-"})` : "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const [activeTab, setActiveTab] = useState<CoreTab>(isCoreTab(params.get("tab")) ? (params.get("tab") as CoreTab) : "fulltext");
  const [learnOpen, setLearnOpen] = useState(params.get("learn") === "1");
  const [learnSection, setLearnSection] = useState<LearnSection>(isLearnSection(params.get("section")) ? (params.get("section") as LearnSection) : "data");

  const [query, setQuery] = useState("criminal mastermind");
  const [limit, setLimit] = useState(5);
  const [minRating, setMinRating] = useState<number | undefined>(undefined);
  const [genres, setGenres] = useState<string[]>([]);
  const [alpha, setAlpha] = useState(0.7);
  const [rrfK, setRrfK] = useState(60);
  const [rrfTextWeight, setRrfTextWeight] = useState(0.5);
  const [hybridStrategy, setHybridStrategy] = useState<HybridStrategy>("weighted");

  const [overview, setOverview] = useState<DataOverview | null>(null);
  const [textResult, setTextResult] = useState<SearchResponse | null>(null);
  const [semanticResult, setSemanticResult] = useState<SearchResponse | null>(null);
  const [hybridResult, setHybridResult] = useState<SearchResponse | null>(null);
  const [compareResults, setCompareResults] = useState<Partial<Record<CoreMode, SearchResponse>>>({});
  const [similarityA, setSimilarityA] = useState("A lonely astronaut rediscovers purpose on a moon mission.");
  const [similarityB, setSimilarityB] = useState("A space explorer finds meaning during a lunar expedition.");
  const [similarityResult, setSimilarityResult] = useState<SimilarityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const payload = useMemo<SearchPayload>(
    () => ({
      query,
      limit,
      filters: { genres, min_rating: minRating },
      hybrid: { alpha },
      advanced: {
        rrf_k: rrfK,
        rrf_weights: [rrfTextWeight, 1 - rrfTextWeight],
      },
    }),
    [query, limit, genres, minRating, alpha, rrfK, rrfTextWeight],
  );

  useEffect(() => {
    const next = new URLSearchParams(window.location.search);
    next.set("tab", activeTab);
    next.set("learn", learnOpen ? "1" : "0");
    if (learnOpen) {
      next.set("section", learnSection);
    } else {
      next.delete("section");
    }
    const url = `${window.location.pathname}?${next.toString()}`;
    window.history.replaceState({}, "", url);
  }, [activeTab, learnOpen, learnSection]);

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      setBootstrapping(true);
      try {
        const initialPayload: SearchPayload = {
          query: "criminal mastermind",
          limit: 5,
          filters: { genres: [], min_rating: undefined },
          hybrid: { alpha: 0.7 },
          advanced: { rrf_k: 60, rrf_weights: [0.5, 0.5] },
        };
        const [overviewData, text, semantic, hybrid] = await Promise.all([
          fetchOverview(),
          searchText(initialPayload),
          searchVector(initialPayload),
          searchHybrid(initialPayload),
        ]);
        if (cancelled) return;
        setOverview(overviewData);
        setTextResult(text);
        setSemanticResult(semantic);
        setHybridResult(hybrid);
        setCompareResults({ text, vector: semantic, hybrid });
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to bootstrap demo data");
        }
      } finally {
        if (!cancelled) setBootstrapping(false);
      }
    }
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  async function searchByMode(mode: CoreMode, useRrfForHybrid = false): Promise<SearchResponse> {
    if (mode === "text") return searchText(payload);
    if (mode === "vector") return searchVector(payload);
    if (useRrfForHybrid) return searchAdvancedRrf(payload);
    return searchHybrid(payload);
  }

  async function runCurrentSearch() {
    setLoading(true);
    setError(null);
    try {
      if (activeTab === "fulltext") {
        const res = await searchByMode("text");
        setTextResult(res);
      } else if (activeTab === "semantic") {
        const res = await searchByMode("vector");
        setSemanticResult(res);
      } else if (activeTab === "hybrid") {
        const res = await searchByMode("hybrid", hybridStrategy === "rrf");
        setHybridResult(res);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function runCompare() {
    setLoading(true);
    setError(null);
    try {
      const [text, semantic, hybrid] = await Promise.all([searchByMode("text"), searchByMode("vector"), searchByMode("hybrid")]);
      setCompareResults({ text, vector: semantic, hybrid });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Compare run failed");
    } finally {
      setLoading(false);
    }
  }

  async function runSimilarity() {
    setLoading(true);
    setError(null);
    try {
      const res = await compareSimilarity(similarityA, similarityB);
      setSimilarityResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Similarity check failed");
    } finally {
      setLoading(false);
    }
  }

  function toggleGenre(genre: string) {
    setGenres((prev) => (prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]));
  }

  useEffect(() => {
    if (bootstrapping) return;
    const trimmed = query.trim();
    if (!trimmed) {
      setError(null);
      return;
    }

    const debounce = setTimeout(() => {
      if (activeTab === "compare") {
        void runCompare();
      } else {
        void runCurrentSearch();
      }
    }, 220);

    return () => clearTimeout(debounce);
  }, [activeTab, query, limit, minRating, genres, alpha, rrfK, rrfTextWeight, hybridStrategy, bootstrapping]);

  return (
    <div className="app-shell minimal">
      <section className="spotlight">
        <img
          className="redis-logo"
          src="https://commons.wikimedia.org/wiki/Special:FilePath/Redis_logo.svg"
          alt="Redis"
          onError={(event) => {
            const target = event.currentTarget;
            target.style.display = "none";
            const fallback = target.nextElementSibling as HTMLElement | null;
            if (fallback) fallback.style.display = "block";
          }}
        />
        <p className="redis-wordmark" aria-label="Redis">
          Redis
        </p>
        <div className="spotlight-search">
          <span className="search-icon" aria-hidden="true">
            🔎
          </span>
          <label htmlFor="query" className="sr-only">
            Query
          </label>
          <input id="query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search movies in real time..." />
          {loading && (
            <span className="live-status" aria-live="polite">
              <span className="dot-spinner" aria-hidden="true" />
              Searching
            </span>
          )}
        </div>
        <div className="top-nav compact">
          <nav className="demo-tabs" role="tablist" aria-label="Core search tabs">
            {CORE_TABS.map((tab) => (
              <button
                key={tab}
                role="tab"
                aria-selected={tab === activeTab}
                className={`tab-chip ${tab === activeTab ? "active" : ""}`}
                onClick={() => setActiveTab(tab)}
              >
                {TAB_LABELS[tab]}
              </button>
            ))}
          </nav>
          <button
            className={`secondary-btn ${learnOpen ? "active" : ""}`}
            onClick={() => setLearnOpen((prev) => !prev)}
            aria-expanded={learnOpen}
          >
            Learn
          </button>
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      <section className="panel stack">
        <h2>{TAB_LABELS[activeTab]}</h2>

        <details className="controls-accordion">
          <summary>More controls</summary>
          <div className="accordion-content">
            <div className="preset-row">
              {PRESETS[activeTab].map((preset) => (
                <button key={preset} className="preset" onClick={() => setQuery(preset)}>
                  {preset}
                </button>
              ))}
            </div>
            <div className={`controls-grid ${activeTab === "fulltext" ? "fulltext-grid" : ""}`}>
              <label htmlFor="limit">
                Limit
                <input id="limit" type="number" min={1} max={30} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
              </label>
              {(activeTab === "semantic" || activeTab === "hybrid" || activeTab === "compare") && (
                <label htmlFor="min-rating">
                  Min Rating
                  <input
                    id="min-rating"
                    type="number"
                    min={0}
                    max={10}
                    step={0.1}
                    value={minRating ?? ""}
                    onChange={(e) => setMinRating(e.target.value ? Number(e.target.value) : undefined)}
                  />
                </label>
              )}
              {(activeTab === "hybrid" || activeTab === "compare") && (
                <label htmlFor="alpha">
                  Hybrid Alpha ({alpha.toFixed(2)})
                  <input id="alpha" type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
                </label>
              )}
            </div>
            <div className="genres" role="group" aria-label="Genre filters">
              {GENRE_OPTIONS.map((genre) => (
                <button
                  key={genre}
                  aria-pressed={genres.includes(genre)}
                  className={`genre-pill ${genres.includes(genre) ? "on" : ""}`}
                  onClick={() => toggleGenre(genre)}
                >
                  {genre}
                </button>
              ))}
            </div>
            {activeTab === "hybrid" && (
              <div className="hybrid-controls">
                <div className="switch-row">
                  <button className={hybridStrategy === "weighted" ? "mini active" : "mini"} onClick={() => setHybridStrategy("weighted")}>
                    Weighted
                  </button>
                  <button className={hybridStrategy === "rrf" ? "mini active" : "mini"} onClick={() => setHybridStrategy("rrf")}>
                    RRF
                  </button>
                </div>
                {hybridStrategy === "rrf" && (
                  <>
                    <label htmlFor="rrf-k">
                      RRF k ({rrfK})
                      <input id="rrf-k" type="range" min={10} max={120} step={5} value={rrfK} onChange={(e) => setRrfK(Number(e.target.value))} />
                    </label>
                    <label htmlFor="rrf-weight">
                      Text Weight ({rrfTextWeight.toFixed(2)})
                      <input
                        id="rrf-weight"
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={rrfTextWeight}
                        onChange={(e) => setRrfTextWeight(Number(e.target.value))}
                      />
                    </label>
                  </>
                )}
              </div>
            )}
          </div>
        </details>

        {activeTab === "fulltext" && <CompactResults result={textResult} query={query} title="Top matches" />}
        {activeTab === "semantic" && <CompactResults result={semanticResult} query={query} title="Top matches" />}
        {activeTab === "hybrid" && <CompactResults result={hybridResult} query={query} title={hybridStrategy === "weighted" ? "Weighted blend results" : "RRF results"} />}
        {activeTab === "compare" && <CompareTable compareResults={compareResults} />}
      </section>

      {learnOpen && (
        <aside className="panel learn-panel muted-learn" aria-label="Learn area">
          <div className="learn-tabs" role="tablist" aria-label="Learn sections">
            {LEARN_SECTIONS.map((section) => (
              <button
                key={section}
                role="tab"
                aria-selected={learnSection === section}
                className={`mini ${learnSection === section ? "active" : ""}`}
                onClick={() => setLearnSection(section)}
              >
                {LEARN_LABELS[section]}
              </button>
            ))}
          </div>

          {learnSection === "data" && (
            <section className="stack">
              <h3>Data & Indexes</h3>
              {bootstrapping ? (
                <p className="empty">Loading overview...</p>
              ) : (
                <>
                  <div className="summary-grid">
                    <article className="metric">
                      <h4>Dataset</h4>
                      <p>{overview?.dataset_source}</p>
                    </article>
                    <article className="metric">
                      <h4>Docs</h4>
                      <p>{overview?.total_documents ?? 0}</p>
                    </article>
                    <article className="metric">
                      <h4>Index</h4>
                      <p>{overview?.index_name}</p>
                    </article>
                  </div>
                  <details>
                    <summary>Show details</summary>
                    <div className="data-panels">
                      <article>
                        <h4>Raw shape</h4>
                        <pre>
                          <code>{JSON.stringify(overview?.raw_shape_example ?? {}, null, 2)}</code>
                        </pre>
                      </article>
                      <article>
                        <h4>Normalized shape</h4>
                        <pre>
                          <code>{JSON.stringify(overview?.normalized_shape_example ?? {}, null, 2)}</code>
                        </pre>
                      </article>
                    </div>
                    <h4>FT.INFO stats</h4>
                    <pre>
                      <code>{JSON.stringify(overview?.redis_index_info ?? {}, null, 2)}</code>
                    </pre>
                  </details>
                </>
              )}
            </section>
          )}

          {learnSection === "similarity" && (
            <section className="stack">
              <h3>Similarity Lab</h3>
              <div className="two-col">
                <label htmlFor="sentence-a">
                  Sentence A
                  <textarea id="sentence-a" rows={4} value={similarityA} onChange={(e) => setSimilarityA(e.target.value)} />
                </label>
                <label htmlFor="sentence-b">
                  Sentence B
                  <textarea id="sentence-b" rows={4} value={similarityB} onChange={(e) => setSimilarityB(e.target.value)} />
                </label>
              </div>
              <button className="primary-btn fit" onClick={runSimilarity} disabled={loading}>
                {loading ? "Running..." : "Check Similarity"}
              </button>
              {similarityResult && (
                <article className="similarity-result">
                  <h4>Cosine Similarity: {similarityResult.cosine_similarity.toFixed(4)}</h4>
                  <p>{similarityResult.interpretation}</p>
                  <pre>
                    <code>{JSON.stringify(similarityResult.embedding_preview, null, 2)}</code>
                  </pre>
                </article>
              )}
            </section>
          )}

          {learnSection === "explainer" && (
            <section className="explain-grid">
              <article>
                <h4>Full-text vs normal text search</h4>
                <p>Full-text uses tokenization + inverted index + BM25-style ranking. Normal text matching is usually simpler exact/substring filtering.</p>
              </article>
              <article>
                <h4>Embedding models</h4>
                <p>Embeddings map text into vectors. Similar meaning ends up close in vector space, allowing paraphrase-friendly retrieval.</p>
              </article>
              <article>
                <h4>Hybrid search</h4>
                <p>Hybrid combines lexical and semantic signals by weighted score blending or rank-fusion (RRF) when score scales differ.</p>
              </article>
              <article>
                <h4>Recommended demo order</h4>
                <p>Data/shape in Learn panel first, then Full-Text, Semantic, Hybrid, and Compare.</p>
              </article>
            </section>
          )}
        </aside>
      )}
    </div>
  );
}
