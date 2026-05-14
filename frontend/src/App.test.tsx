import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function searchResult(mode: "text" | "vector" | "hybrid" | "rrf", title: string) {
  return {
    mode,
    query: "criminal mastermind",
    results: [
      {
        id: `${mode}-1`,
        title,
        year: 2011,
        plot: "A clever criminal plan unfolds with dramatic twists and turns.",
        rating: 8.3,
        genres: ["Drama"],
        actors: ["Actor One"],
        score: 0.91,
        explanation: "match rationale",
      },
    ],
    timings: { search_ms: 10 },
  };
}

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: RequestInfo | URL) => {
        const u = String(url);
        if (u.includes("/api/overview")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_source: "data/movies.json",
                total_documents: 100,
                raw_shape_example: { title: "Raw", year: 2000, info: { plot: "plot" } },
                normalized_shape_example: { id: "abc", title: "Norm", year: 2000, plot: "plot" },
                index_name: "idx:movies",
                index_fields: [],
                redis_index_info: { num_docs: "100" },
              }),
              { status: 200 },
            ),
          );
        }
        if (u.includes("/api/search/text")) return Promise.resolve(new Response(JSON.stringify(searchResult("text", "Text Match")), { status: 200 }));
        if (u.includes("/api/search/vector")) return Promise.resolve(new Response(JSON.stringify(searchResult("vector", "Vector Match")), { status: 200 }));
        if (u.includes("/api/search/hybrid")) return Promise.resolve(new Response(JSON.stringify(searchResult("hybrid", "Hybrid Match")), { status: 200 }));
        if (u.includes("/api/search/advanced/rrf")) return Promise.resolve(new Response(JSON.stringify(searchResult("rrf", "RRF Match")), { status: 200 }));
        if (u.includes("/api/similarity")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                sentence_a: "a",
                sentence_b: "b",
                cosine_similarity: 0.88,
                interpretation: "Very similar meaning",
                embedding_preview: { sentence_a_first8: [0.1], sentence_b_first8: [0.2] },
              }),
              { status: 200 },
            ),
          );
        }
        return Promise.resolve(new Response("not found", { status: 404 }));
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("renders only the 4 core tabs at top level", async () => {
    render(<App />);
    expect(screen.getByRole("tab", { name: "Full-Text" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Semantic" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Hybrid" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Compare" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Data & Indexes" })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Text Match")).toBeInTheDocument());
  });

  it("keeps advanced controls hidden until More controls is expanded", async () => {
    render(<App />);
    expect(screen.getByLabelText("Query")).toBeInTheDocument();
    expect(screen.getByLabelText("Limit")).not.toBeVisible();
    fireEvent.click(screen.getByText("More controls"));
    await waitFor(() => expect(screen.getByLabelText("Limit")).toBeVisible());
    expect(screen.queryByLabelText("Min Rating")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Semantic" }));
    await waitFor(() => expect(screen.getByLabelText("Min Rating")).toBeVisible());
  });

  it("renders compact results and supports row expand", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("Text Match")).toBeInTheDocument());
    const expand = screen.getByRole("button", { name: "Details" });
    fireEvent.click(expand);
    expect(screen.getByRole("button", { name: "Hide" })).toBeInTheDocument();
  });

  it("renders compare table view", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Compare" }));
    await waitFor(() => expect(screen.getByRole("table")).toBeInTheDocument());
    const table = screen.getByRole("table");
    expect(within(table).getByText("Text")).toBeInTheDocument();
    expect(within(table).getByText("Vector")).toBeInTheDocument();
    expect(within(table).getByText("Hybrid")).toBeInTheDocument();
  });

  it("shows learn area with data/similarity/explainer", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Learn" }));
    expect(screen.getByRole("tab", { name: "Data & Indexes" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Similarity Lab" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Quick Explainer" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("data/movies.json")).toBeInTheDocument());
  });

  it("updates URL query params for deep-link state", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Hybrid" }));
    fireEvent.click(screen.getByRole("button", { name: "Learn" }));
    fireEvent.click(screen.getByRole("tab", { name: "Similarity Lab" }));
    await waitFor(() => {
      expect(window.location.search).toContain("tab=hybrid");
      expect(window.location.search).toContain("learn=1");
      expect(window.location.search).toContain("section=similarity");
    });
  });

  it("walks through guided story mode with fixed steps and code stops", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Story Mode" }));

    expect(screen.getByRole("heading", { name: "Step 1: Full-text baseline" })).toBeInTheDocument();
    expect(screen.getByDisplayValue("criminal mastermind")).toBeInTheDocument();
    expect(screen.getByText("Open next in code")).toBeInTheDocument();
    expect(screen.getByText(/build_text_query/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Next step" }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Step 2: Full-text failure" })).toBeInTheDocument();
      expect(screen.getByDisplayValue("a teacher inspires troubled students")).toBeInTheDocument();
      expect(window.location.search).toContain("story=1");
      expect(window.location.search).toContain("storyStep=fulltext-failure");
    });

    fireEvent.click(screen.getByRole("button", { name: "Next step" }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Step 3: Semantic rescue" })).toBeInTheDocument();
      expect(screen.getByText(/embed_query/)).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Semantic" })).toHaveAttribute("aria-selected", "true");
    });

    fireEvent.click(screen.getByRole("button", { name: "Next step" }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Step 4: Hybrid synthesis" })).toBeInTheDocument();
      expect(screen.getByDisplayValue("spy thriller with emotional depth")).toBeInTheDocument();
      expect(screen.getByText(/build_hybrid_query/)).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Next steps after this demo" })).toBeInTheDocument();
    });
  });
});
