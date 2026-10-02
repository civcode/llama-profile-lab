import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HeatmapTable, ParetoPlot } from "./components";
import type { MatrixProjection, ParetoResult } from "./types";

describe("analysis components", () => {
  it("renders sparse matrix coordinates without inventing missing cells", () => {
    const matrix: MatrixProjection = {
      experiment_id: "exp",
      x_path: "compute.batch_size",
      y_path: "compute.ubatch_size",
      metric: "throughput.median",
      facet_path: null,
      x_values: [2048, 4096],
      y_values: [512, 1024],
      facets: [
        {
          value: null,
          cells: [
            {
              x: 2048,
              y: 512,
              value: 123.45,
              candidate_id: "a",
              workload_case_id: "w",
              run_count: 1,
              sample_count: 3
            }
          ]
        }
      ]
    };
    render(<HeatmapTable matrix={matrix} />);
    expect(screen.getByText("123.45")).toBeInTheDocument();
    expect(screen.getAllByText("—")).toHaveLength(3);
  });

  it("renders the Pareto frontier as observed candidates", () => {
    const result: ParetoResult = {
      experiment_id: "exp",
      objectives: [
        {
          key: "PP8K",
          direction: "maximize",
          metric: "throughput.median",
          filters: []
        },
        {
          key: "TG4K",
          direction: "maximize",
          metric: "throughput.median",
          filters: []
        }
      ],
      evaluated_count: 2,
      frontier: [
        {
          candidate_id: "cand-a",
          candidate_ordinal: 0,
          values: { PP8K: 100, TG4K: 50 }
        },
        {
          candidate_id: "cand-b",
          candidate_ordinal: 1,
          values: { PP8K: 90, TG4K: 60 }
        }
      ],
      excluded: {}
    };
    render(<ParetoPlot result={result} />);
    expect(screen.getByLabelText("Pareto frontier scatter plot")).toBeInTheDocument();
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
  });
});
