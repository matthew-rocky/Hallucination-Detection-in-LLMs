"use client";

import { Clipboard, Download } from "lucide-react";
import type { DashboardState } from "@/lib/types";
import { EmptyState } from "./EmptyState";

function reportMarkdown(state: DashboardState) {
  const lines = [
    "# Hallucination Detection Report",
    "",
    `Mode: ${state.mode}`,
    `Selected methods: ${state.selectedMethods.join(", ")}`,
    `Question: ${state.question || "N/A"}`,
    "",
    "## Results",
    ...state.results.map((r) => `### ${r.method_name}\n- Risk: ${r.risk_label} (${r.risk_score ?? "N/A"})\n- Confidence: ${r.confidence == null ? "N/A" : Math.round(r.confidence * 100) + "%"}\n- Summary: ${r.summary || r.explanation}\n- Limitations: ${r.limitations || "N/A"}\n${r.revised_answer ? `- Revised answer: ${r.revised_answer}` : ""}`)
  ];
  return lines.join("\n");
}

export function ReportPage({ state }: { state: DashboardState }) {
  if (!state.results.length) return <EmptyState icon={Clipboard} title="No report available" message="Run an analysis first, then export the resulting method findings as Markdown or JSON." />;
  const md = reportMarkdown(state);
  const download = (name: string, text: string, type: string) => {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="space-y-5">
      <div className="glass rounded-3xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h1 className="text-2xl font-semibold text-white">Report / Export</h1><p className="mt-1 text-sm text-slate-400">Executive summary built from the latest real detector outputs.</p></div>
          <div className="flex flex-wrap gap-2"><button onClick={() => navigator.clipboard.writeText(md)} className="rounded-2xl bg-cyan-300 px-4 py-3 text-sm font-semibold text-slate-950">Copy report text</button><button onClick={() => download("hallucination-report.json", JSON.stringify(state.results, null, 2), "application/json")} className="rounded-2xl bg-white/10 px-4 py-3 text-sm font-semibold text-white">Download JSON</button><button onClick={() => download("hallucination-report.md", md, "text/markdown")} className="inline-flex items-center gap-2 rounded-2xl bg-fuchsia-300/15 px-4 py-3 text-sm font-semibold text-fuchsia-100"><Download size={16} /> Download Markdown</button></div>
        </div>
      </div>
      <article className="glass glass-panel-ai rounded-3xl p-6">
        <pre className="whitespace-pre-wrap text-sm leading-7 text-slate-300">{md}</pre>
      </article>
    </div>
  );
}
