"use client";

import { Background, Controls, ReactFlow } from "@xyflow/react";

export function DetectorFlow({ selectedMethod = "" }: { selectedMethod?: string }) {
  const isInternal = selectedMethod.includes("Internal") || selectedMethod.includes("SEP");
  const isTool = selectedMethod.includes("CRITIC");
  const nodes = [
    ["input", "Input Question / Answer", 0, 120, "#22d3ee"],
    ["method", "Method Selection", 220, 120, "#a78bfa"],
    ["claims", "Claim Extraction", 450, 40, "#34d399"],
    ["retrieval", isInternal ? "Internal Signal" : "Evidence Retrieval / Source Matching", 450, 200, isInternal ? "#38bdf8" : "#2dd4bf"],
    ["verify", isTool ? "Tool Check / Critique" : "Verification", 710, 120, isTool ? "#fb7185" : "#f59e0b"],
    ["score", "Risk Scoring", 960, 120, "#f97316"],
    ["explain", "Explanation + Evidence", 1190, 40, "#c084fc"],
    ["revise", "Revised Answer", 1190, 200, "#818cf8"]
  ].map(([id, label, x, y, color]) => ({
    id: String(id),
    position: { x: Number(x), y: Number(y) },
    data: { label },
    style: { borderRadius: 18, border: `1px solid ${color}`, background: "rgba(15,23,42,0.9)", color: "#f8fafc", width: 180, padding: 14, boxShadow: `0 0 26px ${color}33` }
  }));
  const edges = [
    ["e1", "input", "method"],
    ["e2", "method", "claims"],
    ["e3", "method", "retrieval"],
    ["e4", "claims", "verify"],
    ["e5", "retrieval", "verify"],
    ["e6", "verify", "score"],
    ["e7", "score", "explain"],
    ["e8", "score", "revise"]
  ].map(([id, source, target]) => ({ id: String(id), source: String(source), target: String(target), animated: true, style: { stroke: "#67e8f9", strokeWidth: 2 } }));
  return (
    <div className="glass-panel-ai ai-scan-bg h-[calc(100vh-16rem)] min-h-[560px] overflow-hidden rounded-3xl border border-white/10 bg-slate-950/35 shadow-2xl backdrop-blur-xl">
      <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
        <Background color="rgba(148,163,184,0.2)" gap={24} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
