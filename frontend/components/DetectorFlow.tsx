"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type OnNodesChange
} from "@xyflow/react";
import { motion } from "framer-motion";
import {
  Bot,
  ClipboardCheck,
  Cpu,
  Database,
  FileText,
  Gauge,
  GitBranch,
  Layers3,
  ListChecks,
  Network,
  RefreshCcw,
  Route,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Wrench
} from "lucide-react";

type IconKey =
  | "input"
  | "cpu"
  | "database"
  | "gauge"
  | "branch"
  | "search"
  | "shield"
  | "alert"
  | "list"
  | "network"
  | "route"
  | "tools"
  | "refresh"
  | "spark"
  | "check"
  | "layers";

type FlowNodeData = {
  label: string;
  subtitle?: string;
  description: string;
  icon: IconKey;
  color: string;
};

type FlowNode = Node<FlowNodeData, "methodNode">;

type FlowDefinition = {
  id: string;
  methodName: string;
  title: string;
  subtitle: string;
  family: string;
  explanation: string;
  requiredInputs: string[];
  output: string;
  limitation: string;
  palette: {
    color: string;
    edge: string;
    soft: string;
    ring: string;
  };
  nodes: Array<{
    id: string;
    label: string;
    subtitle?: string;
    description: string;
    icon: IconKey;
    x: number;
    y: number;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    optional?: boolean;
    label?: string;
  }>;
};

const ICONS = {
  input: FileText,
  cpu: Cpu,
  database: Database,
  gauge: Gauge,
  branch: GitBranch,
  search: Search,
  shield: ShieldCheck,
  alert: ShieldAlert,
  list: ListChecks,
  network: Network,
  route: Route,
  tools: Wrench,
  refresh: RefreshCcw,
  spark: Sparkles,
  check: ClipboardCheck,
  layers: Layers3
};

const METHOD_FLOWS: Record<string, FlowDefinition> = {
  "internal-signal-baseline": {
    id: "internal-signal-baseline",
    methodName: "Internal-Signal Baseline",
    title: "Internal-Signal Baseline",
    subtitle: "Uncertainty-only hallucination risk estimate",
    family: "Internal-signal methods",
    explanation: "This method estimates hallucination risk from local model uncertainty. It does not retrieve external evidence, so a low score does not prove factual correctness.",
    requiredInputs: ["question", "answer"],
    output: "A hallucination risk score with uncertainty-oriented explanation.",
    limitation: "No source, retrieval, or citation evidence is checked.",
    palette: { color: "#38bdf8", edge: "#67e8f9", soft: "rgba(56,189,248,0.12)", ring: "rgba(56,189,248,0.36)" },
    nodes: [
      { id: "input", label: "Input Answer", subtitle: "Question + answer", description: "Receives the prompt and single answer being checked.", icon: "input", x: 0, y: 260 },
      { id: "tokenize", label: "Local LM Tokenization", subtitle: "No external evidence", description: "Tokenizes the answer for local language-model scoring.", icon: "cpu", x: 260, y: 70 },
      { id: "scoring", label: "Teacher-Forced Scoring", subtitle: "Answer likelihood", description: "Scores answer tokens under a local model using teacher-forced evaluation.", icon: "gauge", x: 620, y: 260 },
      { id: "features", label: "Confidence Features", subtitle: "NLL / confidence", description: "Summarizes token confidence, negative log likelihood, and related uncertainty signals.", icon: "spark", x: 760, y: 70 },
      { id: "calibration", label: "Risk Calibration", subtitle: "Heuristics", description: "Maps uncertainty features into calibrated hallucination-risk heuristics.", icon: "shield", x: 1120, y: 260 },
      { id: "risk", label: "Hallucination Risk", subtitle: "Score + explanation", description: "Returns a risk label, score, and uncertainty-based explanation.", icon: "alert", x: 1380, y: 80 }
    ],
    edges: [
      { id: "e1", source: "input", target: "tokenize" },
      { id: "e2", source: "tokenize", target: "scoring" },
      { id: "e3", source: "scoring", target: "features" },
      { id: "e4", source: "features", target: "calibration" },
      { id: "e5", source: "calibration", target: "risk" }
    ]
  },
  "sep-inspired-internal-signal": {
    id: "sep-inspired-internal-signal",
    methodName: "SEP-Inspired Internal Signal",
    title: "SEP-Inspired Internal Signal",
    subtitle: "Sample stability plus SEP-lite uncertainty features",
    family: "Internal-signal / SEP-inspired",
    explanation: "This method uses sampled-answer behavior and internal uncertainty-style features to estimate hallucination risk. It focuses on stability and semantic drift rather than external evidence.",
    requiredInputs: ["question", "answer", "sampled_answers"],
    output: "SEP-lite risk score with stability and drift explanation.",
    limitation: "The local implementation is SEP-inspired, not a full reproduction of paper-scale SEP systems.",
    palette: { color: "#c084fc", edge: "#d8b4fe", soft: "rgba(192,132,252,0.12)", ring: "rgba(192,132,252,0.36)" },
    nodes: [
      { id: "input", label: "Input Answer", subtitle: "Question + answer", description: "Starts from the answer under review.", icon: "input", x: 20, y: 160 },
      { id: "samples", label: "Sampled Answers", subtitle: "Alternative completions", description: "Uses provided or generated sampled answers as behavioral evidence.", icon: "layers", x: 250, y: 40 },
      { id: "stability", label: "Stability Check", subtitle: "Agreement / drift", description: "Compares sampled answers for agreement, contradictions, suspicious consensus, and semantic drift.", icon: "branch", x: 510, y: 40 },
      { id: "features", label: "Feature Extraction", subtitle: "Hidden-state style summary", description: "Builds internal-style feature summaries and lightweight SEP proxy features.", icon: "cpu", x: 510, y: 245 },
      { id: "sep", label: "SEP-Lite Scoring", subtitle: "Combined signal", description: "Blends sample stability and feature signals into a SEP-lite hallucination-risk score.", icon: "gauge", x: 790, y: 145 },
      { id: "risk", label: "Risk Explanation", subtitle: "Score + reasons", description: "Explains the risk in terms of sample behavior, uncertainty features, and detected drift.", icon: "spark", x: 1060, y: 145 }
    ],
    edges: [
      { id: "e1", source: "input", target: "samples" },
      { id: "e2", source: "samples", target: "stability" },
      { id: "e3", source: "input", target: "features" },
      { id: "e4", source: "stability", target: "sep" },
      { id: "e5", source: "features", target: "sep" },
      { id: "e6", source: "sep", target: "risk" }
    ]
  },
  "source-grounded-consistency": {
    id: "source-grounded-consistency",
    methodName: "Source-Grounded Consistency",
    title: "Source-Grounded Consistency",
    subtitle: "Claim support against one trusted source passage",
    family: "Source-grounded consistency",
    explanation: "This method checks whether answer claims are supported by a provided source passage. It is strongest when a trusted source text is available.",
    requiredInputs: ["answer", "source_text"],
    output: "Claim-level support findings and a grounded risk score.",
    limitation: "It can only judge the answer against the supplied source passage.",
    palette: { color: "#34d399", edge: "#6ee7b7", soft: "rgba(52,211,153,0.12)", ring: "rgba(52,211,153,0.36)" },
    nodes: [
      { id: "answer", label: "Answer + Source", subtitle: "Two input streams", description: "Receives the model answer and the source passage it should follow.", icon: "input", x: 20, y: 150 },
      { id: "claims", label: "Claim Extraction", subtitle: "Answer branch", description: "Breaks the answer into checkable factual claims.", icon: "list", x: 260, y: 55 },
      { id: "source", label: "Source Chunking", subtitle: "Source branch", description: "Splits the provided source text into searchable chunks.", icon: "database", x: 260, y: 250 },
      { id: "match", label: "Semantic Matching", subtitle: "Claims to source", description: "Matches extracted claims to the most relevant source chunks.", icon: "search", x: 540, y: 150 },
      { id: "classify", label: "Support Classification", subtitle: "Supported / contradicted", description: "Classifies each claim as supported, contradicted, or insufficiently grounded.", icon: "shield", x: 820, y: 150 },
      { id: "findings", label: "Claim Findings", subtitle: "Per-claim evidence", description: "Packages claim-level evidence, verdicts, and explanations.", icon: "check", x: 1100, y: 55 },
      { id: "risk", label: "Grounded Risk", subtitle: "Final score", description: "Aggregates source-grounding findings into the final risk score.", icon: "gauge", x: 1100, y: 250 }
    ],
    edges: [
      { id: "e1", source: "answer", target: "claims" },
      { id: "e2", source: "answer", target: "source" },
      { id: "e3", source: "claims", target: "match" },
      { id: "e4", source: "source", target: "match" },
      { id: "e5", source: "match", target: "classify" },
      { id: "e6", source: "classify", target: "findings" },
      { id: "e7", source: "classify", target: "risk" }
    ]
  },
  "retrieval-grounded-checker": {
    id: "retrieval-grounded-checker",
    methodName: "Retrieval-Grounded Checker",
    title: "Retrieval-Grounded Checker",
    subtitle: "Local evidence retrieval and claim grounding",
    family: "Retrieval-grounded checking",
    explanation: "This method checks answer claims against a retrieved local evidence pool. It uses document chunking, local indexing, retrieval, and evidence ranking before assigning grounding labels.",
    requiredInputs: ["answer", "evidence_text"],
    output: "Grounded claim labels, citations, and a risk score.",
    limitation: "Retrieval quality depends on the evidence supplied to the local index.",
    palette: { color: "#2dd4bf", edge: "#5eead4", soft: "rgba(45,212,191,0.12)", ring: "rgba(45,212,191,0.36)" },
    nodes: [
      { id: "input", label: "Answer + Evidence", subtitle: "Documents or snippets", description: "Receives the answer and local evidence documents or text.", icon: "input", x: 20, y: 150 },
      { id: "claims", label: "Claim Extraction", subtitle: "Answer branch", description: "Extracts factual claims that need grounding.", icon: "list", x: 260, y: 40 },
      { id: "index", label: "Build Local Index", subtitle: "Document branch", description: "Chunks evidence documents and builds a local retrieval index.", icon: "database", x: 260, y: 260 },
      { id: "retrieve", label: "Retrieve Evidence", subtitle: "Candidate passages", description: "Retrieves passages likely to support or contradict each claim.", icon: "search", x: 540, y: 255 },
      { id: "rank", label: "Rank Matches", subtitle: "Evidence quality", description: "Ranks retrieved evidence by relevance and grounding strength.", icon: "gauge", x: 790, y: 150 },
      { id: "ground", label: "Ground Claims", subtitle: "Claim verdicts", description: "Assigns each claim a support, contradiction, or insufficient-evidence status.", icon: "shield", x: 1040, y: 150 },
      { id: "risk", label: "Citations + Risk", subtitle: "Report output", description: "Assembles citations and aggregates findings into risk.", icon: "check", x: 1290, y: 150 }
    ],
    edges: [
      { id: "e1", source: "input", target: "claims" },
      { id: "e2", source: "input", target: "index" },
      { id: "e3", source: "index", target: "retrieve" },
      { id: "e4", source: "claims", target: "rank" },
      { id: "e5", source: "retrieve", target: "rank" },
      { id: "e6", source: "rank", target: "ground" },
      { id: "e7", source: "ground", target: "risk" }
    ]
  },
  "rag-grounded-check": {
    id: "rag-grounded-check",
    methodName: "RAG Grounded Check",
    title: "RAG Grounded Check",
    subtitle: "Post-hoc grounding over retrieved RAG context",
    family: "RAG-style grounded checking",
    explanation: "This is a RAG-style grounded check, not a full RAG generator. It evaluates whether an existing answer is supported by retrieved context.",
    requiredInputs: ["answer", "evidence_text"],
    output: "RAG grounding score with missing-support and contradiction cues.",
    limitation: "It audits grounding against retrieved context but does not regenerate the answer.",
    palette: { color: "#fbbf24", edge: "#fcd34d", soft: "rgba(251,191,36,0.12)", ring: "rgba(251,191,36,0.36)" },
    nodes: [
      { id: "context", label: "RAG Context", subtitle: "Retrieved passages", description: "Starts from the retrieved context available to the answer.", icon: "database", x: 20, y: 245 },
      { id: "parse", label: "Parse Context", subtitle: "Chunk context", description: "Parses retrieved context into usable evidence units.", icon: "layers", x: 250, y: 245 },
      { id: "answer", label: "Input Answer", subtitle: "Generated response", description: "Receives the answer to audit against RAG context.", icon: "input", x: 20, y: 40 },
      { id: "claims", label: "Extract Claims", subtitle: "Answer claims", description: "Extracts factual claims from the answer.", icon: "list", x: 250, y: 40 },
      { id: "align", label: "Alignment Check", subtitle: "Claims vs context", description: "Checks whether answer claims align with retrieved context.", icon: "search", x: 530, y: 145 },
      { id: "missing", label: "Missing Support", subtitle: "Unsupported claims", description: "Detects claims that are not backed by the provided context.", icon: "alert", x: 800, y: 45 },
      { id: "score", label: "Grounding Score", subtitle: "RAG audit score", description: "Combines missing support and contradiction cues into a grounding score.", icon: "gauge", x: 800, y: 250 },
      { id: "explain", label: "Explanation", subtitle: "Grounding rationale", description: "Explains which parts were grounded, missing, or contradicted.", icon: "spark", x: 1080, y: 145 }
    ],
    edges: [
      { id: "e1", source: "context", target: "parse" },
      { id: "e2", source: "answer", target: "claims" },
      { id: "e3", source: "parse", target: "align" },
      { id: "e4", source: "claims", target: "align" },
      { id: "e5", source: "align", target: "missing" },
      { id: "e6", source: "align", target: "score" },
      { id: "e7", source: "missing", target: "explain" },
      { id: "e8", source: "score", target: "explain" }
    ]
  },
  "verification-based-workflow": {
    id: "verification-based-workflow",
    methodName: "Verification-Based Workflow",
    title: "Verification-Based Workflow",
    subtitle: "Staged claim verification and verdict aggregation",
    family: "Verification workflow baseline",
    explanation: "This method uses a staged verification process: it extracts claims, builds verification questions, checks evidence, and aggregates claim verdicts into risk.",
    requiredInputs: ["answer", "evidence_text"],
    output: "Claim verdicts and an aggregated verification risk score.",
    limitation: "It is a lightweight workflow approximation, not a full external research-agent verifier.",
    palette: { color: "#fb923c", edge: "#fdba74", soft: "rgba(251,146,60,0.12)", ring: "rgba(251,146,60,0.36)" },
    nodes: [
      { id: "input", label: "Input Answer", subtitle: "Question + answer", description: "Receives the answer to verify.", icon: "input", x: 20, y: 155 },
      { id: "claims", label: "Extract Claims", subtitle: "Checkable units", description: "Breaks the answer into independent factual claims.", icon: "list", x: 250, y: 155 },
      { id: "questions", label: "Build Verification Questions", subtitle: "Question set", description: "Creates targeted questions for checking each extracted claim.", icon: "branch", x: 530, y: 155 },
      { id: "q1", label: "Question Path A", subtitle: "Verification branch", description: "One verification path checks a subset of claims.", icon: "route", x: 810, y: 35 },
      { id: "q2", label: "Question Path B", subtitle: "Verification branch", description: "A second verification path checks another subset or evidence angle.", icon: "route", x: 810, y: 275 },
      { id: "evidence", label: "Check Evidence", subtitle: "Source lookup", description: "Checks available source or evidence text for verification answers.", icon: "search", x: 1080, y: 155 },
      { id: "verdicts", label: "Verify Claims", subtitle: "Verdicts", description: "Assigns claim verdicts based on verification answers.", icon: "check", x: 1340, y: 155 },
      { id: "risk", label: "Risk Score", subtitle: "Aggregate", description: "Aggregates claim verdicts into the final hallucination-risk score.", icon: "gauge", x: 1600, y: 155 }
    ],
    edges: [
      { id: "e1", source: "input", target: "claims" },
      { id: "e2", source: "claims", target: "questions" },
      { id: "e3", source: "questions", target: "q1" },
      { id: "e4", source: "questions", target: "q2" },
      { id: "e5", source: "q1", target: "evidence" },
      { id: "e6", source: "q2", target: "evidence" },
      { id: "e7", source: "evidence", target: "verdicts" },
      { id: "e8", source: "verdicts", target: "risk" }
    ]
  },
  "cove-style-verification": {
    id: "cove-style-verification",
    methodName: "CoVe-Style Verification",
    title: "CoVe-Style Verification",
    subtitle: "Plan, independently verify, compare, and revise",
    family: "Chain-of-Verification",
    explanation: "This method uses a Chain-of-Verification style decomposition and can revise the original answer when verification findings disagree.",
    requiredInputs: ["answer", "evidence_text"],
    output: "Final risk plus optional revised answer.",
    limitation: "This local prototype uses retrieval and extractive synthesis rather than multiple large-model verification calls.",
    palette: { color: "#818cf8", edge: "#a5b4fc", soft: "rgba(129,140,248,0.12)", ring: "rgba(129,140,248,0.36)" },
    nodes: [
      { id: "initial", label: "Initial Answer", subtitle: "Draft response", description: "Starts with the original answer that may need verification.", icon: "input", x: 20, y: 160 },
      { id: "plan", label: "Verification Plan", subtitle: "Question plan", description: "Plans verification questions from the original answer.", icon: "branch", x: 280, y: 160 },
      { id: "checks", label: "Independent Checks", subtitle: "Verify away from draft", description: "Answers verification questions independently using available evidence.", icon: "search", x: 560, y: 45 },
      { id: "compare", label: "Compare Findings", subtitle: "Original vs verified", description: "Compares original claims against independently verified answers.", icon: "shield", x: 850, y: 160 },
      { id: "issues", label: "Flag Issues", subtitle: "Unsupported claims", description: "Flags unsupported, contradicted, or unstable claims.", icon: "alert", x: 1130, y: 45 },
      { id: "revise", label: "Revise Answer", subtitle: "Rewrite if needed", description: "Revises the answer when verification identifies unsupported or contradicted claims.", icon: "refresh", x: 1130, y: 275 },
      { id: "final", label: "Final Risk", subtitle: "Risk + revised output", description: "Returns final risk and the revised output when available.", icon: "gauge", x: 1410, y: 160 }
    ],
    edges: [
      { id: "e1", source: "initial", target: "plan" },
      { id: "e2", source: "plan", target: "checks" },
      { id: "e3", source: "checks", target: "compare" },
      { id: "e4", source: "compare", target: "issues" },
      { id: "e5", source: "issues", target: "revise" },
      { id: "e6", source: "revise", target: "final" },
      { id: "e7", source: "compare", target: "final", optional: true, label: "no revision" },
      { id: "e8", source: "revise", target: "compare", optional: true, label: "review loop" }
    ]
  },
  "critic-lite-tool-check": {
    id: "critic-lite-tool-check",
    methodName: "CRITIC-lite Tool Check",
    title: "CRITIC-lite Tool Check",
    subtitle: "Tool-routed critique for checkable claims",
    family: "Tool-augmented critique and revision",
    explanation: "This method uses tool-assisted critique for numeric, grounded, or otherwise checkable claims. It routes claims through local tools, critiques the original answer, and can revise it.",
    requiredInputs: ["answer", "evidence_text"],
    output: "Tool-backed critique, optional revision, and final risk.",
    limitation: "The prototype uses local retrieval and numeric comparison tools, not broad external web tools.",
    palette: { color: "#fb7185", edge: "#fda4af", soft: "rgba(251,113,133,0.12)", ring: "rgba(251,113,133,0.36)" },
    nodes: [
      { id: "input", label: "Input Answer", subtitle: "Question + answer", description: "Receives the answer draft to critique.", icon: "input", x: 20, y: 160 },
      { id: "claims", label: "Extract Claims", subtitle: "Checkable claims", description: "Extracts numeric, date, entity, and grounded factual claims.", icon: "list", x: 260, y: 160 },
      { id: "router", label: "Route to Tools", subtitle: "Tool router", description: "Routes each claim to the most relevant local checking tool.", icon: "route", x: 530, y: 160 },
      { id: "numeric", label: "Numeric Checks", subtitle: "Numbers / dates", description: "Checks number-bearing and date-bearing claims against evidence.", icon: "tools", x: 810, y: 20 },
      { id: "entity", label: "Entity Checks", subtitle: "Names / facts", description: "Checks entity and grounded claims against local evidence.", icon: "search", x: 810, y: 160 },
      { id: "evidence", label: "Tool Evidence", subtitle: "Returned evidence", description: "Collects tool outputs and evidence snippets for critique.", icon: "database", x: 1080, y: 160 },
      { id: "critique", label: "Critique", subtitle: "Original answer review", description: "Critiques the original answer using tool evidence.", icon: "alert", x: 1350, y: 160 },
      { id: "revise", label: "Revise if Needed", subtitle: "Optional revision", description: "Revises the answer when tool evidence reveals issues.", icon: "refresh", x: 1620, y: 50 },
      { id: "risk", label: "Final Risk", subtitle: "Risk + critique", description: "Returns the final risk, critique, evidence, and revised answer when available.", icon: "gauge", x: 1620, y: 275 }
    ],
    edges: [
      { id: "e1", source: "input", target: "claims" },
      { id: "e2", source: "claims", target: "router" },
      { id: "e3", source: "router", target: "numeric" },
      { id: "e4", source: "router", target: "entity" },
      { id: "e5", source: "numeric", target: "evidence" },
      { id: "e6", source: "entity", target: "evidence" },
      { id: "e7", source: "evidence", target: "critique" },
      { id: "e8", source: "critique", target: "revise", optional: true },
      { id: "e9", source: "critique", target: "risk" },
      { id: "e10", source: "revise", target: "risk" }
    ]
  }
};

function methodIdFromName(name: string) {
  return name
    .toLowerCase()
    .replaceAll(" / ", "-")
    .replaceAll(" ", "-")
    .replaceAll("/", "-")
    .replaceAll("--", "-");
}

function resolveFlow(selectedMethod: string) {
  const id = methodIdFromName(selectedMethod);
  return METHOD_FLOWS[id] ?? Object.values(METHOD_FLOWS).find((flow) => flow.methodName === selectedMethod) ?? METHOD_FLOWS["internal-signal-baseline"];
}

function createNodes(flow: FlowDefinition, positions?: Record<string, { x: number; y: number }>): FlowNode[] {
  return flow.nodes.map((node) => ({
    id: node.id,
    type: "methodNode",
    position: positions?.[node.id] ?? defaultPosition(flow, { x: node.x, y: node.y }),
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    data: {
      label: node.label,
      subtitle: node.subtitle,
      description: node.description,
      icon: node.icon,
      color: flow.palette.color
    }
  }));
}

function getEdgeHandles(flow: FlowDefinition, sourceId: string, targetId: string) {
  const source = flow.nodes.find((node) => node.id === sourceId);
  const target = flow.nodes.find((node) => node.id === targetId);
  if (!source || !target) return { sourceHandle: "source-right", targetHandle: "target-left" };
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0
      ? { sourceHandle: "source-right", targetHandle: "target-left" }
      : { sourceHandle: "source-left", targetHandle: "target-right" };
  }
  return dy >= 0
    ? { sourceHandle: "source-bottom", targetHandle: "target-top" }
    : { sourceHandle: "source-top", targetHandle: "target-bottom" };
}

function createEdges(flow: FlowDefinition): Edge[] {
  const nodeIds = new Set(flow.nodes.map((node) => node.id));
  return flow.edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .map((edge) => {
      const handles = getEdgeHandles(flow, edge.source, edge.target);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: handles.sourceHandle,
        targetHandle: handles.targetHandle,
        type: "smoothstep",
        animated: true,
        label: edge.label,
        markerEnd: { type: MarkerType.ArrowClosed, color: flow.palette.edge },
        style: {
          stroke: flow.palette.edge,
          strokeWidth: edge.optional ? 1.9 : 2.7,
          strokeDasharray: edge.optional ? "5 9" : "9 7",
          filter: `drop-shadow(0 0 7px ${flow.palette.edge})`
        },
        labelStyle: { fill: "#cbd5e1", fontSize: 11 },
        labelBgStyle: { fill: "rgba(15,23,42,0.86)" }
      };
    });
}

function storageKey(methodId: string) {
  return `hds-method-flow-layout:v3:${methodId}`;
}

function readSavedPositions(methodId: string) {
  if (typeof window === "undefined") return undefined;
  try {
    const raw = window.localStorage.getItem(storageKey(methodId));
    return raw ? JSON.parse(raw) as Record<string, { x: number; y: number }> : undefined;
  } catch {
    return undefined;
  }
}

function extractPositions(nodes: FlowNode[]) {
  return Object.fromEntries(nodes.map((node) => [node.id, node.position]));
}

type RevealEvent = { type: "node" | "edge"; id: string };

function buildRevealEvents(flow: FlowDefinition): RevealEvent[] {
  const orderedNodes = [...flow.nodes].sort((a, b) => a.x - b.x || a.y - b.y);
  const seenNodes = new Set<string>();
  const emittedEdges = new Set<string>();
  const events: RevealEvent[] = [];

  orderedNodes.forEach((node) => {
    events.push({ type: "node", id: node.id });
    seenNodes.add(node.id);
    flow.edges.forEach((edge) => {
      const connectsToVisibleNode =
        (edge.target === node.id && seenNodes.has(edge.source)) ||
        (edge.source === node.id && seenNodes.has(edge.target));
      if (connectsToVisibleNode && !emittedEdges.has(edge.id)) {
        events.push({ type: "edge", id: edge.id });
        emittedEdges.add(edge.id);
      }
    });
  });

  flow.edges.forEach((edge) => {
    if (!emittedEdges.has(edge.id)) events.push({ type: "edge", id: edge.id });
  });

  return events;
}

function visibleIds(events: RevealEvent[], step: number) {
  const activeEvents = events.slice(0, Math.max(0, step));
  return {
    nodes: new Set(activeEvents.filter((event) => event.type === "node").map((event) => event.id)),
    edges: new Set(activeEvents.filter((event) => event.type === "edge").map((event) => event.id))
  };
}

function MethodNode({ data, selected }: NodeProps<FlowNode>) {
  const Icon = ICONS[data.icon];
  const handleStyle = {
    width: 10,
    height: 10,
    border: `1px solid ${data.color}`,
    background: "#020617",
    boxShadow: `0 0 12px ${data.color}88`
  };
  return (
    <motion.div
      initial={{ opacity: 0, x: -18, scale: 0.96 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.34, ease: "easeOut" }}
      className={`group min-h-[92px] w-[252px] rounded-3xl border bg-slate-950/84 p-5 text-left shadow-2xl backdrop-blur-xl transition hover:-translate-y-0.5 ${selected ? "ring-2" : ""}`}
      style={{
        borderColor: data.color,
        boxShadow: selected ? `0 0 42px ${data.color}66` : `0 0 30px ${data.color}2f`,
        outlineColor: data.color
      }}
    >
      <Handle id="target-left" type="target" position={Position.Left} style={handleStyle} />
      <Handle id="source-left" type="source" position={Position.Left} style={{ ...handleStyle, opacity: 0 }} />
      <Handle id="target-right" type="target" position={Position.Right} style={{ ...handleStyle, opacity: 0 }} />
      <Handle id="source-right" type="source" position={Position.Right} style={handleStyle} />
      <Handle id="target-top" type="target" position={Position.Top} style={handleStyle} />
      <Handle id="source-top" type="source" position={Position.Top} style={{ ...handleStyle, opacity: 0 }} />
      <Handle id="target-bottom" type="target" position={Position.Bottom} style={{ ...handleStyle, opacity: 0 }} />
      <Handle id="source-bottom" type="source" position={Position.Bottom} style={handleStyle} />
      <div className="flex items-start gap-3">
        <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl border border-white/10 bg-white/8" style={{ color: data.color }}>
          <Icon size={22} />
        </div>
        <div className="min-w-0">
          <p className="text-base font-semibold leading-5 text-white">{data.label}</p>
          {data.subtitle && <p className="mt-2 text-xs leading-4 text-slate-400">{data.subtitle}</p>}
        </div>
      </div>
    </motion.div>
  );
}

const nodeTypes = { methodNode: MethodNode };
const LAYOUT_X_SCALE = 1.38;

function isInternalBaseline(flow: FlowDefinition) {
  return flow.id === "internal-signal-baseline";
}

function defaultPosition(flow: FlowDefinition, position: { x: number; y: number }) {
  return isInternalBaseline(flow) ? position : spaciousPosition(position);
}

function spaciousPosition(position: { x: number; y: number }) {
  return {
    x: Math.round(position.x * LAYOUT_X_SCALE),
    y: position.y
  };
}

function FlowCanvas({ selectedMethod }: { selectedMethod: string }) {
  const flow = useMemo(() => resolveFlow(selectedMethod), [selectedMethod]);
  const initialNodes = useMemo(() => createNodes(flow, readSavedPositions(flow.id)), [flow]);
  const initialEdges = useMemo(() => createEdges(flow), [flow]);
  const revealEvents = useMemo(() => buildRevealEvents(flow), [flow]);
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>(initialNodes);
  const [edges, setEdges] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<FlowNodeData | undefined>(initialNodes[0]?.data);
  const [status, setStatus] = useState("");
  const [activeStep, setActiveStep] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [userInteractedWithViewport, setUserInteractedWithViewport] = useState(false);
  const { fitView } = useReactFlow();
  const autoCameraRef = useRef(false);
  const sessionPositions = useRef<Record<string, Record<string, { x: number; y: number }>>>({});
  const flowNodeIds = useMemo(() => new Set(flow.nodes.map((node) => node.id)), [flow]);
  const visible = useMemo(() => visibleIds(revealEvents, activeStep), [activeStep, revealEvents]);
  const renderedNodes = useMemo(() => nodes.map((node) => ({ ...node, hidden: !visible.nodes.has(node.id) })), [nodes, visible.nodes]);
  const renderedEdges = useMemo(
    () => edges.map((edge) => ({ ...edge, hidden: !(visible.edges.has(edge.id) && visible.nodes.has(edge.source) && visible.nodes.has(edge.target)) })),
    [edges, visible.edges, visible.nodes]
  );
  const visibleNodeIds = useMemo(() => nodes.filter((node) => visible.nodes.has(node.id)).map((node) => node.id), [nodes, visible.nodes]);
  const runCamera = useCallback((nodeIds: string[], options?: { duration?: number; padding?: number; final?: boolean }) => {
    if (!nodeIds.length) return;
    const duration = options?.duration ?? 760;
    autoCameraRef.current = true;
    fitView({
      nodes: nodeIds.map((id) => ({ id })),
      padding: options?.padding ?? (options?.final ? 0.28 : 0.34),
      duration,
      minZoom: options?.final ? 0.42 : 0.5,
      maxZoom: options?.final ? 1.05 : 1.15,
      interpolate: "smooth"
    });
    window.setTimeout(() => {
      autoCameraRef.current = false;
    }, duration + 80);
  }, [fitView]);
  const fitFullFlow = useCallback((duration = 780) => {
    runCamera(nodes.map((node) => node.id), { duration, padding: 0.28, final: true });
  }, [nodes, runCamera]);

  useEffect(() => {
    const positions = sessionPositions.current[flow.id] ?? readSavedPositions(flow.id);
    const nextNodes = createNodes(flow, positions);
    setNodes(nextNodes);
    setEdges(createEdges(flow));
    setSelectedNode(nextNodes[0]?.data);
    setStatus("");
    setActiveStep(1);
    setPlaying(true);
    setUserInteractedWithViewport(false);
    window.setTimeout(() => runCamera([nextNodes[0]?.id].filter(Boolean), { duration: 720, padding: 0.38 }), 0);
  }, [flow, runCamera, setEdges, setNodes]);

  useEffect(() => {
    if (!playing) return;
    if (activeStep >= revealEvents.length) {
      setPlaying(false);
      return;
    }
    const timeout = window.setTimeout(() => {
      setActiveStep((step) => Math.min(step + 1, revealEvents.length));
    }, 560);
    return () => window.clearTimeout(timeout);
  }, [activeStep, playing, revealEvents.length]);

  useEffect(() => {
    if (userInteractedWithViewport || !visibleNodeIds.length) return;
    const isFinalStep = activeStep >= revealEvents.length;
    window.setTimeout(() => {
      runCamera(visibleNodeIds, { duration: isFinalStep ? 860 : 720, padding: isFinalStep ? 0.28 : 0.34, final: isFinalStep });
    }, 40);
  }, [activeStep, revealEvents.length, runCamera, userInteractedWithViewport, visibleNodeIds]);

  useEffect(() => {
    if (nodes.length && nodes.every((node) => flowNodeIds.has(node.id))) {
      sessionPositions.current[flow.id] = extractPositions(nodes);
    }
  }, [flow.id, flowNodeIds, nodes]);

  const handleNodesChange: OnNodesChange<FlowNode> = useCallback((changes) => {
    onNodesChange(changes);
  }, [onNodesChange]);

  const pauseForUserControl = useCallback(() => {
    if (autoCameraRef.current) return;
    setPlaying(false);
    setUserInteractedWithViewport(true);
    setStatus("Auto camera paused. Use Resume animation or Replay to follow the pipeline again.");
  }, []);

  const handleResumeAnimation = useCallback(() => {
    setUserInteractedWithViewport(false);
    setPlaying(activeStep < revealEvents.length);
    window.setTimeout(() => runCamera(visibleNodeIds, { duration: 720, padding: activeStep >= revealEvents.length ? 0.28 : 0.34, final: activeStep >= revealEvents.length }), 0);
  }, [activeStep, revealEvents.length, runCamera, visibleNodeIds]);

  const handleFit = useCallback(() => {
    setUserInteractedWithViewport(false);
    fitFullFlow(760);
  }, [fitFullFlow]);

  const handleResetLayout = useCallback(() => {
    const nextNodes = createNodes(flow);
    sessionPositions.current[flow.id] = extractPositions(nextNodes);
    setNodes(nextNodes);
    setSelectedNode(nextNodes[0]?.data);
    setActiveStep(revealEvents.length);
    setPlaying(false);
    setStatus("Layout reset for this session.");
    setUserInteractedWithViewport(false);
    window.setTimeout(() => runCamera(nextNodes.map((node) => node.id), { duration: 780, padding: 0.28, final: true }), 0);
  }, [flow, revealEvents.length, runCamera, setNodes]);

  const handleSaveLayout = useCallback(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKey(flow.id), JSON.stringify(extractPositions(nodes)));
    setStatus("Layout saved for this method.");
  }, [flow.id, nodes]);

  const handleResetSaved = useCallback(() => {
    if (typeof window !== "undefined") window.localStorage.removeItem(storageKey(flow.id));
    handleResetLayout();
    setStatus("Saved layout cleared.");
  }, [flow.id, handleResetLayout]);

  const handlePlay = useCallback(() => {
    if (isInternalBaseline(flow)) {
      const nextNodes = createNodes(flow);
      sessionPositions.current[flow.id] = extractPositions(nextNodes);
      setNodes(nextNodes);
      setSelectedNode(nextNodes[0]?.data);
    } else {
      setSelectedNode(nodes.find((node) => revealEvents[0]?.id === node.id)?.data ?? nodes[0]?.data);
    }
    setActiveStep(1);
    setUserInteractedWithViewport(false);
    setPlaying(true);
    const firstId = revealEvents[0]?.type === "node" ? revealEvents[0].id : nodes[0]?.id;
    window.setTimeout(() => runCamera([firstId].filter(Boolean), { duration: 720, padding: 0.38 }), 0);
  }, [flow, nodes, revealEvents, runCamera, setNodes]);

  const handleReplay = useCallback(() => {
    if (isInternalBaseline(flow)) {
      const nextNodes = createNodes(flow);
      sessionPositions.current[flow.id] = extractPositions(nextNodes);
      setNodes(nextNodes);
      setSelectedNode(nextNodes[0]?.data);
    } else {
      setSelectedNode(nodes.find((node) => revealEvents[0]?.id === node.id)?.data ?? nodes[0]?.data);
    }
    setActiveStep(1);
    setPlaying(true);
    setUserInteractedWithViewport(false);
    const firstId = revealEvents[0]?.type === "node" ? revealEvents[0].id : nodes[0]?.id;
    window.setTimeout(() => runCamera([firstId].filter(Boolean), { duration: 720, padding: 0.38 }), 0);
  }, [flow, nodes, revealEvents, runCamera, setNodes]);

  const handleShowAll = useCallback(() => {
    if (isInternalBaseline(flow)) {
      const nextNodes = createNodes(flow);
      sessionPositions.current[flow.id] = extractPositions(nextNodes);
      setNodes(nextNodes);
      setSelectedNode(nextNodes[0]?.data);
    }
    setActiveStep(revealEvents.length);
    setPlaying(false);
    setUserInteractedWithViewport(false);
    window.setTimeout(() => runCamera(nodes.map((node) => node.id), { duration: 820, padding: 0.28, final: true }), 0);
  }, [flow, nodes, revealEvents.length, runCamera, setNodes]);

  return (
    <div className="space-y-5">
      <div className="relative h-[72vh] min-h-[660px] max-h-[880px] overflow-hidden rounded-3xl border border-white/10 bg-slate-950/45 shadow-2xl backdrop-blur-xl">
        <div className="ai-content-absolute left-4 top-4 z-20 flex max-w-[calc(100%-2rem)] flex-wrap gap-2 rounded-3xl border border-white/10 bg-slate-950/70 p-2 backdrop-blur-xl">
          <button onClick={handlePlay} className="rounded-2xl bg-cyan-300 px-3 py-2 text-xs font-semibold text-slate-950">Play animation</button>
          <button onClick={() => setPlaying(false)} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Pause</button>
          {userInteractedWithViewport && <button onClick={handleResumeAnimation} className="rounded-2xl bg-emerald-300/15 px-3 py-2 text-xs font-semibold text-emerald-100">Resume animation</button>}
          <button onClick={handleReplay} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Replay</button>
          <button onClick={handleShowAll} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Show all</button>
          <button onClick={handleFit} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Fit view</button>
          <button onClick={handleResetLayout} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Reset layout</button>
          <button onClick={handleSaveLayout} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Save layout</button>
          <button onClick={handleResetSaved} className="rounded-2xl bg-white/10 px-3 py-2 text-xs font-semibold text-white">Reset saved positions</button>
        </div>
        <div className="ai-content-absolute bottom-4 left-4 z-20 rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-300 backdrop-blur-xl">
          Step {Math.min(activeStep, revealEvents.length)} / {revealEvents.length} {playing ? "- scanning" : ""}
        </div>
        <ReactFlow
          nodes={renderedNodes}
          edges={renderedEdges}
          nodeTypes={nodeTypes}
          onNodesChange={handleNodesChange}
          onNodeClick={(_, node) => setSelectedNode(node.data)}
          onNodeDragStart={pauseForUserControl}
          onMoveStart={(event) => {
            if (event) pauseForUserControl();
          }}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          panOnDrag
          zoomOnScroll
          fitView
          fitViewOptions={{ padding: 0.23, includeHiddenNodes: true }}
          minZoom={0.45}
          maxZoom={1.15}
          defaultEdgeOptions={{ type: "smoothstep" }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="rgba(148,163,184,0.24)" gap={24} />
          <MiniMap
            pannable
            zoomable
            nodeColor={() => flow.palette.color}
            maskColor="rgba(2,6,23,0.72)"
            style={{ background: "rgba(15,23,42,0.82)", border: "1px solid rgba(148,163,184,0.2)", borderRadius: 16 }}
          />
          <Controls />
        </ReactFlow>
      </div>

      <section className="glass rounded-3xl p-5">
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(340px,0.8fr)]">
          <div>
            <div className="flex items-start gap-3">
              <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl border border-white/10 bg-white/8" style={{ color: flow.palette.color }}>
                <Network size={22} />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.18em]" style={{ color: flow.palette.color }}>{flow.family}</p>
                <h2 className="mt-2 text-xl font-semibold text-white">{flow.title}</h2>
                <p className="mt-1 text-sm leading-6 text-slate-400">{flow.subtitle}</p>
              </div>
            </div>
            <p className="mt-4 max-w-5xl text-sm leading-7 text-slate-300">{flow.explanation}</p>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <div className="rounded-2xl bg-white/5 p-3">
                <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Required inputs</p>
                <p className="mt-2 text-sm text-slate-200">{flow.requiredInputs.join(", ")}</p>
              </div>
              <div className="rounded-2xl bg-white/5 p-3">
                <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Produces</p>
                <p className="mt-2 text-sm leading-6 text-slate-200">{flow.output}</p>
              </div>
              <div className="rounded-2xl bg-white/5 p-3">
                <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Key limitation</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">{flow.limitation}</p>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-cyan-200">Selected node details</p>
            {selectedNode ? (
              <div className="mt-3">
                <h3 className="text-lg font-semibold text-white">{selectedNode.label}</h3>
                {selectedNode.subtitle && <p className="mt-1 text-sm text-slate-400">{selectedNode.subtitle}</p>}
                <p className="mt-3 text-sm leading-7 text-slate-300">{selectedNode.description}</p>
              </div>
            ) : (
              <p className="mt-3 text-sm leading-6 text-slate-400">Click a node to inspect what that stage does.</p>
            )}
            {status && <p className="mt-4 rounded-2xl bg-cyan-300/10 p-3 text-xs text-cyan-100">{status}</p>}
          </div>
        </div>
      </section>
    </div>
  );
}

export function DetectorFlow({ selectedMethod = "" }: { selectedMethod?: string }) {
  return (
    <ReactFlowProvider>
      <FlowCanvas selectedMethod={selectedMethod} />
    </ReactFlowProvider>
  );
}
