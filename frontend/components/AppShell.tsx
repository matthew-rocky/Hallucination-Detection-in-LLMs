"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { analyze, getFields, getHealth, getMethods, getSamplePairs, getSamples, uploadAnalyze } from "@/lib/api";
import { fallbackFields, fallbackMethods } from "@/lib/methodFallback";
import { normalizeRiskLabel } from "@/lib/risk";
import type { AnalysisHistoryItem, AnalyzeResponse, DashboardState, DetectorResult, SampleCase, StudioForm, TabId } from "@/lib/types";
import { AnalyzePage } from "./AnalyzePage";
import { AskStudioPage } from "./AskStudioPage";
import { ErrorBanner } from "./ErrorBanner";
import { LoadingState } from "./LoadingState";
import { MethodFlowPage } from "./MethodFlowPage";
import { MethodLibraryPage } from "./MethodLibraryPage";
import { OverviewPage } from "./OverviewPage";
import { ReportPage } from "./ReportPage";
import { ResultsPage } from "./ResultsPage";
import { SamplesPage } from "./SamplesPage";
import { Sidebar } from "./Sidebar";
import { TopStatusBar } from "./TopStatusBar";

const emptySummary = { method_count: 0, claims_checked: 0, low: 0, medium: 0, high: 0, avg_risk: 0 };

const initialState: DashboardState = {
  activeTab: "overview",
  backendOnline: false,
  backendMessage: "Checking backend...",
  methods: fallbackMethods,
  samples: [],
  samplePairs: {},
  fields: fallbackFields,
  mode: "quick",
  selectedMethods: [fallbackMethods[0].name],
  uploadedFiles: [],
  loadedSample: undefined,
  question: "",
  answer: "",
  source_text: "",
  evidence_text: "",
  sampled_answers_text: "",
  results: [],
  selectedResult: undefined,
  summary: emptySummary,
  analysisHistory: [],
  loading: false,
  error: "",
  lastRunAt: undefined,
  lastAnalyzedSampleId: undefined
};

export function AppShell() {
  const [state, setState] = useState<DashboardState>(initialState);

  useEffect(() => {
    refreshBackend();
  }, []);

  const refreshBackend = async () => {
    const [healthRes, methodsRes, fieldsRes, samplesRes, pairsRes] = await Promise.allSettled([getHealth(), getMethods(), getFields(), getSamples(), getSamplePairs()]);
    const health = healthRes.status === "fulfilled" ? healthRes.value : undefined;
    const methods = methodsRes.status === "fulfilled" && methodsRes.value.length ? methodsRes.value : fallbackMethods;
    const fields = fieldsRes.status === "fulfilled" ? { ...fallbackFields, ...fieldsRes.value } : fallbackFields;
    const samples = samplesRes.status === "fulfilled" ? samplesRes.value : [];
    const samplePairs = pairsRes.status === "fulfilled" ? pairsRes.value : {};
    const failures = [healthRes, methodsRes, fieldsRes, samplesRes, pairsRes]
      .filter((item): item is PromiseRejectedResult => item.status === "rejected")
      .map((item) => item.reason instanceof Error ? item.reason.message : "A backend request failed.");
    setState((current) => ({
      ...current,
      backendOnline: Boolean(health?.ok),
      backendMessage: health?.message ?? "Backend offline",
      methods,
      fields,
      samples,
      samplePairs,
      selectedMethods: currentSelection(current.selectedMethods, methods),
      error: failures.length && !health?.ok ? failures[0] : ""
    }));
  };

  const currentSelection = (selection: string[], methods: typeof fallbackMethods) => {
    const valid = Array.from(new Set(selection.filter((name) => methods.some((m) => m.name === name))));
    return valid.length ? valid : [methods[0]?.name ?? fallbackMethods[0].name];
  };

  const setTab = (activeTab: TabId) => setState((s) => {
    if (activeTab === "askQuick") {
      return { ...s, activeTab, mode: "quick", selectedMethods: [s.selectedMethods[0] ?? s.methods[0].name] };
    }
    if (activeTab === "compareDetectors") {
      return { ...s, activeTab, mode: "compare" };
    }
    return { ...s, activeTab };
  });
  const setField = (key: keyof StudioForm, value: string) => setState((s) => ({ ...s, [key]: value }));

  const toggleMethod = (method: string) => {
    setState((s) => {
      if (s.mode === "quick") return { ...s, selectedMethods: [method] };
      const selected = s.selectedMethods.includes(method) ? s.selectedMethods.filter((item) => item !== method) : [...s.selectedMethods, method];
      return { ...s, selectedMethods: selected.length ? selected : [method] };
    });
  };

  const clear = () => setState((s) => ({ ...s, question: "", answer: "", source_text: "", evidence_text: "", sampled_answers_text: "", uploadedFiles: [], loadedSample: undefined, results: [], selectedResult: undefined, summary: emptySummary, lastAnalyzedSampleId: undefined, error: "" }));

  const payload = useMemo(() => ({
    mode: state.mode,
    selected_methods: state.selectedMethods,
    question: state.question,
    answer: state.answer,
    source_text: state.source_text,
    evidence_text: state.evidence_text,
    sampled_answers_text: state.sampled_answers_text
  }), [state.mode, state.selectedMethods, state.question, state.answer, state.source_text, state.evidence_text, state.sampled_answers_text]);

  const runAnalysis = async () => {
    if (!state.backendOnline) {
      setState((s) => ({ ...s, error: "Backend offline. Start it with: python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000" }));
      return;
    }
    setState((s) => ({ ...s, loading: true, error: "" }));
    try {
      const response = state.uploadedFiles.length ? await uploadAnalyze(payload, state.uploadedFiles) : await analyze(payload);
      commitAnalysisResponse(response, undefined, state.loadedSample?.id, state.activeTab === "compareDetectors");
    } catch (exc) {
      setState((s) => ({ ...s, loading: false, error: exc instanceof Error ? exc.message : "Analysis failed." }));
    }
  };

  const loadSample = (sample: SampleCase, tab: "askQuick" | "compareDetectors" | "results" = "askQuick") => {
    const method = sample.recommended_methods?.[0] ?? sample.method_targets?.[0] ?? fallbackMethods[0].name;
    setState((s) => ({
      ...s,
      activeTab: tab,
      selectedMethods: tab === "compareDetectors" ? currentSelection([...(sample.recommended_methods ?? []), ...(sample.method_targets ?? [])], s.methods) : [method],
      mode: tab === "compareDetectors" ? "compare" : "quick",
      question: sample.question,
      answer: sample.answer,
      source_text: sample.source_text,
      evidence_text: sample.evidence_text,
      sampled_answers_text: sample.answer_samples || sample.sampled_answers_text || "",
      loadedSample: sample,
      error: ""
    }));
  };

  const runSample = async (sample: SampleCase) => {
    loadSample(sample, "askQuick");
    const method = sample.recommended_methods?.[0] ?? sample.method_targets?.[0] ?? fallbackMethods[0].name;
    if (!state.backendOnline) return;
    setState((s) => ({ ...s, activeTab: "askQuick", mode: "quick", selectedMethods: [method], loadedSample: sample, loading: true, error: "" }));
    try {
      const response = await analyze({
        mode: "quick",
        selected_methods: [method],
        question: sample.question,
        answer: sample.answer,
        source_text: sample.source_text,
        evidence_text: sample.evidence_text,
        sampled_answers_text: sample.answer_samples || ""
      });
      commitAnalysisResponse(response, sample.question, sample.id);
    } catch (exc) {
      setState((s) => ({ ...s, loading: false, error: exc instanceof Error ? exc.message : "Sample analysis failed." }));
    }
  };

  const quickLoadSample = () => {
    const sample = state.samples[0];
    if (sample) loadSample(sample, "askQuick");
    else setTab("samples");
  };

  const compareLoadSample = () => {
    const sample = state.samples[0];
    if (sample) loadSample(sample, "compareDetectors");
    else setTab("samples");
  };

  const loadRiskSample = (risk: "Low" | "High", tab: "askQuick" | "compareDetectors" = "askQuick") => {
    const selected = state.selectedMethods[0];
    const sample =
      state.samples.find((item) => normalizeRiskLabel(item.risk_level) === risk && (item.method_targets?.includes(selected) || item.recommended_methods?.includes(selected))) ??
      state.samples.find((item) => normalizeRiskLabel(item.risk_level) === risk);
    if (sample) loadSample(sample, tab);
    else setTab("samples");
  };

  const setSelectedResult = (name: string) => {
    const selectedResult = state.results.find((result: DetectorResult) => result.method_name === name);
    setState((s) => ({ ...s, selectedResult }));
  };

  const selectFlowMethod = (name: string) => setState((s) => ({ ...s, selectedMethods: [name] }));

  const makeHistoryItem = (response: AnalyzeResponse, question: string): AnalysisHistoryItem => {
    const ranked = [...response.results].sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1));
    const top = ranked[0];
    const compactQuestion = question.trim().replace(/\s+/g, " ");
    return {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      timestamp: new Date().toISOString(),
      mode: response.mode,
      selected_methods: response.selected_methods,
      question_preview: compactQuestion ? (compactQuestion.length > 120 ? `${compactQuestion.slice(0, 119)}...` : compactQuestion) : "Untitled analysis",
      summary: response.summary,
      highest_risk_method: top?.method_name,
      highest_risk_label: top?.risk_label,
      avg_risk: response.summary.avg_risk
    };
  };

  const commitAnalysisResponse = (response: AnalyzeResponse, questionOverride?: string, sampleId?: string, openReport = false) => {
    const ranked = [...response.results].sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1));
    const selectedResult = response.results.length > 1 ? ranked[0] : response.results[0];
    const historyItem = makeHistoryItem(response, questionOverride ?? state.question);
    setState((s) => ({
      ...s,
      activeTab: openReport ? "report" : s.activeTab,
      loading: false,
      results: response.results,
      selectedResult,
      summary: response.summary,
      analysisHistory: [historyItem, ...s.analysisHistory].slice(0, 20),
      lastRunAt: historyItem.timestamp,
      lastAnalyzedSampleId: sampleId ?? s.loadedSample?.id,
      error: response.warnings.join(" ")
    }));
  };

  const page = {
    overview: <OverviewPage state={state} setTab={setTab} loadSample={quickLoadSample} />,
    askQuick: <AskStudioPage state={state} setField={setField} toggleMethod={toggleMethod} run={runAnalysis} loadRiskSample={loadRiskSample} loadSelectedSample={(sample) => loadSample(sample, "askQuick")} runSample={runSample} setTab={setTab} />,
    compareDetectors: <AnalyzePage state={state} fields={state.fields} files={state.uploadedFiles} toggleMethod={toggleMethod} setField={setField} setFiles={(uploadedFiles) => setState((s) => ({ ...s, uploadedFiles }))} clear={clear} run={runAnalysis} loadSample={compareLoadSample} />,
    samples: <SamplesPage state={state} loadSample={loadSample} runSample={runSample} />,
    results: <ResultsPage state={state} setSelectedResult={setSelectedResult} setTab={setTab} loadRiskSample={loadRiskSample} />,
    flow: <MethodFlowPage state={state} selectMethod={selectFlowMethod} />,
    library: <MethodLibraryPage state={state} />,
    report: <ReportPage state={state} setTab={setTab} />
  }[state.activeTab];

  return (
    <main className="relative z-10 flex min-h-screen w-full gap-5 p-3 md:p-5">
      <Sidebar activeTab={state.activeTab} onTabChange={setTab} state={state} onRunSample={quickLoadSample} />
      <div className="min-w-0 flex-1 space-y-5">
        <TopStatusBar online={state.backendOnline} methodCount={state.methods.length} mode={state.mode} selectedCount={state.selectedMethods.length} />
        <ErrorBanner message={state.error} offline={!state.backendOnline} />
        {state.loading && <LoadingState />}
        <AnimatePresence mode="wait">
          <motion.div key={state.activeTab} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.22 }}>
            {page}
          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  );
}
