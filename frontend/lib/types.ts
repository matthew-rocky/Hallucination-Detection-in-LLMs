export type TabId =
  | "overview"
  | "askQuick"
  | "compareDetectors"
  | "samples"
  | "results"
  | "compareResults"
  | "flow"
  | "library"
  | "report";

export type Mode = "quick" | "compare";

export interface MethodInfo {
  id: string;
  name: string;
  family: string;
  short_purpose: string;
  best_for: string;
  how_it_works: string;
  required_fields: string[];
  optional_fields: string[];
  visible_fields: string[];
  ignored_fields: string[];
  supports_uploads: boolean;
  implementation: string;
  caption: string;
  input_requirements: {
    required: string[];
    optional: string[];
    visible: string[];
  };
  strengths?: string[];
  weaknesses?: string[];
  recommended_use?: string;
  color?: string;
  tone?: string;
  profile: Record<string, unknown>;
}

export interface FieldSpec {
  label: string;
  short_label: string;
  helper: string;
  placeholder: string;
  height: number;
}

export interface SampleCase {
  id: string;
  pair_id: string;
  method_targets: string[];
  recommended_methods?: string[];
  risk_level: string;
  title: string;
  description: string;
  question: string;
  answer: string;
  answer_samples: string;
  sampled_answers_text?: string;
  source_text: string;
  evidence_text: string;
  expected_label: string;
  notes: string;
  question_preview?: string;
  answer_preview?: string;
  evidence_preview?: string;
  available_inputs?: string[];
}

export interface DetectorResult {
  method_name: string;
  family: string;
  score: number | null;
  label: string;
  risk_score: number | null;
  risk_label: string;
  confidence: number | null;
  summary?: string;
  explanation: string;
  evidence_used?: string;
  evidence: Record<string, unknown>[];
  citations: Record<string, unknown>[];
  claim_findings: Record<string, unknown>[];
  intermediate_steps: Record<string, unknown>[];
  revised_answer?: string | null;
  limitations?: string;
  implementation_status: string;
  available: boolean;
  runtime_status?: string;
  latency_ms?: number | null;
  metadata: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AnalyzePayload {
  mode: Mode;
  selected_methods: string[];
  question: string;
  answer: string;
  source_text: string;
  evidence_text: string;
  sampled_answers_text: string;
}

export interface AnalysisSummary {
  method_count: number;
  claims_checked: number;
  low: number;
  medium: number;
  high: number;
  avg_risk: number;
}

export interface AnalyzeResponse {
  ok: boolean;
  mode: Mode;
  selected_methods: string[];
  results: DetectorResult[];
  summary: AnalysisSummary;
  warnings: string[];
}

export interface Health {
  ok: boolean;
  status: "online" | string;
  message: string;
  method_count: number;
  sample_count: number;
}

export interface StudioForm {
  question: string;
  answer: string;
  source_text: string;
  evidence_text: string;
  sampled_answers_text: string;
}

export interface DashboardState extends StudioForm {
  activeTab: TabId;
  backendOnline: boolean;
  backendMessage: string;
  methods: MethodInfo[];
  samples: SampleCase[];
  samplePairs: Record<string, { low?: SampleCase | null; high?: SampleCase | null }>;
  fields: Record<string, FieldSpec>;
  mode: Mode;
  selectedMethods: string[];
  uploadedFiles: File[];
  loadedSample?: SampleCase;
  results: DetectorResult[];
  selectedResult?: DetectorResult;
  summary?: AnalysisSummary;
  analysisHistory: AnalysisHistoryItem[];
  loading: boolean;
  error: string;
  lastRunAt?: string;
  lastAnalyzedSampleId?: string;
}

export interface AnalysisHistoryItem {
  id: string;
  timestamp: string;
  mode: Mode;
  selected_methods: string[];
  question_preview: string;
  summary: AnalysisSummary;
  highest_risk_method?: string;
  highest_risk_label?: string;
  avg_risk: number;
}
