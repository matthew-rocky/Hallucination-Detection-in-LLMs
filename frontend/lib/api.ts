import type { AnalyzePayload, AnalyzeResponse, FieldSpec, Health, MethodInfo, SampleCase } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const BACKEND_COMMAND = "python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 8000);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {})
      }
    });
  } catch (exc) {
    throw new Error(`Backend offline or unreachable at ${API_BASE}. Start it with: ${BACKEND_COMMAND}`);
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getHealth() {
  return request<Health>("/health");
}

export function getMethods() {
  return request<MethodInfo[]>("/api/methods");
}

export function getFields() {
  return request<Record<string, FieldSpec>>("/api/fields");
}

export function getSamples() {
  return request<SampleCase[]>("/api/samples");
}

export function getSamplePairs() {
  return request<Record<string, { low?: SampleCase | null; high?: SampleCase | null }>>("/api/sample-pairs");
}

export function analyze(payload: AnalyzePayload) {
  return request<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function uploadAnalyze(payload: AnalyzePayload, files: File[]) {
  const form = new FormData();
  form.append("mode", payload.mode);
  form.append("selected_methods", JSON.stringify(payload.selected_methods));
  form.append("question", payload.question);
  form.append("answer", payload.answer);
  form.append("source_text", payload.source_text);
  form.append("evidence_text", payload.evidence_text);
  form.append("sampled_answers_text", payload.sampled_answers_text);
  files.forEach((file) => form.append("files", file));

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/upload-analyze`, { method: "POST", body: form });
  } catch {
    throw new Error(`Backend offline or unreachable at ${API_BASE}. Start it with: python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`);
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<AnalyzeResponse>;
}
