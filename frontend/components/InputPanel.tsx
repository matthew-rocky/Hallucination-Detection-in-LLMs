"use client";

import { FileText, Play, UploadCloud } from "lucide-react";
import type { FieldSpec } from "@/lib/types";

const fieldMap: Record<string, keyof FormState> = {
  question: "question",
  answer: "answer",
  source_text: "source_text",
  evidence_text: "evidence_text",
  sampled_answers: "sampled_answers_text"
};

export interface FormState {
  question: string;
  answer: string;
  source_text: string;
  evidence_text: string;
  sampled_answers_text: string;
}

export function InputPanel({
  visibleFields,
  fields,
  form,
  files,
  loading,
  onFormChange,
  onFilesChange,
  onRun
}: {
  visibleFields: string[];
  fields: Record<string, FieldSpec>;
  form: FormState;
  files: File[];
  loading: boolean;
  onFormChange: (next: FormState) => void;
  onFilesChange: (files: File[]) => void;
  onRun: () => void;
}) {
  return (
    <div className="glass rounded-3xl p-5">
      <div className="mb-5 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-cyan-200">Inputs</p>
          <h2 className="mt-1 text-xl font-semibold text-white">Dynamic method form</h2>
        </div>
        <FileText className="text-cyan-200" />
      </div>
      <div className="space-y-4">
        {visibleFields.map((fieldKey) => {
          if (fieldKey === "uploaded_documents") {
            return (
              <label key={fieldKey} className="block rounded-2xl border border-dashed border-cyan-300/35 bg-cyan-300/8 p-4">
                <span className="mb-2 flex items-center gap-2 text-sm font-medium text-cyan-100">
                  <UploadCloud size={17} /> Upload evidence documents
                </span>
                <input
                  type="file"
                  multiple
                  accept=".txt,.md,.json,.jsonl,.pdf"
                  className="block w-full text-sm text-slate-300 file:mr-3 file:rounded-lg file:border-0 file:bg-cyan-300 file:px-3 file:py-2 file:text-sm file:font-medium file:text-slate-950"
                  onChange={(event) => onFilesChange(Array.from(event.target.files ?? []))}
                />
                {files.length > 0 && <p className="mt-2 text-xs text-slate-400">{files.length} file(s) ready for upload-aware methods.</p>}
              </label>
            );
          }
          const spec = fields[fieldKey];
          const formKey = fieldMap[fieldKey];
          if (!spec || !formKey) return null;
          return (
            <label key={fieldKey} className="block">
              <span className="mb-2 block text-sm font-medium text-slate-200">{spec.label}</span>
              <textarea
                value={form[formKey]}
                onChange={(event) => onFormChange({ ...form, [formKey]: event.target.value })}
                placeholder={spec.placeholder}
                rows={Math.max(4, Math.round((spec.height || 150) / 38))}
                className="w-full resize-y rounded-2xl border border-white/10 bg-slate-950/55 px-4 py-3 text-sm leading-6 text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/15"
              />
              <span className="mt-1 block text-xs text-slate-500">{spec.helper}</span>
            </label>
          );
        })}
      </div>
      <button
        onClick={onRun}
        disabled={loading}
        className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-300 to-emerald-300 px-5 py-4 text-sm font-semibold text-slate-950 shadow-glow transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
      >
        <Play size={18} /> {loading ? "Running detectors..." : "Run analysis"}
      </button>
    </div>
  );
}
