function textOf(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (value != null && String(value).trim()) return String(value);
  }
  return "";
}

export function EvidenceCard({ item, index }: { item: Record<string, unknown>; index: number }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <p className="text-xs uppercase tracking-[0.16em] text-cyan-200">{textOf(item, ["title", "citation_id", "evidence_id"]) || `Evidence ${index + 1}`}</p>
      <p className="mt-3 text-sm leading-6 text-slate-300">{textOf(item, ["content", "snippet", "text"]) || "No excerpt available."}</p>
    </div>
  );
}
