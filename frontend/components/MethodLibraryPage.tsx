"use client";

import { useMemo, useState } from "react";
import type { DashboardState } from "@/lib/types";
import { MethodCard } from "./MethodCard";

export function MethodLibraryPage({ state }: { state: DashboardState }) {
  const [family, setFamily] = useState("all");
  const [implementation, setImplementation] = useState("all");
  const [requiredInput, setRequiredInput] = useState("all");
  const families = useMemo(() => Array.from(new Set(state.methods.map((m) => m.family))).sort(), [state.methods]);
  const implementations = useMemo(() => Array.from(new Set(state.methods.map((m) => m.implementation))).sort(), [state.methods]);
  const requiredInputs = useMemo(() => Array.from(new Set(state.methods.flatMap((m) => m.required_fields))).sort(), [state.methods]);
  const filtered = state.methods.filter((method) => {
    const matchesFamily = family === "all" || method.family === family;
    const matchesImplementation = implementation === "all" || method.implementation === implementation;
    const matchesInput = requiredInput === "all" || method.required_fields.includes(requiredInput);
    return matchesFamily && matchesImplementation && matchesInput;
  });

  return (
    <div className="space-y-5">
      <div className="glass rounded-3xl p-5">
        <h1 className="text-2xl font-semibold text-white">Method Library</h1>
        <p className="mt-1 text-sm text-slate-400">{state.backendOnline ? "Loaded from FastAPI metadata." : "Backend offline: showing local fallback metadata until FastAPI is started."}</p>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <select value={family} onChange={(e) => setFamily(e.target.value)} className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm outline-none">
            <option value="all">All families</option>
            {families.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <select value={implementation} onChange={(e) => setImplementation(e.target.value)} className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm outline-none">
            <option value="all">All implementation states</option>
            {implementations.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <select value={requiredInput} onChange={(e) => setRequiredInput(e.target.value)} className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm outline-none">
            <option value="all">All required inputs</option>
            {requiredInputs.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {filtered.map((m) => <MethodCard key={m.name} method={m} />)}
      </div>
    </div>
  );
}
