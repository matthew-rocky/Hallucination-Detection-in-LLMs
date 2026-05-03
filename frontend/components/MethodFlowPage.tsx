"use client";

import type { DashboardState } from "@/lib/types";
import { DetectorFlow } from "./DetectorFlow";

export function MethodFlowPage({ state, selectMethod }: { state: DashboardState; selectMethod: (name: string) => void }) {
  const selected = state.selectedMethods[0] ?? "";
  return (
    <div className="space-y-5">
      <div className="glass rounded-3xl p-5">
        <h1 className="text-2xl font-semibold text-white">Method Flow</h1>
        <p className="mt-1 text-sm text-slate-400">Animated detector pipeline. Select a method to highlight the conceptual path used by that family.</p>
        <div className="mt-4 flex flex-wrap gap-2">{state.methods.map((m) => <button key={m.name} onClick={() => selectMethod(m.name)} className={`rounded-full px-3 py-2 text-xs ${selected === m.name ? "bg-cyan-300 text-slate-950" : "bg-white/8 text-slate-300"}`}>{m.name}</button>)}</div>
      </div>
      <DetectorFlow selectedMethod={selected} />
    </div>
  );
}
