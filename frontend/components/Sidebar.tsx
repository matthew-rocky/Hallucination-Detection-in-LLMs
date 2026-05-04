"use client";

import { BarChart3, BookOpen, Bot, FileDown, FlaskConical, GitBranch, LayoutDashboard, Library, Network, SearchCheck } from "lucide-react";
import type { TabId } from "@/lib/types";

const tabs: { id: TabId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "askQuick", label: "ASK Quick Mode", icon: Bot },
  { id: "compareDetectors", label: "Compare Detectors", icon: FlaskConical },
  { id: "samples", label: "Samples", icon: BookOpen },
  { id: "results", label: "Results", icon: SearchCheck },
  { id: "compareResults", label: "Compare Results", icon: BarChart3 },
  { id: "flow", label: "Method Flow", icon: GitBranch },
  { id: "library", label: "Method Library", icon: Library },
  { id: "report", label: "Report / Export", icon: FileDown }
];

export function Sidebar({ activeTab, onTabChange }: { activeTab: TabId; onTabChange: (tab: TabId) => void }) {
  return (
    <aside className="glass sticky top-5 hidden h-[calc(100vh-2.5rem)] w-72 shrink-0 rounded-3xl p-5 lg:block">
      <div className="mb-7 flex items-center gap-3">
        <div className="grid h-12 w-12 place-items-center rounded-2xl bg-cyan-300/15 text-cyan-100 ring-1 ring-cyan-300/30">
          <Network size={23} />
        </div>
        <div>
          <p className="text-sm font-semibold text-white">Hallucination</p>
          <p className="text-xs text-slate-400">Detection Studio</p>
        </div>
      </div>
      <nav className="space-y-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left text-sm transition ${
                active ? "bg-cyan-300 text-slate-950 shadow-glow" : "text-slate-300 hover:bg-white/8 hover:text-white"
              }`}
            >
              <Icon size={18} />
              {tab.label}
            </button>
          );
        })}
      </nav>
      <div className="mt-6 rounded-2xl border border-cyan-300/15 bg-cyan-300/8 p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-200">Local-first</p>
        <p className="mt-2 text-sm leading-6 text-slate-300">Next.js renders the studio. FastAPI calls the existing Python detector methods.</p>
      </div>
    </aside>
  );
}
