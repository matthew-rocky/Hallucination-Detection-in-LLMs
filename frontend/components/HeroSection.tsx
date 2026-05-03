"use client";

import { motion } from "framer-motion";
import { ShieldCheck, Sparkles } from "lucide-react";

export function HeroSection({ backendStatus }: { backendStatus: string }) {
  return (
    <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-[linear-gradient(135deg,rgba(20,184,166,0.2),rgba(99,102,241,0.12),rgba(236,72,153,0.13))] p-6 shadow-glow md:p-8">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300 to-transparent" />
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <div className="mb-5 flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1 text-xs font-medium text-cyan-100">
            <ShieldCheck size={14} /> Full-stack detector lab
          </span>
          <span className="inline-flex items-center gap-2 rounded-full border border-emerald-300/25 bg-emerald-300/10 px-3 py-1 text-xs font-medium text-emerald-100">
            <Sparkles size={14} /> Backend {backendStatus}
          </span>
        </div>
        <h1 className="max-w-4xl text-4xl font-semibold leading-tight text-white md:text-6xl">
          Hallucination Detection Studio
        </h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300 md:text-lg">
          Compare internal-signal, retrieval-grounded, verification, CoVe, and CRITIC-style checks through a modern research dashboard backed by the existing Python methods.
        </p>
      </motion.div>
    </section>
  );
}
