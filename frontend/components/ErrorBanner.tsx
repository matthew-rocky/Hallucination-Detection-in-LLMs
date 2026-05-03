import { AlertTriangle, Server } from "lucide-react";

export function ErrorBanner({ message, offline }: { message: string; offline?: boolean }) {
  if (!message) return null;
  return (
    <div className="rounded-2xl border border-amber-300/25 bg-amber-300/10 p-4 text-sm text-amber-100">
      <div className="flex items-start gap-3">
        {offline ? <Server className="mt-0.5 shrink-0" size={18} /> : <AlertTriangle className="mt-0.5 shrink-0" size={18} />}
        <div>
          <p className="font-semibold">{offline ? "Backend offline" : "Dashboard notice"}</p>
          <p className="mt-1 leading-6">{message}</p>
          {offline && (
            <code className="mt-3 block rounded-xl bg-slate-950/60 p-3 text-xs text-cyan-100">
              python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
            </code>
          )}
        </div>
      </div>
    </div>
  );
}
