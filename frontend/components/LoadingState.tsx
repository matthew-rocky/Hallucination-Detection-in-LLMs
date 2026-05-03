export function LoadingState({ label = "Running detector methods..." }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-4 text-sm text-cyan-100">
      <span className="h-3 w-3 animate-ping rounded-full bg-cyan-300" />
      {label}
    </div>
  );
}
