import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export function EmptyState({
  icon: Icon,
  title,
  message,
  actions
}: {
  icon: LucideIcon;
  title: string;
  message: string;
  actions?: ReactNode;
}) {
  return (
    <div className="glass grid min-h-72 place-items-center rounded-3xl p-8 text-center">
      <div>
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-cyan-300/12 text-cyan-100 ring-1 ring-cyan-300/25">
          <Icon size={26} />
        </div>
        <h3 className="mt-5 text-xl font-semibold text-white">{title}</h3>
        <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-400">{message}</p>
        {actions && <div className="mt-5 flex flex-wrap justify-center gap-3">{actions}</div>}
      </div>
    </div>
  );
}
