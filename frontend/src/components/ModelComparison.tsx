import type { PipelineInfo } from "../api";

interface Props {
  pipelines: Record<string, PipelineInfo>;
}

const CFG: Record<string, { color: string; label: string }> = {
  energy:    { color: "#71717a", label: "Energy" },
  channel:   { color: "#60a5fa", label: "Channel" },
  sepformer: { color: "#a78bfa", label: "SepFormer" },
  nemo:      { color: "#fbbf24", label: "NeMo" },
  diarizen:  { color: "#f87171", label: "DiariZen" },
};

const ORDER = ["energy", "channel", "sepformer", "nemo", "diarizen"];

export default function ModelComparison({ pipelines }: Props) {
  const ids = ORDER.filter(id => id in pipelines);
  if (ids.length < 2) return null;

  const ml = ids.filter(id => id !== "energy");
  const fastest = ml.reduce((a, b) =>
    (pipelines[a]?.elapsed_s ?? Infinity) < (pipelines[b]?.elapsed_s ?? Infinity) ? a : b,
    ml[0]
  );
  const maxTime = Math.max(...ids.map(id => pipelines[id]?.elapsed_s ?? 0));
  const maxSegs = Math.max(...ids.map(id => pipelines[id]?.segments ?? 0));

  return (
    <div className="space-y-2">
      {ids.map(id => {
        const p = pipelines[id];
        const c = CFG[id] ?? { color: "#fff", label: id };
        const isFastest = id === fastest;

        return (
          <div key={id} className="rounded-lg p-3" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)" }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: c.color }} />
                <span className="text-xs font-medium" style={{ color: c.color }}>{c.label}</span>
                {isFastest && id !== "energy" && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: `${c.color}15`, color: c.color }}>
                    Fastest
                  </span>
                )}
                {p.error && <span className="text-[10px]" style={{ color: "#f87171" }}>Error</span>}
              </div>
              <div className="flex items-center gap-3 text-[11px]" style={{ color: "#52525b" }}>
                {p.elapsed_s > 0 && <span>{p.elapsed_s.toFixed(0)}s</span>}
                {p.segments > 0 && <span>{p.segments} segs</span>}
                {p.overlap_pct > 0 && <span>{p.overlap_pct}%</span>}
              </div>
            </div>

            {/* Time bar */}
            {maxTime > 0 && p.elapsed_s > 0 && (
              <div className="h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                <div
                  className="h-full rounded-full"
                  style={{ width: `${Math.max(2, (1 - p.elapsed_s / maxTime) * 100)}%`, background: c.color, opacity: 0.6 }}
                />
              </div>
            )}
            {maxSegs > 0 && p.segments > 0 && (
              <div className="h-1 rounded-full overflow-hidden mt-1" style={{ background: "rgba(255,255,255,0.04)" }}>
                <div
                  className="h-full rounded-full"
                  style={{ width: `${(p.segments / maxSegs) * 100}%`, background: c.color, opacity: 0.35 }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
