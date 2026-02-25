import type { PipelineInfo } from "../api";

interface Props {
  pipelines: Record<string, PipelineInfo>;
  activeId: string;
  onSelect: (id: string) => void;
}

const PIPE_CFG: Record<string, { color: string; label: string }> = {
  energy:    { color: "#71717a", label: "Energy" },
  channel:   { color: "#60a5fa", label: "Channel" },
  sepformer: { color: "#a78bfa", label: "SepFormer" },
  nemo:      { color: "#fbbf24", label: "NeMo" },
  diarizen:  { color: "#f87171", label: "DiariZen" },
};

const ORDER = ["energy", "channel", "sepformer", "nemo", "diarizen"];

function fmtTime(s: number): string {
  if (s < 1) return "<1s";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

export default function PipelineTabs({ pipelines, activeId, onSelect }: Props) {
  const ids = ORDER.filter(id => id in pipelines);
  const active = pipelines[activeId];
  const cfg = PIPE_CFG[activeId] ?? PIPE_CFG.energy;

  return (
    <div>
      {/* Tab row */}
      <div className="flex overflow-x-auto" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
        {ids.map(id => {
          const c = PIPE_CFG[id] ?? PIPE_CFG.energy;
          const isActive = id === activeId;
          const p = pipelines[id];
          return (
            <button
              key={id}
              onClick={() => onSelect(id)}
              className="px-3 py-2 text-xs font-medium transition-colors whitespace-nowrap flex-shrink-0 flex items-center gap-1.5"
              id={`pipeline-tab-${id}`}
              style={{
                color: isActive ? c.color : "#52525b",
                borderBottom: isActive ? `2px solid ${c.color}` : "2px solid transparent",
              }}
            >
              {c.label}
              {p.error && <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#f87171" }} />}
            </button>
          );
        })}
      </div>

      {/* Stats row */}
      {active && (
        <div className="px-4 py-2 flex items-center gap-4 text-[11px]" style={{ color: "#52525b" }}>
          <span style={{ color: cfg.color }}>{cfg.label}</span>
          {active.elapsed_s > 0 && <span>{fmtTime(active.elapsed_s)}</span>}
          {active.segments > 0 && <span>{active.segments} segs</span>}
          {active.overlap_pct > 0 && <span>{active.overlap_pct}% overlap</span>}
          {active.error && <span style={{ color: "#f87171" }}>{active.error.slice(0, 60)}</span>}
        </div>
      )}
    </div>
  );
}
