import type { PipelineInfo } from "../api";

interface Props {
  pipelines: Record<string, PipelineInfo>;
  activeId: string;
  onSelect: (id: string) => void;
}

const TAB_COLORS: Record<string, { active: string; inactive: string; dot: string }> = {
  energy: {
    active: "border-zinc-400 text-zinc-200",
    inactive: "border-transparent text-zinc-500 hover:text-zinc-400",
    dot: "bg-zinc-400",
  },
  channel: {
    active: "border-blue-500 text-blue-400",
    inactive: "border-transparent text-zinc-500 hover:text-zinc-400",
    dot: "bg-blue-500",
  },
  sepformer: {
    active: "border-purple-500 text-purple-400",
    inactive: "border-transparent text-zinc-500 hover:text-zinc-400",
    dot: "bg-purple-500",
  },
};

const PIPELINE_ORDER = ["energy", "channel", "sepformer"];

function formatTime(s: number): string {
  if (s < 1) return "<1s";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

export default function PipelineTabs({ pipelines, activeId, onSelect }: Props) {
  const orderedIds = PIPELINE_ORDER.filter((id) => id in pipelines);
  const active = pipelines[activeId];

  return (
    <div>
      {/* Tab bar */}
      <div className="flex border-b border-zinc-800">
        {orderedIds.map((id) => {
          const p = pipelines[id];
          const colors = TAB_COLORS[id] ?? TAB_COLORS.energy;
          const isActive = id === activeId;

          return (
            <button
              key={id}
              onClick={() => onSelect(id)}
              className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
                isActive ? colors.active : colors.inactive
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${colors.dot}`} />
              {p.label}
            </button>
          );
        })}
      </div>

      {/* Metadata row for active pipeline */}
      {active && (
        <div className="flex items-center gap-4 px-4 py-2 text-[11px] text-zinc-500">
          <span>
            Time: <span className="text-zinc-400">{formatTime(active.elapsed_s)}</span>
          </span>
          {active.segments > 0 && (
            <span>
              Segments: <span className="text-zinc-400">{active.segments}</span>
            </span>
          )}
          {active.segments > 0 && (
            <span>
              Overlap: <span className="text-zinc-400">{active.overlap_pct}%</span>
            </span>
          )}
          {active.id === "energy" && (
            <span className="text-zinc-600">No GPU required</span>
          )}
        </div>
      )}
    </div>
  );
}
