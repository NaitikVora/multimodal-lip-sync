import { useEffect, useRef, useState, useCallback } from "react";
import type { TranscriptUtterance, Transcript } from "../api";
import { fetchTranscript, startTranscription } from "../api";

interface Props {
  sessionNum: number;
  hasTranscript: boolean;
  currentTime?: number;
  onSeek?: (time: number) => void;
}

const SPK = {
  A: { bubble: "speaker-bubble-a", label: "Speaker A", color: "#818cf8" },
  B: { bubble: "speaker-bubble-b", label: "Speaker B", color: "#34d399" },
};

function fmt(s: number): string {
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

export default function TranscriptPanel({ sessionNum, hasTranscript, currentTime, onSeek }: Props) {
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "A" | "B">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [view, setView] = useState<"chat" | "raw">("chat");
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadTranscript = useCallback(async () => {
    try {
      const data = await fetchTranscript(sessionNum);
      if (data.error) { setError(data.error); }
      else { setTranscript(data); }
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    } catch { /* not ready */ }
  }, [sessionNum]);

  useEffect(() => {
    if (hasTranscript) { setLoading(true); loadTranscript().finally(() => setLoading(false)); }
  }, [hasTranscript, loadTranscript]);

  const handleGenerate = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const r = await startTranscription(sessionNum);
      if (r.status === "cached") { await loadTranscript(); setLoading(false); }
      else {
        pollRef.current = setInterval(async () => {
          try {
            const data = await fetchTranscript(sessionNum);
            if (data.utterances?.length > 0) {
              setTranscript(data); setLoading(false);
              if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
            }
          } catch { /* still loading */ }
        }, 3000);
      }
    } catch (e) { setError(String(e)); setLoading(false); }
  }, [sessionNum, loadTranscript]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);
  useEffect(() => { if (autoScroll && activeRef.current) activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [currentTime, autoScroll]);

  const getActiveIdx = useCallback((utts: TranscriptUtterance[]) => {
    if (currentTime == null) return -1;
    return utts.findIndex(u => u.start <= currentTime && currentTime <= u.end);
  }, [currentTime]);

  const filtered = transcript?.utterances.filter(u => {
    if (filter !== "all" && u.speaker !== filter) return false;
    if (searchQuery && !u.text.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  }) ?? [];

  const activeIdx = transcript ? getActiveIdx(transcript.utterances) : -1;

  // No transcript, show generate button
  if (!hasTranscript && !loading && !transcript) {
    return (
      <div className="rounded-lg p-4 flex items-center justify-between" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)" }}>
        <div>
          <p className="text-sm font-medium" style={{ color: "#d4d4d8" }}>Transcript</p>
          <p className="text-xs" style={{ color: "#52525b" }}>Whisper large-v3</p>
        </div>
        <button onClick={handleGenerate} className="btn-primary" id="transcript-generate-btn">Generate</button>
      </div>
    );
  }

  if (loading && !transcript) {
    return (
      <div className="rounded-lg p-4" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)" }}>
        <p className="text-sm" style={{ color: "#a1a1aa" }}>Generating transcript...</p>
        <div className="mt-3 loading-bar rounded-full"><div className="loading-bar-fill rounded-full" /></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg p-4" style={{ background: "rgba(248,113,113,0.04)", border: "1px solid rgba(248,113,113,0.15)" }}>
        <p className="text-sm" style={{ color: "#f87171" }}>Transcription failed</p>
        <p className="text-xs mt-1" style={{ color: "#71717a" }}>{error}</p>
        <button onClick={handleGenerate} className="btn-ghost mt-2" id="transcript-retry-btn">Retry</button>
      </div>
    );
  }

  if (!transcript) return null;

  const counts = transcript.utterances.reduce((a, u) => { a[u.speaker] = (a[u.speaker] ?? 0) + 1; return a; }, {} as Record<string, number>);

  return (
    <div className="rounded-lg overflow-hidden" style={{ border: "1px solid var(--border-subtle)" }}>
      {/* Header */}
      <div className="px-4 py-3" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
        <div className="flex items-center justify-between mb-2.5">
          <p className="text-xs font-medium" style={{ color: "#a1a1aa" }}>
            {transcript.utterances.length} utterances · {Math.round(transcript.duration_s)}s
          </p>
          <div className="flex items-center gap-1.5">
            {(["chat", "raw"] as const).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className="px-2 py-1 text-[11px] rounded transition-colors capitalize"
                style={{ background: view === v ? "rgba(255,255,255,0.08)" : "transparent", color: view === v ? "#d4d4d8" : "#52525b" }}
                id={`transcript-view-${v}`}
              >{v}</button>
            ))}
            <button onClick={() => setAutoScroll(p => !p)} className="px-2 py-1 text-[11px] rounded transition-colors" style={{ color: autoScroll ? "#818cf8" : "#52525b" }} id="transcript-autoscroll-btn">
              {autoScroll ? "Auto" : "Manual"}
            </button>
            <button
              onClick={() => {
                const text = transcript.utterances.map(u => `[${u.speaker}] ${fmt(u.start)} - ${fmt(u.end)}\n${u.text}`).join("\n\n");
                const blob = new Blob([text], { type: "text/plain" });
                const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
                a.download = `transcript_session_${sessionNum}.txt`; a.click();
              }}
              className="px-2 py-1 text-[11px] rounded transition-colors"
              style={{ color: "#52525b" }}
              id="transcript-download-btn"
            >Export</button>
          </div>
        </div>

        {/* Filters + search */}
        <div className="flex items-center gap-2">
          {(["all", "A", "B"] as const).map(f => {
            const spk = f !== "all" ? SPK[f] : null;
            const count = f === "all" ? transcript.utterances.length : (counts[f] ?? 0);
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className="flex items-center gap-1 px-2 py-1 rounded text-[11px] transition-colors"
                id={`transcript-filter-${f}`}
                style={{
                  background: filter === f ? "rgba(255,255,255,0.06)" : "transparent",
                  color: filter === f ? (spk ? spk.color : "#d4d4d8") : "#52525b",
                }}
              >
                {spk && <span className="w-1.5 h-1.5 rounded-full" style={{ background: spk.color }} />}
                {f === "all" ? "All" : spk!.label} ({count})
              </button>
            );
          })}
          <input
            id="transcript-search"
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="ml-auto px-2 py-1 text-[11px] rounded-md w-36"
            style={{ background: "rgba(255,255,255,0.04)", border: "1px solid var(--border-subtle)", color: "#d4d4d8", outline: "none" }}
          />
        </div>
      </div>

      {/* Content */}
      <div ref={containerRef} className="overflow-y-auto" style={{ maxHeight: 420 }}>
        {view === "chat" ? (
          <div className="p-3 space-y-1.5">
            {filtered.length === 0 ? (
              <p className="text-center text-xs py-6" style={{ color: "#52525b" }}>No results</p>
            ) : (
              filtered.map((u, i) => {
                const s = SPK[u.speaker as "A" | "B"] ?? SPK.A;
                const isActive = transcript.utterances.indexOf(u) === activeIdx;
                return (
                  <div
                    key={i}
                    ref={isActive ? activeRef : undefined}
                    className={`transcript-entry p-2.5 rounded-lg cursor-pointer transition-colors ${s.bubble}`}
                    style={isActive ? { background: u.speaker === "A" ? "rgba(129,140,248,0.08)" : "rgba(52,211,153,0.08)" } : {}}
                    onClick={() => onSeek?.(u.start)}
                    id={`utterance-${i}`}
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[11px] font-medium" style={{ color: s.color }}>{s.label}</span>
                      <span className="text-[10px] tabular-nums" style={{ color: "#3f3f46" }}>{fmt(u.start)}</span>
                    </div>
                    <p className="text-xs leading-relaxed" style={{ color: "#a1a1aa" }}>
                      {searchQuery ? <Highlight text={u.text} q={searchQuery} /> : u.text}
                    </p>
                  </div>
                );
              })
            )}
          </div>
        ) : (
          <div className="p-3">
            <pre className="text-[11px] leading-6 font-mono whitespace-pre-wrap" style={{ color: "#71717a" }}>
              {filtered.map(u => `[${u.speaker}] ${fmt(u.start).padEnd(7)} ${u.text}`).join("\n")}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

function Highlight({ text, q }: { text: string; q: string }) {
  if (!q) return <>{text}</>;
  const parts = text.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, "gi"));
  return <>{parts.map((p, i) => p.toLowerCase() === q.toLowerCase() ? <mark key={i} style={{ background: "rgba(129,140,248,0.25)", color: "inherit", borderRadius: 2, padding: "0 1px" }}>{p}</mark> : <span key={i}>{p}</span>)}</>;
}
