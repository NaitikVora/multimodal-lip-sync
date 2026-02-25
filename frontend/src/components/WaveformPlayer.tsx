import { useEffect, useRef, useState, useCallback } from "react";
import WaveSurfer from "wavesurfer.js";

interface Props {
  label: string;
  speaker: "A" | "B";
  url: string;
  filename: string;
  sizeKb: number;
  onTimeUpdate?: (time: number) => void;
}

const CFG = {
  A: { wave: "rgba(129,140,248,0.4)", progress: "#818cf8", label: "#818cf8" },
  B: { wave: "rgba(52,211,153,0.4)", progress: "#34d399", label: "#34d399" },
};

export default function WaveformPlayer({ label, speaker, url, filename, sizeKb, onTimeUpdate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [ready, setReady] = useState(false);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);

  const c = CFG[speaker];

  useEffect(() => {
    if (!containerRef.current) return;
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: c.wave,
      progressColor: c.progress,
      cursorColor: "rgba(255,255,255,0.2)",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      height: 48,
      normalize: true,
      backend: "WebAudio",
    });
    ws.load(url);
    ws.on("ready", () => { setDuration(ws.getDuration()); setReady(true); });
    ws.on("audioprocess", () => { const t = ws.getCurrentTime(); setCurrentTime(t); onTimeUpdate?.(t); });
    ws.on("seeking", () => { const t = ws.getCurrentTime(); setCurrentTime(t); onTimeUpdate?.(t); });
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    wsRef.current = ws;
    return () => ws.destroy();
  }, [url, c.wave, c.progress, onTimeUpdate]);

  const togglePlay = useCallback(() => wsRef.current?.playPause(), []);
  const toggleMute = useCallback(() => {
    const ws = wsRef.current;
    if (!ws) return;
    const next = !muted;
    ws.setVolume(next ? 0 : volume);
    setMuted(next);
  }, [muted, volume]);
  const handleVol = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setVolume(v); setMuted(v === 0);
    wsRef.current?.setVolume(v);
  }, []);

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className="rounded-lg overflow-hidden" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)" }}>
      {/* Label row */}
      <div className="flex items-center justify-between px-3 pt-3 pb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium" style={{ color: c.label }}>{label}</span>
          <span className="text-[11px]" style={{ color: "#3f3f46" }}>
            {ready ? fmt(duration) : "..."} · {sizeKb} KB
          </span>
        </div>
        <a
          href={url} download={filename}
          className="text-xs px-2 py-1 rounded transition-colors"
          style={{ color: "#52525b" }}
          title="Download"
          id={`download-${speaker.toLowerCase()}-btn`}
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </a>
      </div>

      {/* Waveform */}
      <div className="px-3">
        <div ref={containerRef} />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2.5 px-3 pb-3 pt-2">
        <button
          onClick={togglePlay}
          disabled={!ready}
          className="w-7 h-7 rounded-md flex items-center justify-center transition-colors disabled:opacity-20"
          style={{ background: "rgba(255,255,255,0.06)" }}
          id={`play-${speaker.toLowerCase()}-btn`}
        >
          {playing ? (
            <svg className="w-3 h-3" style={{ color: c.label }} fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          ) : (
            <svg className="w-3 h-3 ml-0.5" style={{ color: c.label }} fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {ready && (
          <span className="text-[11px] tabular-nums font-mono" style={{ color: "#52525b" }}>
            {fmt(currentTime)} / {fmt(duration)}
          </span>
        )}

        <div className="flex items-center gap-1.5 ml-auto">
          <button onClick={toggleMute} style={{ color: "#3f3f46" }} id={`mute-${speaker.toLowerCase()}-btn`}>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {muted || volume === 0 ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072M12 6v12M9 9H5a2 2 0 00-2 2v2a2 2 0 002 2h4l5 5V4L9 9z" />
              )}
            </svg>
          </button>
          <input type="range" min={0} max={1} step={0.05} value={volume} onChange={handleVol} className="w-16" id={`vol-${speaker.toLowerCase()}`} />
        </div>
      </div>
    </div>
  );
}
