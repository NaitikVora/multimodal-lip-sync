import { useEffect, useState, useCallback } from "react";
import type { Session, JobState, SessionResults } from "../api";
import { startProcessing, reprocessSession, subscribeToJob, fetchResults } from "../api";
import WaveformPlayer from "./WaveformPlayer";
import PipelineTabs from "./PipelineTabs";
import TranscriptPanel from "./TranscriptPanel";
import ModelComparison from "./ModelComparison";

interface Props {
  session: Session;
  onProcessingDone: () => void;
}

type CardStatus = "idle" | "processing" | "done" | "error";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  loaded: "Loaded",
  energy_processing: "Energy pipeline",
  energy_done: "Launching GPU",
  gpu_processing: "ML pipelines",
  transcribing: "Transcribing",
  saving: "Saving",
  complete: "Complete",
  failed: "Failed",
};

const PIPELINE_ORDER = ["energy", "channel", "sepformer", "nemo", "diarizen"];
const STAGE_PROGRESS: Record<string, number> = {
  queued: 2, loaded: 8, energy_processing: 12, energy_done: 20,
  gpu_processing: 60, transcribing: 88, saving: 96, complete: 100,
};

export default function SessionCard({ session, onProcessingDone }: Props) {
  const [status, setStatus] = useState<CardStatus>(session.has_results ? "done" : "idle");
  const [job, setJob] = useState<JobState | null>(null);
  const [results, setResults] = useState<SessionResults | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [activePipeline, setActivePipeline] = useState("channel");
  const [activeTab, setActiveTab] = useState<"audio" | "transcript" | "compare">("audio");
  const [currentPlayTime, setCurrentPlayTime] = useState<number>(0);
  const [expanded, setExpanded] = useState(session.has_results);

  useEffect(() => {
    if (session.has_results && !results) {
      fetchResults(session.num).then(r => {
        setResults(r);
        const ids = Object.keys(r.pipelines);
        const prefOrder = PIPELINE_ORDER.filter(p => ids.includes(p));
        if (prefOrder.length > 0) setActivePipeline(prefOrder[1] ?? prefOrder[0]);
      }).catch(() => { });
    }
  }, [session.has_results, session.num, results]);

  useEffect(() => {
    if (!session.active_job) return;
    setStatus("processing");
    setExpanded(true);
    const unsub = subscribeToJob(
      session.active_job,
      j => { setJob(j); if (j.elapsed_s) setElapsed(j.elapsed_s); },
      j => {
        setJob(j);
        if (j.status === "done") {
          setStatus("done");
          fetchResults(session.num).then(setResults).catch(() => { });
          onProcessingDone();
        } else {
          setStatus("error");
        }
      },
    );
    return unsub;
  }, [session.active_job, session.num, onProcessingDone]);

  useEffect(() => {
    if (status !== "processing") return;
    const start = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - start) / 1000), 500);
    return () => clearInterval(id);
  }, [status]);

  const _launchJob = useCallback(async (force: boolean) => {
    try {
      setStatus("processing");
      setElapsed(0);
      setExpanded(true);
      const { job_id } = force
        ? await reprocessSession(session.num)
        : await startProcessing(session.num);
      subscribeToJob(
        job_id,
        j => setJob(j),
        j => {
          setJob(j);
          if (j.status === "done") {
            setStatus("done");
            if (j.elapsed_s) setElapsed(j.elapsed_s);
            fetchResults(session.num).then(setResults).catch(() => { });
            onProcessingDone();
          } else {
            setStatus("error");
            if (j.elapsed_s) setElapsed(j.elapsed_s);
          }
        },
      );
    } catch { setStatus("error"); }
  }, [session.num, onProcessingDone]);

  const handleProcess = useCallback(() => _launchJob(false), [_launchJob]);
  const handleReprocess = useCallback(() => _launchJob(true), [_launchJob]);

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  const stageProgress = STAGE_PROGRESS[job?.stage ?? ""] ?? 0;
  const activePipelineData = results?.pipelines[activePipeline];

  return (
    <div
      className="rounded-xl overflow-hidden animate-slide-up"
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-subtle)",
      }}
    >
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center justify-between cursor-pointer highlight-on-hover"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-medium text-sm" style={{ color: "#e4e4e7" }}>
                Session {session.num}
              </h3>
              {status === "done" && (
                <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "rgba(52,211,153,0.1)", color: "#34d399" }}>
                  Done
                </span>
              )}
              {session.has_transcript && (
                <span className="text-[11px] px-1.5 py-0.5 rounded" style={{ background: "rgba(129,140,248,0.1)", color: "#818cf8" }}>
                  Transcript
                </span>
              )}
            </div>
            <p className="text-xs mt-0.5" style={{ color: "#52525b" }}>
              {session.num_files} file{session.num_files > 1 ? "s" : ""} · {session.total_mb} MB
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
          {status === "idle" && (
            <button onClick={handleProcess} className="btn-primary" id={`process-btn-${session.num}`}>
              Run
            </button>
          )}
          {status === "done" && (
            <button onClick={handleReprocess} className="btn-ghost" id={`reprocess-btn-${session.num}`}>
              Reprocess
            </button>
          )}
          {status === "processing" && (
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "#fbbf24" }} />
              <span className="text-xs" style={{ color: "#a1a1aa" }}>
                {STAGE_LABELS[job?.stage ?? ""] ?? "Processing"} · {fmt(elapsed)}
              </span>
            </div>
          )}
          {status === "error" && (
            <div className="flex items-center gap-2">
              <span className="text-xs" style={{ color: "#f87171" }}>
                {job?.error ? job.error.slice(0, 40) : "Failed"}
              </span>
              <button onClick={handleProcess} className="btn-ghost" id={`retry-btn-${session.num}`}>
                Retry
              </button>
            </div>
          )}

          <div style={{ color: "#3f3f46", transform: expanded ? "rotate(180deg)" : "", transition: "transform 0.15s" }}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {/* Progress */}
      {status === "processing" && (
        <div className="h-px" style={{ background: "rgba(255,255,255,0.04)" }}>
          <div
            className="h-full transition-all duration-1000"
            style={{ width: `${stageProgress}%`, background: "#52525b" }}
          />
        </div>
      )}

      {/* Expanded content */}
      {expanded && status === "done" && results && Object.keys(results.pipelines).length > 0 && (
        <div style={{ borderTop: "1px solid var(--border-subtle)" }}>
          {/* Tab bar */}
          <div className="flex" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
            {(["audio", "transcript", "compare"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="px-4 py-2.5 text-xs font-medium transition-colors capitalize"
                id={`tab-${tab}-${session.num}`}
                style={{
                  color: activeTab === tab ? "#e4e4e7" : "#52525b",
                  borderBottom: activeTab === tab ? "1px solid #a1a1aa" : "1px solid transparent",
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {activeTab === "audio" && (
            <div>
              <PipelineTabs pipelines={results.pipelines} activeId={activePipeline} onSelect={setActivePipeline} />
              {activePipelineData && (
                <div className="p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {activePipelineData.tracks.speaker_a && (
                      <WaveformPlayer
                        label="Speaker A"
                        speaker="A"
                        url={activePipelineData.tracks.speaker_a.url}
                        filename={activePipelineData.tracks.speaker_a.filename}
                        sizeKb={activePipelineData.tracks.speaker_a.size_kb}
                        onTimeUpdate={setCurrentPlayTime}
                      />
                    )}
                    {activePipelineData.tracks.speaker_b && (
                      <WaveformPlayer
                        label="Speaker B"
                        speaker="B"
                        url={activePipelineData.tracks.speaker_b.url}
                        filename={activePipelineData.tracks.speaker_b.filename}
                        sizeKb={activePipelineData.tracks.speaker_b.size_kb}
                        onTimeUpdate={setCurrentPlayTime}
                      />
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "compare" && (
            <div className="p-4">
              <ModelComparison pipelines={results.pipelines} />
            </div>
          )}

          {activeTab === "transcript" && (
            <div className="p-4">
              <TranscriptPanel
                sessionNum={session.num}
                hasTranscript={session.has_transcript}
                currentTime={currentPlayTime}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
