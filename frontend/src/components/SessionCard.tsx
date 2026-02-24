import { useEffect, useState, useCallback } from "react";
import type { Session, JobState, SessionResults } from "../api";
import { startProcessing, subscribeToJob, fetchResults } from "../api";
import WaveformPlayer from "./WaveformPlayer";
import PipelineTabs from "./PipelineTabs";

interface Props {
  session: Session;
  onProcessingDone: () => void;
}

type CardStatus = "idle" | "processing" | "done" | "error";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  loaded: "Audio loaded",
  energy_processing: "Running energy-based separation\u2026",
  energy_done: "Energy pipeline done",
  gpu_processing: "Running ML pipelines on GPU\u2026",
  saving: "Saving results",
  complete: "Complete",
  failed: "Failed",
};

const WAVEFORM_COLORS: Record<string, { wave: string; progress: string }> = {
  energy: { wave: "#a1a1aa", progress: "#71717a" },
  channel: { wave: "#3b82f6", progress: "#1d4ed8" },
  sepformer: { wave: "#a855f7", progress: "#7c3aed" },
};

export default function SessionCard({ session, onProcessingDone }: Props) {
  const [status, setStatus] = useState<CardStatus>(
    session.has_results ? "done" : "idle",
  );
  const [job, setJob] = useState<JobState | null>(null);
  const [results, setResults] = useState<SessionResults | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [activePipeline, setActivePipeline] = useState("channel");

  useEffect(() => {
    if (session.has_results && !results) {
      fetchResults(session.num).then((r) => {
        setResults(r);
        const ids = Object.keys(r.pipelines);
        if (ids.includes("channel")) setActivePipeline("channel");
        else if (ids.length > 0) setActivePipeline(ids[0]);
      }).catch(() => {});
    }
  }, [session.has_results, session.num, results]);

  useEffect(() => {
    if (!session.active_job) return;
    setStatus("processing");

    const unsub = subscribeToJob(
      session.active_job,
      (j) => {
        setJob(j);
        if (j.elapsed_s) setElapsed(j.elapsed_s);
      },
      (j) => {
        setJob(j);
        if (j.status === "done") {
          setStatus("done");
          fetchResults(session.num).then(setResults).catch(() => {});
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

  const handleProcess = useCallback(async () => {
    try {
      setStatus("processing");
      setElapsed(0);
      const { job_id } = await startProcessing(session.num);

      subscribeToJob(
        job_id,
        (j) => setJob(j),
        (j) => {
          setJob(j);
          if (j.status === "done") {
            setStatus("done");
            if (j.elapsed_s) setElapsed(j.elapsed_s);
            fetchResults(session.num).then(setResults).catch(() => {});
            onProcessingDone();
          } else {
            setStatus("error");
            if (j.elapsed_s) setElapsed(j.elapsed_s);
          }
        },
      );
    } catch {
      setStatus("error");
    }
  }, [session.num, onProcessingDone]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  const activeColors = WAVEFORM_COLORS[activePipeline] ?? WAVEFORM_COLORS.channel;
  const activePipelineData = results?.pipelines[activePipeline];

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-zinc-100 text-sm">
            Session {session.num}
          </h3>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-zinc-500">
              {session.num_files} file{session.num_files > 1 ? "s" : ""}
            </span>
            <span className="text-zinc-700">&middot;</span>
            <span className="text-xs text-zinc-500">{session.total_mb} MB</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {status === "idle" && (
            <button
              onClick={handleProcess}
              className="px-4 py-1.5 bg-blue-600 text-white text-xs font-medium rounded-lg hover:bg-blue-500 transition-colors cursor-pointer"
            >
              Process All Pipelines
            </button>
          )}

          {status === "processing" && (
            <div className="flex items-center gap-2 text-xs text-zinc-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
              </span>
              <span>{STAGE_LABELS[job?.stage ?? ""] ?? "Processing\u2026"}</span>
              <span className="text-zinc-600 tabular-nums">{formatTime(elapsed)}</span>
            </div>
          )}

          {status === "done" && elapsed > 0 && (
            <span className="text-xs text-zinc-600">
              Total: {formatTime(elapsed)}
            </span>
          )}

          {status === "done" && (
            <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400">
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              3 pipelines
            </span>
          )}

          {status === "error" && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-red-400">
                Failed{job?.error ? `: ${job.error.slice(0, 60)}` : ""}
              </span>
              <button
                onClick={handleProcess}
                className="px-3 py-1 text-xs border border-red-800 text-red-400 rounded-lg hover:bg-red-900/30 transition-colors cursor-pointer"
              >
                Retry
              </button>
            </div>
          )}
        </div>
      </div>

      {/* File pills */}
      <div className="px-5 pb-3 flex flex-wrap gap-1.5">
        {session.files.map((f) => (
          <span
            key={f}
            className="text-[11px] px-2 py-0.5 bg-zinc-800 text-zinc-500 rounded-md"
          >
            {f}
          </span>
        ))}
      </div>

      {/* Processing progress bar */}
      {status === "processing" && (
        <div className="h-0.5 bg-zinc-800 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-blue-600 to-blue-400 animate-shimmer w-full" />
        </div>
      )}

      {/* Results: tabbed pipeline comparison */}
      {status === "done" && results && Object.keys(results.pipelines).length > 0 && (
        <div className="border-t border-zinc-800">
          <PipelineTabs
            pipelines={results.pipelines}
            activeId={activePipeline}
            onSelect={setActivePipeline}
          />

          {activePipelineData && (
            <div className="px-5 py-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {activePipelineData.tracks.speaker_a && (
                  <WaveformPlayer
                    label="Speaker A"
                    url={activePipelineData.tracks.speaker_a.url}
                    filename={activePipelineData.tracks.speaker_a.filename}
                    sizeKb={activePipelineData.tracks.speaker_a.size_kb}
                    color={activeColors.wave}
                    progressColor={activeColors.progress}
                  />
                )}
                {activePipelineData.tracks.speaker_b && (
                  <WaveformPlayer
                    label="Speaker B"
                    url={activePipelineData.tracks.speaker_b.url}
                    filename={activePipelineData.tracks.speaker_b.filename}
                    sizeKb={activePipelineData.tracks.speaker_b.size_kb}
                    color="#10b981"
                    progressColor="#059669"
                  />
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
