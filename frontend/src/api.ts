export interface Session {
  num: number;
  folder: string;
  files: string[];
  num_files: number;
  total_bytes: number;
  total_mb: number;
  has_results: boolean;
  has_transcript: boolean;
  active_job: string | null;
}

export interface TrackInfo {
  filename: string;
  size_kb: number;
  url: string;
}

export interface PipelineInfo {
  id: string;
  label: string;
  description: string;
  tracks: Record<string, TrackInfo>;
  elapsed_s: number;
  segments: number;
  overlap_pct: number;
  model?: string;
  error?: string;
}

export interface SessionResults {
  session_num: number;
  pipelines: Record<string, PipelineInfo>;
}

export interface JobState {
  job_id: string;
  session_num: number;
  status: "pending" | "processing" | "done" | "error";
  stage: string;
  audio_info?: { num_files: number; duration_s: number; size_mb: number };
  pipeline_metadata?: Record<string, { elapsed_s: number; segments: number; overlap_pct: number; model?: string; error?: string }>;
  error?: string;
  elapsed_s?: number;
  created_at: number;
  started_at?: number;
  finished_at?: number;
}

export interface TranscriptUtterance {
  speaker: "A" | "B";
  start: number;
  end: number;
  text: string;
}

export interface Transcript {
  utterances: TranscriptUtterance[];
  language: string;
  duration_s: number;
  num_segments: number;
  error?: string;
}

export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error("Failed to fetch sessions");
  const data = await res.json();
  return data.sessions;
}

export async function fetchHealth(): Promise<{ status: string; detail: string; pipelines?: string[] }> {
  const res = await fetch("/api/health");
  return res.json();
}

export async function startProcessing(sessionNum: number, force = false): Promise<{ job_id: string }> {
  const url = `/api/process/${sessionNum}${force ? "?force=true" : ""}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error("Failed to start processing");
  return res.json();
}

export async function reprocessSession(sessionNum: number): Promise<{ job_id: string }> {
  return startProcessing(sessionNum, true);
}

export async function fetchResults(sessionNum: number): Promise<SessionResults> {
  const res = await fetch(`/api/results/${sessionNum}`);
  if (!res.ok) throw new Error("No results available");
  return res.json();
}

export async function fetchTranscript(sessionNum: number): Promise<Transcript> {
  const res = await fetch(`/api/transcript/${sessionNum}`);
  if (!res.ok) throw new Error("No transcript available");
  return res.json();
}

export async function startTranscription(sessionNum: number): Promise<{ status: string }> {
  const res = await fetch(`/api/transcribe/${sessionNum}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to start transcription");
  return res.json();
}

export async function uploadFiles(sessionName: string, files: FileList | File[]): Promise<void> {
  const form = new FormData();
  form.append("session_name", sessionName);
  for (const f of files) {
    form.append("files", f);
  }
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error("Upload failed");
}

export function subscribeToJob(
  jobId: string,
  onUpdate: (job: JobState) => void,
  onDone: (job: JobState) => void,
): () => void {
  const evtSource = new EventSource(`/api/process/${jobId}/status`);

  evtSource.onmessage = (event) => {
    const job: JobState = JSON.parse(event.data);
    onUpdate(job);
    if (job.status === "done" || job.status === "error") {
      evtSource.close();
      onDone(job);
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
  };

  return () => evtSource.close();
}
