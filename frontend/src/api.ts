export interface Session {
  num: number;
  folder: string;
  files: string[];
  num_files: number;
  total_bytes: number;
  total_mb: number;
  has_results: boolean;
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
  tracks: Record<string, TrackInfo>;
  elapsed_s: number;
  segments: number;
  overlap_pct: number;
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
  pipeline_metadata?: Record<string, { elapsed_s: number; segments: number; overlap_pct: number }>;
  error?: string;
  elapsed_s?: number;
  created_at: number;
  started_at?: number;
  finished_at?: number;
}

export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error("Failed to fetch sessions");
  const data = await res.json();
  return data.sessions;
}

export async function fetchHealth(): Promise<{ status: string; detail: string }> {
  const res = await fetch("/api/health");
  return res.json();
}

export async function startProcessing(sessionNum: number): Promise<{ job_id: string }> {
  const res = await fetch(`/api/process/${sessionNum}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to start processing");
  return res.json();
}

export async function fetchResults(sessionNum: number): Promise<SessionResults> {
  const res = await fetch(`/api/results/${sessionNum}`);
  if (!res.ok) throw new Error("No results available");
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
