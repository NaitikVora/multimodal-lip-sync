import { useEffect, useState, useCallback } from "react";
import type { Session } from "./api";
import { fetchSessions, fetchHealth } from "./api";
import SessionCard from "./components/SessionCard";
import UploadZone from "./components/UploadZone";

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [health, setHealth] = useState<"ok" | "error" | "loading">("loading");
  const [loading, setLoading] = useState(true);

  const loadSessions = useCallback(async () => {
    try {
      const s = await fetchSessions();
      setSessions(s);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
    fetchHealth()
      .then((h) => setHealth(h.status === "ok" ? "ok" : "error"))
      .catch(() => setHealth("error"));
  }, [loadSessions]);

  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Accent bar */}
      <div className="h-0.5 bg-gradient-to-r from-blue-600 via-blue-500 to-emerald-500" />

      {/* Header */}
      <header className="border-b border-zinc-800/80">
        <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100 tracking-tight">
              Speaker Separation
            </h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              Audio &rarr; separated Speaker A &amp; Speaker B tracks
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <span
              className={`w-2 h-2 rounded-full ${
                health === "ok"
                  ? "bg-emerald-500"
                  : health === "error"
                    ? "bg-red-500"
                    : "bg-zinc-600 animate-pulse"
              }`}
            />
            Modal{" "}
            {health === "ok"
              ? "connected"
              : health === "error"
                ? "unavailable"
                : "checking\u2026"}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        <section className="mb-10">
          <UploadZone onUploadComplete={loadSessions} />
        </section>

        <section>
          <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-widest mb-4">
            Sessions ({sessions.length})
          </h2>

          {loading && (
            <p className="text-sm text-zinc-600 py-12 text-center">
              Loading sessions...
            </p>
          )}

          {!loading && sessions.length === 0 && (
            <div className="text-center py-16 text-zinc-600">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-zinc-800/80 flex items-center justify-center">
                <svg className="w-6 h-6 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m-4-4h8m-4-8a3 3 0 00-3-3v0a3 3 0 00-3 3v4a3 3 0 003 3h0a3 3 0 003-3V7z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-zinc-400 mb-1">No sessions found</p>
              <p className="text-xs text-zinc-600">
                Upload WAV files or add them to the Audio_Files/ directory
              </p>
            </div>
          )}

          <div className="space-y-4">
            {sessions.map((s) => (
              <SessionCard
                key={s.num}
                session={s}
                onProcessingDone={loadSessions}
              />
            ))}
          </div>
        </section>
      </main>

      <footer className="text-center text-[11px] text-zinc-700 py-8">
        Powered by Modal GPU Cloud &middot; pyannote &middot; SpeechBrain SepFormer
      </footer>
    </div>
  );
}
