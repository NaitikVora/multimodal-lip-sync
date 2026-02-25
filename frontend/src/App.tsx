import { useEffect, useState, useCallback } from "react";
import type { Session } from "./api";
import { fetchSessions, fetchHealth } from "./api";
import SessionCard from "./components/SessionCard";
import UploadZone from "./components/UploadZone";

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [health, setHealth] = useState<"ok" | "error" | "loading">("loading");
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);

  const loadSessions = useCallback(async () => {
    try {
      const s = await fetchSessions();
      setSessions(s);
      if (s.length === 0) setShowUpload(true);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
    fetchHealth()
      .then(h => setHealth(h.status === "ok" ? "ok" : "error"))
      .catch(() => setHealth("error"));
  }, [loadSessions]);

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-base)" }}>
      {/* Header */}
      <header
        className="sticky top-0 z-50"
        style={{
          background: "rgba(9,9,11,0.9)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <div className="max-w-3xl mx-auto px-5 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <h1 className="text-sm font-semibold tracking-tight" style={{ color: "#e4e4e7" }}>
              SpeakerLens
            </h1>
            <span
              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{
                background: health === "ok" ? "#34d399" : health === "error" ? "#f87171" : "#52525b",
              }}
              title={`Modal ${health}`}
            />
          </div>

          <div className="flex items-center gap-2">
            {sessions.length > 0 && (
              <button onClick={loadSessions} className="btn-ghost" id="refresh-sessions-btn">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            )}
            <button
              onClick={() => setShowUpload(prev => !prev)}
              className="btn-ghost"
              id="toggle-upload-btn"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New
            </button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-3xl mx-auto px-5 py-6">
        {showUpload && (
          <section className="mb-6 animate-fade-in">
            <UploadZone onUploadComplete={() => { loadSessions(); setShowUpload(false); }} />
          </section>
        )}

        {loading && (
          <div className="space-y-3">
            {[1, 2].map(n => (
              <div key={n} className="h-16 rounded-xl" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }} />
            ))}
          </div>
        )}

        {!loading && sessions.length === 0 && !showUpload && (
          <div className="text-center py-16 animate-fade-in">
            <p className="text-sm mb-1" style={{ color: "#52525b" }}>No sessions yet</p>
            <p className="text-xs mb-4" style={{ color: "#3f3f46" }}>Upload WAV recordings to get started</p>
            <button onClick={() => setShowUpload(true)} className="btn-primary" id="get-started-btn">
              Upload Audio
            </button>
          </div>
        )}

        <div className="space-y-3">
          {sessions.map(s => (
            <SessionCard key={s.num} session={s} onProcessingDone={loadSessions} />
          ))}
        </div>
      </main>
    </div>
  );
}
