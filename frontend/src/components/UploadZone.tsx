import { useCallback, useState, useRef } from "react";
import { uploadFiles } from "../api";

interface Props {
  onUploadComplete: () => void;
}

export default function UploadZone({ onUploadComplete }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [sessionName, setSessionName] = useState("");
  const [done, setDone] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const name = sessionName.trim() || `Session_${String(Date.now()).slice(-6)}`;
      setUploading(true);
      try {
        await uploadFiles(name, files);
        setDone(true);
        setSessionName("");
        setTimeout(() => { setDone(false); onUploadComplete(); }, 1200);
      } catch (e) {
        alert(`Upload failed: ${e}`);
      } finally {
        setUploading(false);
      }
    },
    [sessionName, onUploadComplete],
  );

  return (
    <div
      className="rounded-xl p-5 text-center transition-colors"
      style={{
        background: dragging ? "rgba(255,255,255,0.03)" : "var(--bg-card)",
        border: `1px dashed ${dragging ? "rgba(255,255,255,0.15)" : "var(--border-base)"}`,
      }}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files); }}
    >
      <input
        ref={fileRef} type="file" accept=".wav,.WAV" multiple className="hidden"
        onChange={e => { if (e.target.files?.length) handleFiles(e.target.files); }}
        id="file-upload-input"
      />

      {done ? (
        <p className="text-sm" style={{ color: "#34d399" }}>Uploaded</p>
      ) : uploading ? (
        <div>
          <p className="text-sm" style={{ color: "#a1a1aa" }}>Uploading...</p>
          <div className="mt-3 mx-auto w-32 loading-bar rounded-full"><div className="loading-bar-fill rounded-full" /></div>
        </div>
      ) : (
        <>
          <p className="text-sm mb-3" style={{ color: "#71717a" }}>
            {dragging ? "Drop files here" : "Drop WAV files or browse"}
          </p>
          <div className="flex items-center justify-center gap-2 flex-wrap">
            <input
              type="text"
              placeholder="Session name"
              value={sessionName}
              onChange={e => setSessionName(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg w-44"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid var(--border-subtle)",
                color: "#d4d4d8",
                outline: "none",
              }}
              id="session-name-input"
              onKeyDown={e => e.key === "Enter" && fileRef.current?.click()}
            />
            <button onClick={() => fileRef.current?.click()} disabled={uploading} className="btn-ghost" id="browse-files-btn">
              Browse
            </button>
          </div>
        </>
      )}
    </div>
  );
}
