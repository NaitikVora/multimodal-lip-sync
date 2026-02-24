import { useCallback, useState, useRef } from "react";
import { uploadFiles } from "../api";

interface Props {
  onUploadComplete: () => void;
}

export default function UploadZone({ onUploadComplete }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [sessionName, setSessionName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const name = sessionName.trim() || `Session_${String(Date.now()).slice(-6)}`;
      setUploading(true);
      try {
        await uploadFiles(name, files);
        setSessionName("");
        onUploadComplete();
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
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
        dragging
          ? "border-blue-500/50 bg-blue-500/5"
          : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={fileRef}
        type="file"
        accept=".wav"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) handleFiles(e.target.files);
        }}
      />

      <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-zinc-800 flex items-center justify-center">
        <svg className="w-5 h-5 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      </div>

      <p className="text-sm text-zinc-400 mb-4">
        {uploading ? (
          <span className="text-blue-400">Uploading\u2026</span>
        ) : (
          "Drag & drop WAV files here, or click to browse"
        )}
      </p>

      <div className="flex items-center justify-center gap-3">
        <input
          type="text"
          placeholder="Session name (optional)"
          value={sessionName}
          onChange={(e) => setSessionName(e.target.value)}
          className="px-3 py-1.5 bg-zinc-800 border border-zinc-700 text-zinc-200 placeholder-zinc-600 rounded-lg text-sm w-52 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500/40"
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          className="px-4 py-1.5 bg-zinc-800 text-zinc-200 text-sm font-medium rounded-lg border border-zinc-700 hover:bg-zinc-700 disabled:opacity-40 transition-colors cursor-pointer"
        >
          Browse Files
        </button>
      </div>
    </div>
  );
}
