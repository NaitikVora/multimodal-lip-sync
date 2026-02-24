interface Props {
  label: string;
  url: string;
  filename: string;
  sizeKb: number;
}

export default function AudioPlayer({ label, url, filename, sizeKb }: Props) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-neutral-700 capitalize">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-400">{sizeKb} KB</span>
          <a
            href={url}
            download={filename}
            className="text-xs text-blue-600 hover:text-blue-800 underline"
          >
            Download
          </a>
        </div>
      </div>
      <audio controls preload="none" src={url} className="w-full h-9" />
    </div>
  );
}
