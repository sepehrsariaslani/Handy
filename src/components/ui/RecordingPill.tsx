import { useEffect, useState } from "react";
import { Square } from "lucide-react";

type Props = {
  level: number;
  onStop: () => void;
  disabled?: boolean;
  label?: string;
};

/**
 * Shared recording control, visually matched to Handy's compact dictation
 * overlay: pink dot, nine responsive bars, elapsed time and a stop button.
 * Audio level comes from the same recorder as the actual captured audio.
 */
export function RecordingPill({
  level,
  onStop,
  disabled = false,
  label = "Recording",
}: Props) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const timer = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - start) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, []);
  const time = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}`;
  const power = Math.max(0, Math.min(1, level));
  const bands = [0.55, 0.88, 0.69, 1, 0.8, 0.97, 0.72, 0.92, 0.62];

  return (
    <div
      className="inline-flex max-w-full items-center gap-4 rounded-full border border-line bg-background px-4 py-2 min-h-14"
      role="status"
      aria-label={label}
    >
      <span className="h-3.5 w-3.5 shrink-0 rounded-full bg-logo-primary" />
      <span className="flex h-6 items-center gap-1" aria-hidden="true">
        {bands.map((weight, i) => (
          <span
            key={i}
            className="w-[5px] rounded-full bg-logo-primary transition-[height] duration-100"
            style={{
              height: `${Math.max(5, Math.min(23, 7 + 36 * Math.sqrt(power) * weight))}px`,
              opacity: 0.7 + weight * 0.3,
            }}
          />
        ))}
      </span>
      <span className="min-w-10 font-medium text-xs tabular-nums text-ink-2">
        {time}
      </span>
      <button
        type="button"
        disabled={disabled}
        onClick={onStop}
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-wash text-text hover:bg-line disabled:opacity-50"
        aria-label="Stop recording"
        title="Stop recording"
      >
        <Square size={13} fill="currentColor" />
      </button>
    </div>
  );
}
