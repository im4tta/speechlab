import { useRef, useState } from "react";
import {
  Activity,
  Cpu,
  Database,
  GripVertical,
  HardDrive,
  MemoryStick,
  Zap,
} from "lucide-react";
import { useSystemStats } from "@/hooks/useSystemStats";
import type { MemStat } from "@/types/models";

interface Props {
  isDark: boolean;
  open: boolean;
  onToggle: (open: boolean) => void;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(0)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function barColor(pct: number): string {
  if (pct >= 90) return "bg-red-500";
  if (pct >= 75) return "bg-amber-500";
  return "bg-indigo-500";
}

function memValue(m: MemStat): string {
  return `${fmtBytes(m.used_bytes)}/${fmtBytes(m.total_bytes)}`;
}

function Row({
  icon,
  label,
  pct,
  value,
  isDark,
}: {
  icon: React.ReactNode;
  label: string;
  pct?: number;
  value: string;
  isDark: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className={`shrink-0 ${isDark ? "text-zinc-500" : "text-zinc-400"}`}>{icon}</span>
      <span
        className={`w-10 shrink-0 text-[11px] font-medium ${isDark ? "text-zinc-300" : "text-zinc-600"}`}
      >
        {label}
      </span>
      {pct != null ? (
        <span
          className={`h-1.5 flex-1 rounded-full overflow-hidden ${
            isDark ? "bg-zinc-800" : "bg-gray-200"
          }`}
        >
          <span
            className={`block h-full rounded-full ${barColor(pct)}`}
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
        </span>
      ) : (
        <span className="flex-1" />
      )}
      <span
        className={`shrink-0 text-[11px] tabular-nums ${
          pct != null && pct >= 75
            ? "text-amber-500"
            : isDark
              ? "text-zinc-400"
              : "text-zinc-500"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

/**
 * Floating, draggable system monitor. Minimizes to a small chip; drag the grip
 * to move it anywhere. Polls /api/system/stats only while open.
 */
export function SystemMonitorWidget({ isDark, open, onToggle }: Props) {
  const stats = useSystemStats(open);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const dragState = useRef<{ dx: number; dy: number } | null>(null);

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => onToggle(true)}
        className={`fixed bottom-4 left-4 z-40 flex items-center gap-2 px-3 py-2 rounded-xl border shadow-2xl backdrop-blur-xl transition-colors ${
          isDark
            ? "bg-zinc-900/90 border-zinc-700 text-zinc-300 hover:text-white"
            : "bg-white/95 border-gray-200 text-gray-700 hover:text-gray-900"
        }`}
        title="Open system monitor"
      >
        <Activity className="w-4 h-4 text-indigo-500" />
        <span className="text-xs font-medium">System</span>
      </button>
    );
  }

  const onPointerDown = (e: React.PointerEvent) => {
    dragState.current = { dx: e.clientX - (pos?.x ?? 16), dy: e.clientY - (pos?.y ?? 16) };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragState.current) return;
    setPos({
      x: Math.max(0, Math.min(window.innerWidth - 320, e.clientX - dragState.current.dx)),
      y: Math.max(0, Math.min(window.innerHeight - 260, e.clientY - dragState.current.dy)),
    });
  };
  const onPointerUp = () => {
    dragState.current = null;
  };

  const dash = <span className={isDark ? "text-zinc-600" : "text-gray-400"}>—</span>;

  return (
    <div
      className={`fixed z-40 w-72 rounded-2xl border shadow-2xl backdrop-blur-xl overflow-hidden ${
        isDark ? "bg-zinc-900/95 border-zinc-700" : "bg-white/95 border-gray-200"
      }`}
      style={{ left: pos?.x ?? 16, top: pos?.y ?? 16 }}
    >
      {/* Drag handle / header */}
      <div
        className={`flex items-center gap-2 px-3 py-2 border-b cursor-grab active:cursor-grabbing select-none ${
          isDark ? "border-zinc-800" : "border-gray-200"
        }`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        title="Drag to move"
      >
        <GripVertical className={`w-3.5 h-3.5 ${isDark ? "text-zinc-600" : "text-gray-400"}`} />
        <Activity className={`w-4 h-4 ${isDark ? "text-indigo-400" : "text-indigo-600"}`} />
        <span
          className={`text-xs font-semibold ${isDark ? "text-zinc-200" : "text-gray-800"}`}
        >
          System monitor
        </span>
        <button
          type="button"
          onClick={() => onToggle(false)}
          className={`ml-auto p-1 rounded transition-colors ${
            isDark
              ? "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800"
              : "text-gray-500 hover:text-gray-800 hover:bg-gray-100"
          }`}
          title="Minimize"
          aria-label="Minimize system monitor"
        >
          —
        </button>
      </div>

      {/* Body */}
      <div className="p-3 space-y-2.5">
        {stats ? (
          <>
            <Row
              icon={<Cpu size={13} />}
              label="CPU"
              pct={stats.cpu_percent}
              value={`${stats.cpu_percent.toFixed(0)}%`}
              isDark={isDark}
            />
            <Row
              icon={<MemoryStick size={13} />}
              label="RAM"
              pct={stats.ram.percent}
              value={memValue(stats.ram)}
              isDark={isDark}
            />
            {stats.vram ? (
              <Row
                icon={<Zap size={13} />}
                label="VRAM"
                pct={stats.vram.percent}
                value={memValue(stats.vram)}
                isDark={isDark}
              />
            ) : (
              <Row icon={<Zap size={13} />} label="VRAM" value="n/a" isDark={isDark} />
            )}
            <Row
              icon={<HardDrive size={13} />}
              label="Disk"
              pct={stats.disk.percent}
              value={memValue(stats.disk)}
              isDark={isDark}
            />
            <Row
              icon={<Database size={13} />}
              label="Cache"
              value={fmtBytes(stats.cache_bytes)}
              isDark={isDark}
            />
          </>
        ) : (
          <div className={`text-xs text-center py-3 ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
            Loading system stats {dash}
          </div>
        )}
      </div>
    </div>
  );
}
