import { Camera } from "lucide-react";

export default function CameraCard({ camera }) {
  const isActive = camera.status === "Active";

  return (
    <div className="bg-white border rounded-xl overflow-hidden">
      <div className="relative bg-slate-100 h-40 flex items-center justify-center">
        <Camera size={48} className="text-slate-300" />

        <span
          className={`absolute top-3 right-3 px-3 py-1 rounded-full text-xs font-semibold
            ${isActive ? "bg-green-100 text-green-700" : "bg-slate-200 text-slate-600"}`}
        >
          {camera.status}
        </span>

        <span className="absolute bottom-3 left-3 bg-slate-700 text-white text-xs px-2 py-1 rounded">
          {camera.id}
        </span>
      </div>

      <div className="p-4 space-y-2">
        <h3 className="font-semibold">{camera.name}</h3>
        <p className="text-sm text-blue-600 break-all">
          {camera.stream}
        </p>
      </div>
    </div>
  );
}
