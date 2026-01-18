import { Camera, Circle } from "lucide-react";

export default function CameraStats() {
  return (
    <div className="grid grid-cols-3 gap-6">
      <div className="bg-white border rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 bg-blue-100 text-blue-600 rounded-full">
          <Camera />
        </div>
        <div>
          <p className="text-sm text-slate-500">Total Cameras</p>
          <p className="text-2xl font-bold">5</p>
        </div>
      </div>

      <div className="bg-white border rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 bg-green-100 text-green-600 rounded-full">
          <Circle size={16} fill="currentColor" />
        </div>
        <div>
          <p className="text-sm text-slate-500">Active</p>
          <p className="text-2xl font-bold">4</p>
        </div>
      </div>

      <div className="bg-white border rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 bg-slate-200 text-slate-500 rounded-full">
          <Circle size={16} fill="currentColor" />
        </div>
        <div>
          <p className="text-sm text-slate-500">Offline</p>
          <p className="text-2xl font-bold">1</p>
        </div>
      </div>
    </div>
  );
}
