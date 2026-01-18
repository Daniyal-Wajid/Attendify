import Topbar from "../../components/Topbar";
import CameraStats from "../../components/CameraStats";
import CamerasGrid from "../../components/CamerasGrid";

export default function Cameras() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Cameras Management</h1>
            <p className="text-slate-500 dark:text-gray-400">Manage surveillance cameras</p>
          </div>

          <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
            + Add Camera
          </button>
        </div>

        <CameraStats />
        <CamerasGrid />
      </div>
    </>
  );
}
