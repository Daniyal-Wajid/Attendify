import { cameras } from "../data/mockCameras";
import CameraCard from "./CameraCard";

export default function CamerasGrid() {
  return (
    <div className="grid grid-cols-3 gap-6">
      {cameras.map((cam) => (
        <CameraCard key={cam.id} camera={cam} />
      ))}
    </div>
  );
}
