import { useRef, useState } from "react";

export default function Register() {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);

  const [form, setForm] = useState({
    name: "",
    rollNumber: "",
    email: "",
  });

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;

      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);

      mediaRecorderRef.current = recorder;
      setCameraReady(true);
    } catch {
      alert("Camera permission denied");
    }
  }

  function startRecording() {
    if (!mediaRecorderRef.current) return alert("Camera not ready");

    chunksRef.current = [];
    mediaRecorderRef.current.start();
    setRecording(true);
    setTimeout(stopRecording, 10000);
  }

  function stopRecording() {
    mediaRecorderRef.current.stop();
    setRecording(false);

    mediaRecorderRef.current.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const fd = new FormData();

      fd.append("name", form.name);
      fd.append("rollNumber", form.rollNumber);
      fd.append("email", form.email);
      fd.append("video", blob);

      await fetch("http://localhost:5000/api/students/register", {
        method: "POST",
        body: fd,
      });

      alert("Registration complete. AI training started.");
    };
  }

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 
      flex items-center justify-center p-4">

      <div className="w-full max-w-md bg-white dark:bg-gray-800 
        rounded-2xl shadow-xl p-6 space-y-4">

        <h2 className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
          Attendify Registration
        </h2>

        <video
          ref={videoRef}
          autoPlay
          muted
          className="w-full rounded-lg border dark:border-gray-700"
        />

        {["name", "rollNumber", "email"].map((field) => (
          <input
            key={field}
            placeholder={field === "rollNumber" ? "Roll Number" : field}
            onChange={(e) => setForm({ ...form, [field]: e.target.value })}
            className="w-full px-3 py-2 rounded-lg border 
              bg-gray-50 dark:bg-gray-700 
              border-gray-300 dark:border-gray-600"
          />
        ))}

        <div className="flex gap-3 pt-2">
          <button
            onClick={startCamera}
            className="flex-1 bg-indigo-600 text-white py-2 rounded-lg"
          >
            Start Camera
          </button>

          <button
            onClick={startRecording}
            disabled={!cameraReady || recording}
            className="flex-1 bg-green-600 text-white py-2 rounded-lg
              disabled:opacity-50"
          >
            Record 10s
          </button>
        </div>

        <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
          Please keep your face centered during recording.
        </p>
      </div>
    </div>
  );
}
