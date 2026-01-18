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
    } catch (err) {
      alert("Camera permission denied");
      console.error(err);
    }
  }

  function startRecording() {
    if (!mediaRecorderRef.current) {
      alert("Camera not ready");
      return;
    }

    chunksRef.current = [];
    mediaRecorderRef.current.start();
    setRecording(true);

    setTimeout(stopRecording, 10000);
  }

  function stopRecording() {
    if (!mediaRecorderRef.current) return;

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

      alert("Registered & AI training started");
    };
  }

  return (
    <div className="p-6 max-w-md mx-auto space-y-3">
      <video
        ref={videoRef}
        autoPlay
        muted
        className="w-full rounded-lg border"
      />

      <input
        className="border p-2 w-full"
        placeholder="Name"
        onChange={(e) => setForm({ ...form, name: e.target.value })}
      />
      <input
        className="border p-2 w-full"
        placeholder="Roll No"
        onChange={(e) => setForm({ ...form, rollNumber: e.target.value })}
      />
      <input
        className="border p-2 w-full"
        placeholder="Email"
        onChange={(e) => setForm({ ...form, email: e.target.value })}
      />

      <div className="flex gap-2 pt-2">
        <button
          onClick={startCamera}
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Start Camera
        </button>

        <button
          onClick={startRecording}
          disabled={!cameraReady || recording}
          className="bg-green-600 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Record 10s
        </button>
      </div>
    </div>
  );
}
