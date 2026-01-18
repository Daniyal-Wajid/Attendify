import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

export default function Recognize() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const captureAndRecognize = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    setLoading(true);
    setResult(null);

    try {
      const res = await axios.post(
        "http://localhost:5000/api/attendance/recognize",
        { imageBase64: imageSrc }
      );

      setResult(res.data);
    } catch (err) {
      alert("Recognition failed");
    }

    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h2>🎥 Face Recognition Attendance</h2>

      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
        videoConstraints={{ facingMode: "user" }}
      />

      <br />
      <button onClick={captureAndRecognize} disabled={loading}>
        {loading ? "Recognizing..." : "Recognize"}
      </button>

      {result && result.recognized && (
        <div style={{ marginTop: 20 }}>
          <h3>✅ Recognized</h3>
          <p><b>Name:</b> {result.student.name}</p>
          <p><b>Roll No:</b> {result.student.rollNumber}</p>
          <p><b>Email:</b> {result.student.email}</p>
        </div>
      )}

      {result && !result.recognized && (
        <h3 style={{ color: "red" }}>❌ Face Not Recognized</h3>
      )}
    </div>
  );
}
