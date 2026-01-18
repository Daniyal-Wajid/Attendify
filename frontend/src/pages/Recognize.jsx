import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { Camera, Globe, RefreshCw, UserCheck, AlertCircle, ScanFace } from "lucide-react";

export default function Recognize() {
  const webcamRef = useRef(null);
  const ipCameraRef = useRef(null);

  const [mode, setMode] = useState("webcam"); // 'webcam' | 'ip'
  const [ipUrl, setIpUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Helper to capture from IP stream securely
  const captureIpFrame = () => {
    return new Promise((resolve, reject) => {
      const img = ipCameraRef.current;
      if (!img) {
        reject("IP Camera image element not found");
        return;
      }

      // Check if image is actually loaded
      if (!img.complete || img.naturalWidth === 0) {
        reject("IP Camera stream not loaded. Check URL.");
        return;
      }

      try {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");

        // This might fail if CORS is not enabled on the camera
        ctx.drawImage(img, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg");
        resolve(dataUrl);
      } catch (e) {
        reject("CORS Error: Unable to capture frame from this IP Camera. The camera must support simple CORS.");
      }
    });
  };

  const captureAndRecognize = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    let imageSrc = null;

    try {
      if (mode === "webcam") {
        if (webcamRef.current) {
          imageSrc = webcamRef.current.getScreenshot();
        } else {
          throw new Error("Webcam reference not ready");
        }
      } else {
        // IP Camera Mode
        if (!ipUrl) throw new Error("Please enter a valid IP Camera URL");
        imageSrc = await captureIpFrame();
      }

      if (!imageSrc) {
        throw new Error("Failed to capture image. Please try again.");
      }

      const res = await axios.post(
        "http://localhost:5000/api/attendance/recognize",
        { imageBase64: imageSrc }
      );

      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || err.message || "Recognition failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center py-10 px-4">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-900 flex items-center justify-center gap-3">
          <ScanFace className="text-blue-600" size={36} />
          Face Recognition
        </h1>
        <p className="text-slate-500 mt-2">Mark attendance effortlessly</p>
      </div>

      <div className="w-full max-w-4xl grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Camera Section */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100 flex flex-col">
          {/* Toggles */}
          <div className="flex border-b">
            <button
              onClick={() => { setMode("webcam"); setResult(null); setError(null); }}
              className={`flex-1 py-4 font-medium flex items-center justify-center gap-2 transition-colors
                ${mode === "webcam" ? "bg-blue-50 text-blue-700 border-b-2 border-blue-600" : "text-slate-500 hover:bg-slate-50"}`}
            >
              <Camera size={20} /> Laptop Webcam
            </button>
            <button
              onClick={() => { setMode("ip"); setResult(null); setError(null); }}
              className={`flex-1 py-4 font-medium flex items-center justify-center gap-2 transition-colors
                ${mode === "ip" ? "bg-blue-50 text-blue-700 border-b-2 border-blue-600" : "text-slate-500 hover:bg-slate-50"}`}
            >
              <Globe size={20} /> IP Camera
            </button>
          </div>

          {/* Viewport */}
          <div className="relative bg-black h-80 lg:h-96 flex items-center justify-center overflow-hidden group">
            {mode === "webcam" ? (
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="w-full h-full object-cover"
                videoConstraints={{ facingMode: "user" }}
                onUserMediaError={() => setError("Could not access webcam")}
              />
            ) : (
              <div className="w-full h-full flex flex-col">
                {ipUrl ? (
                  <img
                    ref={ipCameraRef}
                    src={ipUrl}
                    alt="IP Stream"
                    crossOrigin="anonymous" // Attempt to fix CORS
                    className="w-full h-full object-contain bg-black"
                    onError={() => setError("Failed to load stream. Check URL.")}
                  />
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-2">
                    <Globe size={48} className="opacity-20" />
                    <p>Enter IP URL below</p>
                  </div>
                )}
              </div>
            )}

            {/* Overlay Loading Spinner */}
            {loading && (
              <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-10 backdrop-blur-sm">
                <div className="flex flex-col items-center text-white">
                  <RefreshCw className="animate-spin mb-2" size={32} />
                  <span className="font-medium">Processing...</span>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="p-6 space-y-4 bg-white">
            {mode === "ip" && (
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Stream URL</label>
                <input
                  type="text"
                  placeholder="http://192.168.1.X:8080/video"
                  value={ipUrl}
                  onChange={(e) => { setIpUrl(e.target.value); setError(null); }}
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                />
                <p className="text-xs text-slate-400 mt-1">Ensure the camera supports CORS or simple image streaming.</p>
              </div>
            )}

            <button
              onClick={captureAndRecognize}
              disabled={loading || (mode === "ip" && !ipUrl)}
              className={`w-full py-3 rounded-xl font-bold text-white shadow-lg transition-all transform active:scale-95
                ${loading || (mode === "ip" && !ipUrl)
                  ? "bg-slate-300 cursor-not-allowed shadow-none"
                  : "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:shadow-blue-200"}`}
            >
              {loading ? "Recognizing..." : "Capture & Recognize"}
            </button>

            {error && (
              <div className="flex items-start gap-2 text-red-600 bg-red-50 p-3 rounded-lg text-sm">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100 p-6 h-full flex flex-col justify-center">
            {!result ? (
              <div className="text-center text-slate-400 flex flex-col items-center">
                <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                  <ScanFace size={32} />
                </div>
                <p>Scan a face to see student details here.</p>
              </div>
            ) : result.recognized ? (
              <div className="text-center animate-in fade-in zoom-in duration-300">
                <div className="w-20 h-20 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                  <UserCheck size={40} />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-1">{result.student.name}</h2>
                <div className="flex items-center justify-center gap-2 mb-6">
                  <p className="text-blue-600 font-medium">{result.student.rollNumber}</p>
                  {result.confidence && (
                    <span className="px-2 py-0.5 bg-blue-50 text-blue-500 text-xs font-bold rounded border border-blue-100 italic">
                      {(result.confidence * 100).toFixed(1)}% match
                    </span>
                  )}
                </div>

                <div className="bg-slate-50 rounded-xl p-4 text-left space-y-3 border border-slate-100">
                  <div className="flex justify-between">
                    <span className="text-slate-500 text-sm">Email</span>
                    <span className="text-slate-700 font-medium text-right">{result.student.email}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 text-sm">Department</span>
                    <span className="text-slate-700 font-medium text-right">{result.student.department || "N/A"}</span>
                  </div>
                  <div className="pt-3 border-t border-slate-200 flex justify-between items-center">
                    <span className="text-slate-500 text-sm">Status</span>
                    <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-bold rounded-full uppercase tracking-wide">
                      Present
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-red-500 animate-in fade-in zoom-in duration-300">
                <div className="w-20 h-20 bg-red-50 text-red-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <ScanFace size={40} />
                </div>
                <h3 className="text-xl font-bold">Face Not Recognized</h3>
                <p className="text-slate-400 text-sm mt-2">Please try again or register the student.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

