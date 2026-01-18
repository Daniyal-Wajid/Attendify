import { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Camera, Globe } from "lucide-react";

export default function StudentRegister() {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const navigate = useNavigate();

  const [showCameraScreen, setShowCameraScreen] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [videoSource, setVideoSource] = useState(null); // "camera" or "upload"
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [uploadedVideoPreview, setUploadedVideoPreview] = useState(null);
  const [videoDuration, setVideoDuration] = useState(null);
  const [loading, setLoading] = useState(false);
  const [timer, setTimer] = useState(10);
  const [stream, setStream] = useState(null);
  const [cameraMode, setCameraMode] = useState("webcam"); // 'webcam' | 'ip'
  const [ipUrl, setIpUrl] = useState("");
  const [ipError, setIpError] = useState(false);
  const ipCameraRef = useRef(null);
  const recordingCanvasRef = useRef(null);
  const canvasIntervalRef = useRef(null);

  const [form, setForm] = useState({
    name: "",
    rollNumber: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (recording && timer > 0) {
      const interval = setInterval(() => {
        setTimer((prev) => {
          if (prev <= 1) {
            clearInterval(interval);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [recording, timer]);

  // Cleanup video preview URL on unmount
  useEffect(() => {
    return () => {
      if (uploadedVideoPreview) {
        URL.revokeObjectURL(uploadedVideoPreview);
      }
    };
  }, [uploadedVideoPreview]);

  async function openCameraScreen() {
    setIpError(false);
    if (cameraMode === "webcam") {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        setStream(mediaStream);

        // Set video source
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }

        // Create MediaRecorder when camera is ready
        const recorder = new MediaRecorder(mediaStream, {
          mimeType: 'video/webm;codecs=vp8'
        });
        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            chunksRef.current.push(e.data);
          }
        };

        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: "video/webm" });
          setVideoFile(blob);
          setVideoSource("camera");
          setRecording(false);
          setTimer(10);
          setErrors({ ...errors, video: "" });
        };

        mediaRecorderRef.current = recorder;
        setCameraReady(true);
        setShowCameraScreen(true);
      } catch (err) {
        alert("Camera permission denied");
        console.error(err);
      }
    } else {
      // IP Mode
      if (!ipUrl) {
        alert("Please enter IP Camera URL first");
        return;
      }
      setShowCameraScreen(true);
      // Give modal some time to render refs
      setTimeout(() => {
        if (recordingCanvasRef.current && ipCameraRef.current) {
          const canvas = recordingCanvasRef.current;
          const img = ipCameraRef.current;
          const ctx = canvas.getContext("2d");

          const streamFromCanvas = canvas.captureStream(25); // 25 FPS
          setStream(streamFromCanvas);

          if (videoRef.current) {
            videoRef.current.srcObject = streamFromCanvas;
          }

          const recorder = new MediaRecorder(streamFromCanvas, {
            mimeType: 'video/webm;codecs=vp8'
          });

          recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
              chunksRef.current.push(e.data);
            }
          };

          recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: "video/webm" });
            setVideoFile(blob);
            setVideoSource("camera");
            setRecording(false);
            setTimer(10);
            setErrors({ ...errors, video: "" });
          };

          mediaRecorderRef.current = recorder;

          // Start the drawing loop
          canvasIntervalRef.current = setInterval(() => {
            if (img.complete && img.naturalWidth !== 0) {
              canvas.width = img.naturalWidth;
              canvas.height = img.naturalHeight;
              ctx.drawImage(img, 0, 0);
            }
          }, 40); // ~25 FPS

          setCameraReady(true);
        }
      }, 500);
    }
  }

  function startRecording() {
    if (!stream) {
      alert("Camera stream not available");
      return;
    }

    // Recreate MediaRecorder if needed
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === "inactive") {
      try {
        const recorder = new MediaRecorder(stream, {
          mimeType: 'video/webm;codecs=vp8'
        });
        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            chunksRef.current.push(e.data);
          }
        };

        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: "video/webm" });
          setVideoFile(blob);
          setVideoSource("camera");
          setRecording(false);
          setTimer(10);
          setErrors({ ...errors, video: "" });
        };

        mediaRecorderRef.current = recorder;
      } catch (err) {
        console.error("Error creating MediaRecorder:", err);
        alert("Failed to initialize recorder. Please try again.");
        return;
      }
    }

    if (mediaRecorderRef.current.state === "recording") {
      return; // Already recording
    }

    chunksRef.current = [];
    try {
      mediaRecorderRef.current.start();
      setRecording(true);
      setTimer(10);

      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
          mediaRecorderRef.current.stop();
        }
      }, 10000);
    } catch (err) {
      console.error("Error starting recording:", err);
      alert("Failed to start recording. Please try again.");
    }
  }

  function closeCameraScreen() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (canvasIntervalRef.current) {
      clearInterval(canvasIntervalRef.current);
      canvasIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        // Ignore errors if already stopped
      }
    }
    mediaRecorderRef.current = null;
    setShowCameraScreen(false);
    setCameraReady(false);
    setRecording(false);
    setTimer(10);
    setIpError(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function retakeVideo() {
    setVideoFile(null);
    setVideoSource(null);
    setUploadedVideo(null);
    setUploadedVideoPreview(null);
    setVideoDuration(null);
    setTimer(10);
    chunksRef.current = [];
    // Recreate MediaRecorder if needed
    if (stream && !mediaRecorderRef.current) {
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoFile(blob);
        setRecording(false);
        setTimer(10);
      };
      mediaRecorderRef.current = recorder;
    }
  }

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      alert("Please select a valid video file");
      e.target.value = '';
      return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      alert("Video file size must be less than 50MB");
      e.target.value = '';
      return;
    }

    // Check video duration
    const videoElement = document.createElement('video');
    videoElement.preload = 'metadata';

    videoElement.onloadedmetadata = () => {
      window.URL.revokeObjectURL(videoElement.src);
      const duration = videoElement.duration;
      setVideoDuration(duration);

      // Check if video is approximately 10 seconds (allow 8-12 seconds range)
      if (duration < 8 || duration > 12) {
        alert(`Video duration is ${duration.toFixed(1)} seconds. Please upload a video that is approximately 10 seconds (8-12 seconds allowed).`);
        e.target.value = '';
        setUploadedVideo(null);
        setUploadedVideoPreview(null);
        setVideoDuration(null);
        return;
      }

      // Video is valid, set it
      setUploadedVideo(file);
      setVideoFile(file); // Use same state for consistency
      setVideoSource("upload");
      setErrors({ ...errors, video: "" });

      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      setUploadedVideoPreview(previewUrl);
    };

    videoElement.onerror = () => {
      alert("Error loading video file. Please try another file.");
      e.target.value = '';
      setUploadedVideo(null);
      setUploadedVideoPreview(null);
      setVideoDuration(null);
    };

    videoElement.src = URL.createObjectURL(file);
  };

  const removeUploadedVideo = () => {
    if (uploadedVideoPreview) {
      URL.revokeObjectURL(uploadedVideoPreview);
    }
    setUploadedVideo(null);
    setUploadedVideoPreview(null);
    setVideoFile(null);
    setVideoSource(null);
    setVideoDuration(null);
    // Reset file input
    const fileInput = document.getElementById('video-upload-input');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!form.name.trim()) {
      newErrors.name = "Name is required";
    }
    if (!form.rollNumber.trim()) {
      newErrors.rollNumber = "Roll number is required";
    }
    if (!form.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(form.email)) {
      newErrors.email = "Email is invalid";
    }
    if (!form.password) {
      newErrors.password = "Password is required";
    } else if (form.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }
    if (form.password !== form.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }
    if (!videoFile && !uploadedVideo) {
      newErrors.video = "Please record a video or upload a video file";
    } else if (uploadedVideo && videoDuration && (videoDuration < 8 || videoDuration > 12)) {
      newErrors.video = "Video must be approximately 10 seconds (8-12 seconds allowed)";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("name", form.name);
      formData.append("rollNumber", form.rollNumber);
      formData.append("email", form.email);
      formData.append("password", form.password);
      formData.append("confirmPassword", form.confirmPassword);

      // Use uploaded video if available, otherwise use recorded video
      const videoToUpload = uploadedVideo || videoFile;
      if (!videoToUpload) {
        alert("Please provide a video");
        setLoading(false);
        return;
      }

      formData.append("video", videoToUpload);

      const res = await fetch("http://localhost:5000/api/students/register", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        // Clean up preview URL if exists
        if (uploadedVideoPreview) {
          URL.revokeObjectURL(uploadedVideoPreview);
        }
        alert("Registration successful! You can now login.");
        navigate("/login");
      } else {
        alert(data.error || "Registration failed. Please try again.");
      }
    } catch (err) {
      alert("Registration failed. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border dark:border-gray-700 p-8 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <h1 className="text-3xl font-bold text-center mb-2 text-gray-800 dark:text-white">
          Attendify - Student Registration
        </h1>
        <p className="text-center text-gray-600 dark:text-gray-400 mb-8">
          Register with your details and video
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Full Name *
              </label>
              <input
                type="text"
                required
                value={form.name}
                onChange={(e) => {
                  setForm({ ...form, name: e.target.value });
                  setErrors({ ...errors, name: "" });
                }}
                className={`w-full px-4 py-3 border dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 ${errors.name ? "border-red-500 dark:border-red-500" : "border-gray-300 dark:border-gray-600"
                  }`}
                placeholder="Enter your full name"
              />
              {errors.name && (
                <p className="text-red-500 text-xs mt-1">{errors.name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Roll Number *
              </label>
              <input
                type="text"
                required
                value={form.rollNumber}
                onChange={(e) => {
                  setForm({ ...form, rollNumber: e.target.value });
                  setErrors({ ...errors, rollNumber: "" });
                }}
                className={`w-full px-4 py-3 border dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 ${errors.rollNumber ? "border-red-500 dark:border-red-500" : "border-gray-300 dark:border-gray-600"
                  }`}
                placeholder="Enter your roll number"
              />
              {errors.rollNumber && (
                <p className="text-red-500 text-xs mt-1">{errors.rollNumber}</p>
              )}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Email *
            </label>
            <input
              type="email"
              required
              value={form.email}
              onChange={(e) => {
                setForm({ ...form, email: e.target.value });
                setErrors({ ...errors, email: "" });
              }}
              className={`w-full px-4 py-3 border dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 ${errors.email ? "border-red-500 dark:border-red-500" : "border-gray-300 dark:border-gray-600"
                }`}
              placeholder="Enter your email"
            />
            {errors.email && (
              <p className="text-red-500 text-xs mt-1">{errors.email}</p>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password *
              </label>
              <input
                type="password"
                required
                value={form.password}
                onChange={(e) => {
                  setForm({ ...form, password: e.target.value });
                  setErrors({ ...errors, password: "" });
                }}
                className={`w-full px-4 py-3 border dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 ${errors.password ? "border-red-500 dark:border-red-500" : "border-gray-300 dark:border-gray-600"
                  }`}
                placeholder="Enter password"
              />
              {errors.password && (
                <p className="text-red-500 text-xs mt-1">{errors.password}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Confirm Password *
              </label>
              <input
                type="password"
                required
                value={form.confirmPassword}
                onChange={(e) => {
                  setForm({ ...form, confirmPassword: e.target.value });
                  setErrors({ ...errors, confirmPassword: "" });
                }}
                className={`w-full px-4 py-3 border dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 ${errors.confirmPassword ? "border-red-500 dark:border-red-500" : "border-gray-300 dark:border-gray-600"
                  }`}
                placeholder="Confirm password"
              />
              {errors.confirmPassword && (
                <p className="text-red-500 text-xs mt-1">
                  {errors.confirmPassword}
                </p>
              )}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Video (10 seconds) *
            </label>
            <div className="space-y-4">
              {!videoFile && !uploadedVideo ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex flex-col gap-4">
                    <div className="flex bg-gray-100 rounded-lg p-1">
                      <button
                        type="button"
                        onClick={() => setCameraMode("webcam")}
                        className={`flex-1 px-4 py-2 text-sm font-medium rounded flex items-center justify-center gap-2 transition-colors ${cameraMode === "webcam" ? "bg-white shadow text-blue-600" : "text-gray-500 hover:text-gray-700"}`}
                      >
                        <Camera size={16} /> Webcam
                      </button>
                      <button
                        type="button"
                        onClick={() => setCameraMode("ip")}
                        className={`flex-1 px-4 py-2 text-sm font-medium rounded flex items-center justify-center gap-2 transition-colors ${cameraMode === "ip" ? "bg-white shadow text-blue-600" : "text-gray-500 hover:text-gray-700"}`}
                      >
                        <Globe size={16} /> IP Camera
                      </button>
                    </div>

                    {cameraMode === "ip" && (
                      <div>
                        <input
                          type="text"
                          value={ipUrl}
                          onChange={(e) => setIpUrl(e.target.value)}
                          placeholder="Enter IP Camera URL (e.g., http://.../video)"
                          className="w-full px-4 py-2 border dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
                        />
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={openCameraScreen}
                      disabled={cameraMode === "ip" && !ipUrl}
                      className="px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium text-lg flex items-center justify-center gap-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                      📷 {cameraMode === "webcam" ? "Record Video" : "Connect & Record"}
                    </button>
                  </div>
                  <label className="px-6 py-4 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium text-lg flex items-center justify-center gap-2 cursor-pointer">
                    📁 Upload Video
                    <input
                      id="video-upload-input"
                      type="file"
                      accept="video/*"
                      onChange={handleVideoUpload}
                      className="hidden"
                    />
                  </label>
                </div>
              ) : (
                <div className="space-y-4">
                  {uploadedVideoPreview && (
                    <div className="relative">
                      <video
                        src={uploadedVideoPreview}
                        controls
                        className="w-full rounded-lg border-2 border-green-500"
                        style={{ maxHeight: "300px" }}
                      />
                      {videoDuration && (
                        <div className="absolute top-2 right-2 bg-green-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                          {videoDuration.toFixed(1)}s
                        </div>
                      )}
                    </div>
                  )}
                  <div className="flex items-center gap-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <span className="text-green-600 font-medium">
                      ✓ {videoSource === "upload" ? "Video uploaded" : "Video recorded"} successfully
                      {videoDuration && ` (${videoDuration.toFixed(1)}s)`}
                    </span>
                    <button
                      type="button"
                      onClick={videoSource === "upload" ? removeUploadedVideo : retakeVideo}
                      className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 text-sm"
                    >
                      {videoSource === "upload" ? "Remove" : "Re-record"}
                    </button>
                  </div>
                </div>
              )}
              {errors.video && (
                <p className="text-red-500 text-xs">{errors.video}</p>
              )}
              <div className="text-xs text-gray-500 space-y-1">
                <p>• Record a 10-second video using your camera, OR</p>
                <p>• Upload a video file that is exactly 10 seconds (8-12 seconds allowed)</p>
                <p>• The video will be used for face recognition training</p>
              </div>
            </div>
          </div>

          {/* Camera Screen Modal */}
          {showCameraScreen && (
            <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4">
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border dark:border-gray-700 max-w-2xl w-full p-6">
                <div className="relative">
                  {cameraMode === "webcam" ? (
                    <video
                      ref={(el) => {
                        videoRef.current = el;
                        if (el && stream && cameraMode === "webcam") {
                          el.srcObject = stream;
                        }
                      }}
                      autoPlay
                      muted
                      playsInline
                      className="w-full rounded-lg border-4 border-blue-500"
                      style={{ maxHeight: "70vh" }}
                    />
                  ) : (
                    <div className="relative">
                      <img
                        ref={ipCameraRef}
                        src={ipUrl}
                        crossOrigin="anonymous"
                        className="w-full rounded-lg border-4 border-blue-500"
                        style={{ maxHeight: "70vh", display: ipError ? 'none' : 'block' }}
                        alt="IP Stream"
                        onError={() => setIpError(true)}
                      />
                      {ipError && (
                        <div className="w-full aspect-video bg-gray-900 rounded-lg flex flex-col items-center justify-center text-white gap-2">
                          <Globe size={48} className="text-gray-500" />
                          <p className="text-red-500 font-medium">Stream Error</p>
                          <p className="text-xs text-gray-400">Check URL and CORS settings</p>
                        </div>
                      )}
                      {/* Hidden canvas for recording */}
                      <canvas ref={recordingCanvasRef} className="hidden" />
                      {/* Preview video for what's being captured */}
                      <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                        className="w-full absolute inset-0 opacity-0 pointer-events-none"
                      />
                    </div>
                  )}

                  {/* Timer Overlay */}
                  {recording && (
                    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-6 py-3 rounded-full shadow-lg">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                        <span className="text-2xl font-bold">Recording: {timer}s</span>
                      </div>
                    </div>
                  )}

                  {/* Recording Indicator */}
                  {recording && (
                    <div className="absolute top-4 right-4 bg-red-600 text-white px-4 py-2 rounded-full flex items-center gap-2">
                      <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                      <span className="font-semibold">REC</span>
                    </div>
                  )}
                </div>

                <div className="mt-6 flex flex-col items-center gap-4">
                  {!videoFile ? (
                    <>
                      {!recording ? (
                        <button
                          type="button"
                          onClick={startRecording}
                          className="w-full px-8 py-4 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold text-lg"
                        >
                          🎬 Start Recording (10 seconds)
                        </button>
                      ) : (
                        <div className="text-center">
                          <p className="text-gray-600 mb-4">
                            Recording in progress... Please keep your face visible
                          </p>
                          {timer === 0 && (
                            <p className="text-green-600 font-semibold">
                              ✓ Recording completed!
                            </p>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-center w-full">
                      <p className="text-green-600 font-semibold text-lg mb-4">
                        ✓ Video recorded successfully!
                      </p>
                      <div className="flex gap-4">
                        <button
                          type="button"
                          onClick={retakeVideo}
                          className="flex-1 px-6 py-3 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 font-medium"
                        >
                          Re-record
                        </button>
                        <button
                          type="button"
                          onClick={closeCameraScreen}
                          className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
                        >
                          Use This Video
                        </button>
                      </div>
                    </div>
                  )}

                  {!recording && !videoFile && (
                    <button
                      type="button"
                      onClick={closeCameraScreen}
                      className="px-6 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="flex gap-4 pt-4">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Registering..." : "Register"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/login")}
              className="px-6 py-3 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 font-medium"
            >
              Cancel
            </button>
          </div>
        </form>

        <div className="mt-6 text-center">
          <p className="text-gray-600">
            Already have an account?{" "}
            <button
              onClick={() => navigate("/login")}
              className="text-blue-600 hover:text-blue-700 font-medium"
            >
              Sign in
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

