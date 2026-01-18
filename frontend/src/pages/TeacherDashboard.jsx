import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";
import axios from "axios";
import { Camera, Globe } from "lucide-react";
import TeacherSidebar from "../components/TeacherSidebar";

export default function TeacherDashboard() {
  const [user, setUser] = useState(null);
  const [subjects, setSubjects] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState("");
  const [settings, setSettings] = useState(null);
  const [recognizedStudents, setRecognizedStudents] = useState([]);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [markedToday, setMarkedToday] = useState(new Set()); // Track students marked today
  const [attendance, setAttendance] = useState({});
  const [activeTab, setActiveTab] = useState("camera"); // "camera" or "attendance"
  const [isLiveFeedActive, setIsLiveFeedActive] = useState(false); // Track if live feed is running
  const [students, setStudents] = useState([]); // For manual entry
  const [showManualEntry, setShowManualEntry] = useState(false);
  const [manualEntryForm, setManualEntryForm] = useState({
    studentId: "",
    subjectId: "",
    date: new Date().toISOString().split('T')[0], // Today's date
    status: "present", // Default to present
  });
  const [enrollments, setEnrollments] = useState([]);
  const [enrolledStudents, setEnrolledStudents] = useState({}); // {subjectId: [students]}
  const [showBulkAttendance, setShowBulkAttendance] = useState(false);
  const [bulkAttendanceForm, setBulkAttendanceForm] = useState({
    subjectId: "",
    date: new Date().toISOString().split('T')[0],
    attendances: [], // [{studentId, status}]
  });

  // Camera State
  const [cameraMode, setCameraMode] = useState("webcam"); // 'webcam' | 'ip'
  const [ipUrl, setIpUrl] = useState("");
  const [ipError, setIpError] = useState(false);
  const ipCameraRef = useRef(null);

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null); // Reuse canvas for speed
  const recognitionTimeoutRef = useRef(null);
  const isComponentMounted = useRef(true);
  const isLiveFeedActiveRef = useRef(false); // Ref for loop safety
  const navigate = useNavigate();

  // Helper to capture from IP stream securely (Optimized)
  const captureIpFrame = () => {
    return new Promise((resolve, reject) => {
      const img = ipCameraRef.current;
      if (!img) return reject("No element");
      if (!img.complete || img.naturalWidth === 0) return reject("Not loaded");

      try {
        if (!offscreenCanvasRef.current) {
          offscreenCanvasRef.current = document.createElement("canvas");
        }
        const canvas = offscreenCanvasRef.current;
        if (canvas.width !== img.naturalWidth) canvas.width = img.naturalWidth;
        if (canvas.height !== img.naturalHeight) canvas.height = img.naturalHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL("image/jpeg", 0.7)); // Low quality for speed
      } catch (e) {
        reject("CORS Error");
      }
    });
  };

  useEffect(() => {
    isComponentMounted.current = true;
    return () => { isComponentMounted.current = false; };
  }, []);

  useEffect(() => {
    const token = localStorage.getItem("token");
    const userData = JSON.parse(localStorage.getItem("user") || "{}");

    if (!token || userData.role !== "teacher") {
      navigate("/login");
      return;
    }

    setUser(userData);
    loadData();
    // Don't auto-start live feed - let teacher control it

    return () => {
      if (recognitionTimeoutRef.current) {
        clearTimeout(recognitionTimeoutRef.current);
        recognitionTimeoutRef.current = null;
      }
      setIsLiveFeedActive(false);
    };
  }, [navigate]);

  const loadData = async () => {
    const token = localStorage.getItem("token");
    try {
      const [subjectsRes, settingsRes, attendanceRes, studentsRes, enrollmentsRes] = await Promise.all([
        fetch("http://localhost:5000/api/teachers/my-subjects", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/settings", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/teachers/attendance", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/students", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/enrollments?status=pending", {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      const subjectsData = await subjectsRes.json();
      setSubjects(subjectsData);
      if (subjectsData.length > 0) {
        setSelectedSubject(subjectsData[0]._id);
        // Set default subject in manual entry form too
        if (!manualEntryForm.subjectId) {
          setManualEntryForm({ ...manualEntryForm, subjectId: subjectsData[0]._id });
        }
      }

      if (settingsRes.ok) {
        setSettings(await settingsRes.json());
      }

      if (attendanceRes.ok) {
        setAttendance(await attendanceRes.json());
      }

      if (studentsRes.ok) {
        setStudents(await studentsRes.json());
      }

      if (enrollmentsRes.ok) {
        const enrollmentsData = await enrollmentsRes.json();
        setEnrollments(enrollmentsData);

        // Load enrolled students for each subject
        const enrolledMap = {};
        for (const subject of subjectsData) {
          try {
            const enrolledRes = await fetch(
              `http://localhost:5000/api/enrollments/course/${subject._id}`,
              { headers: { Authorization: `Bearer ${token}` } }
            );
            if (enrolledRes.ok) {
              enrolledMap[subject._id] = await enrolledRes.json();
            }
          } catch (err) {
            console.error(`Error loading enrolled students for ${subject._id}:`, err);
          }
        }
        setEnrolledStudents(enrolledMap);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const startLiveRecognition = () => {
    if (isLiveFeedActive) return;
    if (!selectedSubject) return alert("Please select a subject first.");

    setIsLiveFeedActive(true);
    isLiveFeedActiveRef.current = true;

    const runRecognition = async () => {
      console.log("[Dashboard] Recognition loop tick");
      // Safety checks - use current values from refs
      if (!isComponentMounted.current || !isLiveFeedActiveRef.current) {
        console.log("[Dashboard] Loop stopping: mounted:", isComponentMounted.current, "active:", isLiveFeedActiveRef.current);
        return;
      }

      let imageSrc = null;
      try {
        if (cameraMode === "webcam") {
          imageSrc = webcamRef.current?.getScreenshot();
          if (!imageSrc) console.log("[Dashboard] Webcam screenshot NULL");
        } else if (ipUrl) {
          imageSrc = await captureIpFrame();
          if (!imageSrc) console.log("[Dashboard] IP frame NULL");
        }
      } catch (e) {
        console.error("[Dashboard] Capture failed:", e);
      }

      if (imageSrc) {
        try {
          console.log("[Dashboard] Sending recognition request to backend...");
          const token = localStorage.getItem("token");
          const res = await axios.post(
            "http://localhost:5000/api/attendance/recognize-live",
            { imageBase64: imageSrc },
            { headers: { Authorization: `Bearer ${token}` } }
          );
          console.log("[Dashboard] Backend response:", res.data);

          // Update All detected faces for boxes
          if (res.data.faces) setDetectedFaces(res.data.faces);

          // Update multi-recognitions (Batch support)
          if (res.data.recognitions && res.data.recognitions.length > 0) {
            const now = Date.now();
            setRecognizedStudents((prev) => {
              // Update state while keeping other recent recognitions
              let updated = [...prev];
              res.data.recognitions.forEach(rec => {
                // Filter out old entry for THIS student if it exists
                updated = updated.filter(s => s.student._id !== rec.student?._id);
                // Add fresh one
                updated.push({ ...rec, timestamp: now });
                // Auto-mark attendance
                if (rec.student?._id) markAttendance(rec.student._id, "auto");
              });
              return updated;
            });
          } else if (res.data.faces && res.data.faces.length > 0) {
            // Keep recent names on screen (handled in drawing loop)
          } else {
            setDetectedFaces([]);
            setRecognizedStudents([]);
          }
        } catch (err) {
          console.error("AI Error:", err);
        }
      }

      // Schedule next call only AFTER current one finishes
      if (isLiveFeedActiveRef.current && isComponentMounted.current) {
        recognitionTimeoutRef.current = setTimeout(runRecognition, 5000); // 5 second interval
      }
    };

    runRecognition();
  };

  const stopLiveRecognition = () => {
    if (recognitionTimeoutRef.current) {
      clearTimeout(recognitionTimeoutRef.current);
      recognitionTimeoutRef.current = null;
    }
    setIsLiveFeedActive(false);
    isLiveFeedActiveRef.current = false;
    setRecognizedStudents([]);
    setDetectedFaces([]);
  };

  const markAttendance = async (studentId, markedBy = "manual") => {
    if (!selectedSubject) {
      return;
    }

    // Check if already marked today (client-side check to prevent spam)
    const todayKey = `${studentId}-${selectedSubject}-${new Date().toDateString()}`;
    if (markedToday.has(todayKey)) {
      return; // Already marked today, skip
    }

    const token = localStorage.getItem("token");
    try {
      const res = await fetch("http://localhost:5000/api/attendance/mark", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          studentId,
          subjectId: selectedSubject,
          markedBy,
        }),
      });

      // Check content type before parsing
      const contentType = res.headers.get("content-type");
      let data;

      if (contentType && contentType.includes("application/json")) {
        data = await res.json();
      } else {
        const text = await res.text();
        console.error("Non-JSON response from attendance mark:", text);
        return;
      }

      if (res.ok) {
        if (data.alreadyMarked) {
          // Already marked today - add to set and return silently
          setMarkedToday((prev) => new Set([...prev, todayKey]));
          return;
        }
        // Successfully marked - add to set and log
        setMarkedToday((prev) => new Set([...prev, todayKey]));
        const student = recognizedStudents.find((s) => s.student._id === studentId);
        if (student) {
          console.log(`✓ Attendance marked for ${student.student.name}`);
        }
      } else {
        // Handle different error types
        if (data.error?.includes("already marked") || data.message?.includes("already marked")) {
          // Already marked - add to set and return
          setMarkedToday((prev) => new Set([...prev, todayKey]));
          return;
        }

        if (data.error?.includes("not enrolled") || data.error?.includes("not approved")) {
          // Student not enrolled - log warning but don't show alert for auto-attendance
          const student = recognizedStudents.find((s) => s.student._id === studentId);
          const studentName = student?.student?.name || studentId;
          console.warn(`⚠️ Cannot mark attendance: ${studentName} is not enrolled in this course`);
          // For auto-attendance, don't show alert - just log
          if (markedBy === "manual") {
            alert(`Cannot mark attendance: ${data.error}`);
          }
          return;
        }

        console.error("Attendance marking error:", data.error || data.message);
        if (markedBy === "manual") {
          alert(data.error || data.message || "Failed to mark attendance");
        }
      }
    } catch (err) {
      console.error("Mark attendance error:", err);
      if (markedBy === "manual") {
        alert("Failed to mark attendance. Please try again.");
      }
    }
  };

  const handleManualMark = async (studentId) => {
    if (!settings?.allowManualAttendance) {
      alert("Manual attendance is not allowed by admin");
      return;
    }
    await markAttendance(studentId, "manual");
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const getActiveVideo = () => webcamRef.current?.video || ipCameraRef.current;

    const ctx = canvas.getContext("2d");

    const updateCanvasSize = () => {
      const video = webcamRef.current?.video || ipCameraRef.current;
      if (!video) return;

      const displayWidth = video.clientWidth;
      const displayHeight = video.clientHeight;

      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
      }
    };

    const draw = () => {
      updateCanvasSize();
      const video = webcamRef.current?.video || ipCameraRef.current;
      if (!video || !canvas.width) {
        requestAnimationFrame(draw);
        return;
      }

      // Get scaling factors
      const videoWidth = video.videoWidth || video.naturalWidth || 640;
      const videoHeight = video.videoHeight || video.naturalHeight || 480;
      const scaleX = canvas.width / videoWidth;
      const scaleY = canvas.height / videoHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw recognized students
      recognizedStudents.forEach((rec) => {
        // Only show if recognition is recent (within 2 seconds)
        if (Date.now() - rec.timestamp > 2000) return;

        if (rec.faceBox) {
          const { x, y, w, h } = rec.faceBox;
          const sx = x * scaleX;
          const sy = y * scaleY;
          const sw = w * scaleX;
          const sh = h * scaleY;

          // Draw green box
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 3;
          ctx.strokeRect(sx, sy, sw, sh);

          // Label
          const name = rec.student.name;
          const conf = rec.confidence ? `(${(rec.confidence * 100).toFixed(0)}%)` : "";
          const label = `${name} ${conf}`;

          ctx.font = "bold 14px Inter, system-ui, sans-serif";
          const textWidth = ctx.measureText(label).width;

          ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
          const labelHeight = 24;
          ctx.fillRect(sx, sy - labelHeight, textWidth + 12, labelHeight);

          ctx.fillStyle = "#ffffff";
          ctx.fillText(label, sx + 6, sy - 7);
        }
      });

      // 2. Draw other detected faces as "Unregistered"
      detectedFaces.forEach((face) => {
        if (!face || typeof face.x !== 'number') return;

        // Skip if this face is already handled by a recognized student
        const isRecognized = recognizedStudents.some(rec => {
          // Check if recognition is fresh (within 2 seconds)
          if (Date.now() - rec.timestamp < 2000 && rec.faceBox) {
            // Check overlap (allow some margin for movement/scaling)
            const overlap = Math.abs(rec.faceBox.x - face.x) < 50 && Math.abs(rec.faceBox.y - face.y) < 50;
            return overlap;
          }
          return false;
        });

        if (isRecognized) return;

        const sx = face.x * scaleX;
        const sy = face.y * scaleY;
        const sw = face.w * scaleX;
        const sh = face.h * scaleY;

        // Draw orange box for unregistered
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]); // Dashed line for unregistered
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.setLineDash([]); // Reset

        // "Unregistered" Label
        const label = "Unregistered";
        ctx.font = "bold 12px Inter, system-ui, sans-serif";
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = "rgba(245, 158, 11, 0.9)"; // Orange
        ctx.fillRect(sx, sy - 20, textWidth + 10, 20);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, sx + 5, sy - 5);
      });

      const frameId = requestAnimationFrame(draw);
      return frameId;
    };

    const frameId = draw();

    return () => {
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [recognizedStudents, detectedFaces, cameraMode, isLiveFeedActive]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
      <TeacherSidebar 
        activeTab={activeTab} 
        onTabChange={(tab) => {
          if (tab === "attendance" || tab === "enrollments") {
            loadData();
          }
          setActiveTab(tab);
        }}
        pendingEnrollmentsCount={enrollments.filter(e => e.status === "pending").length}
      />
      
      <main className="flex-1 overflow-y-auto flex flex-col">
        {/* Topbar */}
        <nav className="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700 h-16 flex items-center justify-between px-6 sticky top-0 z-10">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Teacher Dashboard</h2>
          <div className="flex items-center gap-4">
            <select
              value={selectedSubject}
              onChange={(e) => setSelectedSubject(e.target.value)}
              className="px-4 py-2 border dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            >
              {subjects.map((subject) => (
                <option key={subject._id} value={subject._id}>
                  {subject.name} ({subject.code})
                </option>
              ))}
            </select>
            <span className="text-gray-600 dark:text-gray-300 text-sm">{user?.name || user?.email}</span>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
            >
              Logout
            </button>
          </div>
        </nav>

        <div className="p-6">

        {activeTab === "camera" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Live Camera Feed */}
            <div className="lg:col-span-2">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">Live Camera Feed</h2>
                  <div className="flex items-center gap-3">
                    {/* Camera Toggles */}
                    <div className="flex bg-gray-100 rounded-lg p-1 mr-4">
                      <button
                        onClick={() => !isLiveFeedActive && setCameraMode("webcam")}
                        disabled={isLiveFeedActive}
                        className={`px-3 py-1 text-sm font-medium rounded flex items-center gap-2 transition-colors ${cameraMode === "webcam" ? "bg-white shadow text-blue-600" : "text-gray-500 hover:text-gray-700"
                          } ${isLiveFeedActive ? "opacity-50 cursor-not-allowed" : ""}`}
                      >
                        <Camera size={16} /> Webcam
                      </button>
                      <button
                        onClick={() => !isLiveFeedActive && setCameraMode("ip")}
                        disabled={isLiveFeedActive}
                        className={`px-3 py-1 text-sm font-medium rounded flex items-center gap-2 transition-colors ${cameraMode === "ip" ? "bg-white shadow text-blue-600" : "text-gray-500 hover:text-gray-700"
                          } ${isLiveFeedActive ? "opacity-50 cursor-not-allowed" : ""}`}
                      >
                        <Globe size={16} /> IP Camera
                      </button>
                    </div>

                    {isLiveFeedActive && (
                      <span className="flex items-center gap-2 text-red-600">
                        <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse"></div>
                        <span className="text-sm font-medium">Live</span>
                      </span>
                    )}
                    {!isLiveFeedActive ? (
                      <button
                        onClick={startLiveRecognition}
                        disabled={!selectedSubject || (cameraMode === "ip" && !ipUrl)}
                        className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                      >
                        <span>▶</span>
                        Start Live Feed
                      </button>
                    ) : (
                      <button
                        onClick={stopLiveRecognition}
                        className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition-colors flex items-center gap-2"
                      >
                        <span>⏹</span>
                        Stop Live Feed
                      </button>
                    )}
                  </div>
                </div>

                {cameraMode === "ip" && !isLiveFeedActive && (
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">IP Camera URL</label>
                    <input
                      type="text"
                      value={ipUrl}
                      onChange={(e) => setIpUrl(e.target.value)}
                      placeholder="http://192.168.1.X:8080/video"
                      className="w-full px-4 py-2 border rounded-lg"
                    />
                    <p className="text-xs text-gray-500 mt-1">Camera must support CORS for processing.</p>
                  </div>
                )}

                <div className="relative bg-black rounded-lg overflow-hidden min-h-[480px] flex items-center justify-center">
                  {cameraMode === "webcam" ? (
                    <Webcam
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ facingMode: "user" }}
                      className="w-full h-full object-contain"
                      style={{ opacity: isLiveFeedActive ? 1 : 0.5 }}
                    />
                  ) : (
                    <div className="w-full h-full flex flex-col">
                      {ipUrl ? (
                        <img
                          ref={ipCameraRef}
                          src={ipUrl}
                          crossOrigin="anonymous"
                          className="w-full h-full object-contain"
                          alt="IP Stream"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                      ) : (
                        <div className="flex flex-col items-center justify-center h-full text-white opacity-50">
                          <Globe size={48} className="mb-2" />
                          <p>Enter IP Camera URL</p>
                        </div>
                      )}
                      {/* Fallback for error */}
                      <div className={`flex flex-col items-center justify-center h-full text-white opacity-50 absolute inset-0 ${ipError ? 'flex' : 'hidden'}`}>
                        <Globe size={48} className="mb-2" />
                        <p>Stream Error</p>
                      </div>
                    </div>
                  )}

                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    style={{ borderRadius: "0.5rem" }}
                  />
                  {!isLiveFeedActive && (
                    <div className="absolute inset-0 bg-black bg-opacity-40 rounded-lg flex items-center justify-center">
                      <div className="text-center text-white">
                        <p className="text-lg font-semibold mb-2">Live Feed Stopped</p>
                        <p className="text-sm">Click "Start Live Feed" to begin recognition</p>
                      </div>
                    </div>
                  )}
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  {isLiveFeedActive
                    ? "Students detected will be automatically marked for attendance"
                    : "Select a subject and click 'Start Live Feed' to begin"}
                </p>
              </div>
            </div>

            {/* Recognized Students List */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Detected Students</h2>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {recognizedStudents.length === 0 ? (
                  <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                    No students detected yet
                  </p>
                ) : (
                  recognizedStudents.map((item, idx) => (
                    <div
                      key={`${item.student._id}-${idx}`}
                      className="p-3 border dark:border-gray-600 rounded-lg bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-semibold text-gray-800 dark:text-white">
                            {item.student.name}
                          </p>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            Roll: {item.student.rollNumber}
                          </p>
                          <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                            ✓ Auto-marked
                          </p>
                        </div>
                        {settings?.allowManualAttendance && (
                          <button
                            onClick={() => handleManualMark(item.student._id)}
                            className="px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
                          >
                            Mark
                          </button>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "attendance" && (
          <div className="space-y-6">
            {/* Manual Entry Button */}
            {settings?.allowManualAttendance && subjects.length > 0 && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">Manual Attendance Entry</h2>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setShowBulkAttendance(true)}
                      className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
                    >
                      Bulk Mark Attendance
                    </button>
                    <button
                      onClick={() => setShowManualEntry(true)}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
                    >
                      + Mark Single Attendance
                    </button>
                  </div>
                </div>
                <p className="text-sm text-gray-600">
                  Mark attendance for individual students or bulk mark for all enrolled students in a course
                </p>
              </div>
            )}

            {subjects.length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-8 text-center">
                <p className="text-gray-500 dark:text-gray-400 text-lg">
                  No subjects assigned. Please contact admin to assign subjects.
                </p>
              </div>
            ) : Object.keys(attendance).length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-8 text-center">
                <p className="text-gray-500 dark:text-gray-400 text-lg">No attendance records found</p>
              </div>
            ) : (
              Object.entries(attendance).map(([subjectName, records]) => (
                <div key={subjectName} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
                  <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">{subjectName}</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-2">Student Name</th>
                          <th className="text-left p-2">Roll Number</th>
                          <th className="text-left p-2">Date</th>
                          <th className="text-left p-2">Time</th>
                          <th className="text-left p-2">Status</th>
                          <th className="text-left p-2">Marked By</th>
                        </tr>
                      </thead>
                      <tbody>
                        {records.map((record, idx) => (
                          <tr key={idx} className="border-b hover:bg-gray-50">
                            <td className="p-2">{record.student.name}</td>
                            <td className="p-2">{record.student.rollNumber}</td>
                            <td className="p-2">
                              {new Date(record.date).toLocaleDateString("en-US", {
                                year: "numeric",
                                month: "short",
                                day: "numeric",
                              })}
                            </td>
                            <td className="p-2">
                              {new Date(record.date).toLocaleTimeString("en-US", {
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                            </td>
                            <td className="p-2">
                              <span
                                className={`px-2 py-1 rounded text-xs font-medium ${(record.status || "present") === "present"
                                  ? "bg-green-100 text-green-700"
                                  : (record.status || "present") === "absent"
                                    ? "bg-red-100 text-red-700"
                                    : "bg-yellow-100 text-yellow-700"
                                  }`}
                              >
                                {(record.status || "present") === "present" ? "Present" : (record.status || "present") === "absent" ? "Absent" : "Leave"}
                              </span>
                            </td>
                            <td className="p-2">
                              <span
                                className={`px-2 py-1 rounded text-xs font-medium ${record.markedBy === "auto"
                                  ? "bg-green-100 text-green-700"
                                  : "bg-blue-100 text-blue-700"
                                  }`}
                              >
                                {record.markedBy === "auto" ? "Auto" : "Manual"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))
            )}

            {/* Manual Entry Modal */}
            {showManualEntry && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-md w-full p-6">
                  <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Mark Attendance Manually</h2>
                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!manualEntryForm.studentId || !manualEntryForm.subjectId) {
                        alert("Please select student and subject");
                        return;
                      }

                      const token = localStorage.getItem("token");
                      try {
                        const res = await fetch("http://localhost:5000/api/attendance/mark", {
                          method: "POST",
                          headers: {
                            "Content-Type": "application/json",
                            Authorization: `Bearer ${token}`,
                          },
                          body: JSON.stringify({
                            studentId: manualEntryForm.studentId,
                            subjectId: manualEntryForm.subjectId,
                            date: manualEntryForm.date,
                            markedBy: "manual",
                            status: manualEntryForm.status,
                          }),
                        });

                        const data = await res.json();
                        if (res.ok) {
                          alert("Attendance marked successfully!");
                          setShowManualEntry(false);
                          setManualEntryForm({
                            studentId: "",
                            subjectId: subjects.length > 0 ? subjects[0]._id : "",
                            date: new Date().toISOString().split('T')[0],
                          });
                          loadData(); // Refresh attendance data
                        } else {
                          alert(data.error || data.message || "Failed to mark attendance");
                        }
                      } catch (err) {
                        console.error("Manual entry error:", err);
                        alert("Failed to mark attendance");
                      }
                    }}
                    className="space-y-4"
                  >
                    <div>
                      <label className="block text-sm font-medium mb-2">Student</label>
                      <select
                        required
                        value={manualEntryForm.studentId}
                        onChange={(e) =>
                          setManualEntryForm({ ...manualEntryForm, studentId: e.target.value })
                        }
                        className="w-full px-4 py-2 border rounded-lg"
                      >
                        <option value="">Select Student</option>
                        {students.map((student) => (
                          <option key={student._id} value={student._id}>
                            {student.name} (Roll: {student.rollNumber})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">Subject</label>
                      <select
                        required
                        value={manualEntryForm.subjectId}
                        onChange={(e) =>
                          setManualEntryForm({ ...manualEntryForm, subjectId: e.target.value })
                        }
                        className="w-full px-4 py-2 border rounded-lg"
                      >
                        <option value="">Select Subject</option>
                        {subjects.map((subject) => (
                          <option key={subject._id} value={subject._id}>
                            {subject.name} ({subject.code})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">Date</label>
                      <input
                        type="date"
                        required
                        value={manualEntryForm.date}
                        onChange={(e) =>
                          setManualEntryForm({ ...manualEntryForm, date: e.target.value })
                        }
                        className="w-full px-4 py-2 border rounded-lg"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">Status</label>
                      <select
                        required
                        value={manualEntryForm.status}
                        onChange={(e) =>
                          setManualEntryForm({ ...manualEntryForm, status: e.target.value })
                        }
                        className="w-full px-4 py-2 border rounded-lg"
                      >
                        <option value="present">Present</option>
                        <option value="absent">Absent</option>
                        <option value="leave">Leave</option>
                      </select>
                    </div>
                    <div className="flex gap-4 pt-4">
                      <button
                        type="submit"
                        className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                      >
                        Mark Attendance
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setShowManualEntry(false);
                          setManualEntryForm({
                            studentId: "",
                            subjectId: subjects.length > 0 ? subjects[0]._id : "",
                            date: new Date().toISOString().split('T')[0],
                            status: "present",
                          });
                        }}
                        className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                      >
                        Cancel
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}

            {/* Bulk Attendance Modal */}
            {showBulkAttendance && settings?.allowManualAttendance && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-4xl w-full p-6 my-8">
                  <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Bulk Mark Attendance</h2>
                  <div className="space-y-4 mb-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">Subject</label>
                      <select
                        required
                        value={bulkAttendanceForm.subjectId}
                        onChange={(e) => {
                          const subjectId = e.target.value;
                          setBulkAttendanceForm({
                            ...bulkAttendanceForm,
                            subjectId,
                            attendances: (enrolledStudents[subjectId] || []).map((s) => ({
                              studentId: s._id,
                              status: "present",
                            })),
                          });
                        }}
                        className="w-full px-4 py-2 border rounded-lg"
                      >
                        <option value="">Select Subject</option>
                        {subjects.map((subject) => (
                          <option key={subject._id} value={subject._id}>
                            {subject.name} ({subject.code})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">Date</label>
                      <input
                        type="date"
                        required
                        value={bulkAttendanceForm.date}
                        onChange={(e) =>
                          setBulkAttendanceForm({ ...bulkAttendanceForm, date: e.target.value })
                        }
                        className="w-full px-4 py-2 border rounded-lg"
                      />
                    </div>
                  </div>

                  {bulkAttendanceForm.subjectId && enrolledStudents[bulkAttendanceForm.subjectId] && (
                    <div className="border rounded-lg p-4 max-h-96 overflow-y-auto mb-4">
                      <h3 className="font-semibold mb-3">
                        Enrolled Students ({enrolledStudents[bulkAttendanceForm.subjectId].length})
                      </h3>
                      <div className="space-y-2">
                        {enrolledStudents[bulkAttendanceForm.subjectId].map((student) => {
                          const attendance = bulkAttendanceForm.attendances.find(
                            (a) => a.studentId === student._id
                          ) || { studentId: student._id, status: "present" };
                          return (
                            <div
                              key={student._id}
                              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg gap-4"
                            >
                              <div className="flex-1">
                                <p className="font-medium text-gray-800">{student.name}</p>
                                <p className="text-sm text-gray-600">Roll: {student.rollNumber}</p>
                              </div>
                              <select
                                value={attendance.status}
                                onChange={(e) => {
                                  const updated = bulkAttendanceForm.attendances.map((a) =>
                                    a.studentId === student._id
                                      ? { ...a, status: e.target.value }
                                      : a
                                  );
                                  if (!updated.find((a) => a.studentId === student._id)) {
                                    updated.push({
                                      studentId: student._id,
                                      status: e.target.value,
                                    });
                                  }
                                  setBulkAttendanceForm({
                                    ...bulkAttendanceForm,
                                    attendances: updated,
                                  });
                                }}
                                className="px-4 py-2 border rounded-lg bg-white min-w-[120px]"
                              >
                                <option value="present">Present</option>
                                <option value="absent">Absent</option>
                                <option value="leave">Leave</option>
                              </select>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-4 pt-4">
                    <button
                      onClick={async () => {
                        if (!bulkAttendanceForm.subjectId || bulkAttendanceForm.attendances.length === 0) {
                          alert("Please select a subject and ensure students are loaded");
                          return;
                        }

                        const token = localStorage.getItem("token");
                        try {
                          const res = await fetch("http://localhost:5000/api/attendance/bulk", {
                            method: "POST",
                            headers: {
                              "Content-Type": "application/json",
                              Authorization: `Bearer ${token}`,
                            },
                            body: JSON.stringify({
                              subjectId: bulkAttendanceForm.subjectId,
                              date: bulkAttendanceForm.date,
                              attendances: bulkAttendanceForm.attendances,
                            }),
                          });

                          const data = await res.json();
                          if (res.ok) {
                            alert(
                              `Successfully processed ${data.results.length} attendances${data.errors && data.errors.length > 0
                                ? `. ${data.errors.length} errors occurred.`
                                : ""
                              }`
                            );
                            setShowBulkAttendance(false);
                            setBulkAttendanceForm({
                              subjectId: "",
                              date: new Date().toISOString().split('T')[0],
                              attendances: [],
                            });
                            loadData();
                          } else {
                            alert(data.error || "Failed to mark attendance");
                          }
                        } catch (err) {
                          console.error("Bulk attendance error:", err);
                          alert("Failed to mark attendance");
                        }
                      }}
                      className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                    >
                      Mark All Attendance
                    </button>
                    <button
                      onClick={() => {
                        setShowBulkAttendance(false);
                        setBulkAttendanceForm({
                          subjectId: "",
                          date: new Date().toISOString().split('T')[0],
                          attendances: [],
                        });
                      }}
                      className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "enrollments" && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">Pending Enrollment Requests</h2>
            {enrollments.filter((e) => e.status === "pending").length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-8 text-center">
                <p className="text-gray-500 dark:text-gray-400 text-lg">No pending enrollment requests</p>
              </div>
            ) : (
              <div className="space-y-4">
                {enrollments
                  .filter((e) => e.status === "pending")
                  .map((enrollment) => (
                    <div key={enrollment._id} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="flex items-center gap-4 mb-2">
                            <h3 className="text-xl font-bold text-gray-800 dark:text-white">
                              {enrollment.student.name}
                            </h3>
                            <span className="px-3 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded-full text-xs font-medium">
                              Pending
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300 mb-1">
                            Roll Number: {enrollment.student.rollNumber}
                          </p>
                          <p className="text-sm text-gray-600 dark:text-gray-300 mb-1">
                            Email: {enrollment.student.email}
                          </p>
                          <p className="text-lg font-semibold text-gray-800 dark:text-white mt-3">
                            Course: {enrollment.subject.name} ({enrollment.subject.code})
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            Requested: {new Date(enrollment.requestedAt).toLocaleString()}
                          </p>
                        </div>
                        <div className="flex gap-3 ml-4">
                          <button
                            onClick={async () => {
                              const token = localStorage.getItem("token");
                              try {
                                const res = await fetch(
                                  `http://localhost:5000/api/enrollments/${enrollment._id}`,
                                  {
                                    method: "PUT",
                                    headers: {
                                      "Content-Type": "application/json",
                                      Authorization: `Bearer ${token}`,
                                    },
                                    body: JSON.stringify({ status: "approved" }),
                                  }
                                );

                                // Check content type before parsing
                                const contentType = res.headers.get("content-type");
                                let data;

                                if (contentType && contentType.includes("application/json")) {
                                  data = await res.json();
                                } else {
                                  const text = await res.text();
                                  console.error("Non-JSON response:", text);
                                  alert(`Failed to approve enrollment: ${res.status} ${res.statusText}`);
                                  return;
                                }

                                if (res.ok) {
                                  alert("Enrollment approved successfully!");
                                  loadData();
                                } else {
                                  console.error("Approve enrollment error:", data);
                                  alert(data.error || data.message || "Failed to approve enrollment");
                                }
                              } catch (err) {
                                console.error("Approve error:", err);
                                alert(`Failed to approve enrollment: ${err.message || "Please try again"}`);
                              }
                            }}
                            className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium transition-colors"
                          >
                            Approve
                          </button>
                          <button
                            onClick={async () => {
                              const token = localStorage.getItem("token");
                              try {
                                const res = await fetch(
                                  `http://localhost:5000/api/enrollments/${enrollment._id}`,
                                  {
                                    method: "PUT",
                                    headers: {
                                      "Content-Type": "application/json",
                                      Authorization: `Bearer ${token}`,
                                    },
                                    body: JSON.stringify({ status: "rejected" }),
                                  }
                                );

                                // Check content type before parsing
                                const contentType = res.headers.get("content-type");
                                let data;

                                if (contentType && contentType.includes("application/json")) {
                                  data = await res.json();
                                } else {
                                  const text = await res.text();
                                  console.error("Non-JSON response:", text);
                                  alert(`Failed to reject enrollment: ${res.status} ${res.statusText}`);
                                  return;
                                }

                                if (res.ok) {
                                  alert("Enrollment rejected");
                                  loadData();
                                } else {
                                  console.error("Reject enrollment error:", data);
                                  alert(data.error || data.message || "Failed to reject enrollment");
                                }
                              } catch (err) {
                                console.error("Reject error:", err);
                                alert(`Failed to reject enrollment: ${err.message || "Please try again"}`);
                              }
                            }}
                            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition-colors"
                          >
                            Reject
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>
        )}
        </div>
      </main>
    </div>
  );
}

