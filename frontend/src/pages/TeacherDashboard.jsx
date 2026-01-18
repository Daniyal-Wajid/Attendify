import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";
import axios from "axios";

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
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const recognitionIntervalRef = useRef(null);
  const navigate = useNavigate();

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
      // Cleanup on unmount - stop live feed if active
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
        recognitionIntervalRef.current = null;
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
    if (isLiveFeedActive) {
      console.log("Live feed is already running");
      return;
    }

    if (!selectedSubject) {
      alert("Please select a subject first");
      return;
    }

    setIsLiveFeedActive(true);
    recognitionIntervalRef.current = setInterval(async () => {
      if (!webcamRef.current || !selectedSubject) return;

      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      try {
        const token = localStorage.getItem("token");
        const res = await axios.post(
          "http://localhost:5000/api/attendance/recognize-live",
          { imageBase64: imageSrc },
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );

        // Always update detected faces for drawing
        if (res.data.faces && Array.isArray(res.data.faces)) {
          setDetectedFaces(res.data.faces);
        }

        if (res.data.recognized && res.data.student && res.data.faceBox) {
          const now = Date.now();
          const studentId = res.data.student._id;
          
          // Only keep the most recent recognition for each student
          // Remove old recognitions for this student to prevent duplicates
          setRecognizedStudents((prev) => {
            // Remove all previous entries for this student
            const filtered = prev.filter((s) => s.student._id !== studentId);
            // Add the new recognition
            return [
              ...filtered,
              {
                student: res.data.student,
                faceBox: res.data.faceBox,
                timestamp: now,
              },
            ];
          });

          // Auto-mark attendance (will check for duplicates on backend)
          markAttendance(studentId, "auto");
        } else if (res.data.faces && res.data.faces.length > 0) {
          // Face detected but not recognized - clear recognized students
          // Only show gray boxes for unrecognized faces
          setRecognizedStudents([]);
        } else {
          // No faces detected - clear everything
          setDetectedFaces([]);
          setRecognizedStudents([]);
        }
      } catch (err) {
        console.error("Recognition error:", err);
      }
    }, 1000); // Check every second
  };

  const stopLiveRecognition = () => {
    if (recognitionIntervalRef.current) {
      clearInterval(recognitionIntervalRef.current);
      recognitionIntervalRef.current = null;
    }
    setIsLiveFeedActive(false);
    // Clear recognized students and detected faces when stopping
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
    if (!canvas || !webcamRef.current) return;

    const video = webcamRef.current.video;
    if (!video) return;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Only show the most recently recognized student (one at a time)
      if (recognizedStudents.length > 0) {
        // Get the most recent recognition
        const latest = recognizedStudents.reduce((latest, current) => 
          current.timestamp > latest.timestamp ? current : latest
        );
        
        if (latest.faceBox) {
          const { x, y, w, h } = latest.faceBox;
          
          // Draw green box
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);
          
          // Draw name label with background for better visibility
          const name = latest.student.name;
          ctx.font = "bold 16px Arial";
          const textWidth = ctx.measureText(name).width;
          
          // Draw background rectangle for text
          ctx.fillStyle = "rgba(16, 185, 129, 0.8)";
          ctx.fillRect(x, y - 25, textWidth + 10, 20);
          
          // Draw text
          ctx.fillStyle = "#ffffff";
          ctx.fillText(name, x + 5, y - 8);
        }
      }

      // Draw all detected faces (unrecognized) with gray boxes
      // Only show if no student is currently recognized
      if (recognizedStudents.length === 0) {
        detectedFaces.forEach((face) => {
          if (!face || typeof face.x !== 'number' || typeof face.y !== 'number') return;
          
          ctx.strokeStyle = "#9ca3af";
          ctx.lineWidth = 2;
          ctx.strokeRect(face.x, face.y, face.w || 100, face.h || 100);
        });
      }

      requestAnimationFrame(draw);
    };

    draw();
  }, [recognizedStudents, detectedFaces]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-xl font-bold text-gray-800">Teacher Dashboard</h1>
            <div className="flex items-center gap-4">
              <select
                value={selectedSubject}
                onChange={(e) => setSelectedSubject(e.target.value)}
                className="px-4 py-2 border rounded-lg"
              >
                {subjects.map((subject) => (
                  <option key={subject._id} value={subject._id}>
                    {subject.name} ({subject.code})
                  </option>
                ))}
              </select>
              <span className="text-gray-600">{user?.name || user?.email}</span>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex gap-4">
            <button
              onClick={() => setActiveTab("camera")}
              className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                activeTab === "camera"
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              Live Camera
            </button>
            <button
              onClick={() => {
                setActiveTab("attendance");
                loadData(); // Refresh attendance when switching tabs
              }}
              className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                activeTab === "attendance"
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              View Attendance
            </button>
            <button
              onClick={() => {
                setActiveTab("enrollments");
                loadData(); // Refresh enrollments when switching tabs
              }}
              className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                activeTab === "enrollments"
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              Enrollments ({enrollments.filter(e => e.status === "pending").length})
            </button>
          </nav>
        </div>

        {activeTab === "camera" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Live Camera Feed */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-xl shadow p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Live Camera Feed</h2>
                  <div className="flex items-center gap-3">
                    {isLiveFeedActive && (
                      <span className="flex items-center gap-2 text-red-600">
                        <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse"></div>
                        <span className="text-sm font-medium">Live</span>
                      </span>
                    )}
                    {!isLiveFeedActive ? (
                      <button
                        onClick={startLiveRecognition}
                        disabled={!selectedSubject}
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
                <div className="relative">
                  <Webcam
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: "user" }}
                    className="w-full rounded-lg"
                    style={{ opacity: isLiveFeedActive ? 1 : 0.5 }}
                  />
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
            <div className="bg-white rounded-xl shadow p-6">
              <h2 className="text-xl font-bold mb-4">Detected Students</h2>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {recognizedStudents.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">
                    No students detected yet
                  </p>
                ) : (
                  recognizedStudents.map((item, idx) => (
                    <div
                      key={`${item.student._id}-${idx}`}
                      className="p-3 border rounded-lg bg-green-50 border-green-200"
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-semibold text-gray-800">
                            {item.student.name}
                          </p>
                          <p className="text-sm text-gray-600">
                            Roll: {item.student.rollNumber}
                          </p>
                          <p className="text-xs text-green-600 mt-1">
                            ✓ Auto-marked
                          </p>
                        </div>
                        {settings?.allowManualAttendance && (
                          <button
                            onClick={() => handleManualMark(item.student._id)}
                            className="px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
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
              <div className="bg-white rounded-xl shadow p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Manual Attendance Entry</h2>
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
              <div className="bg-white rounded-xl shadow p-8 text-center">
                <p className="text-gray-500 text-lg">
                  No subjects assigned. Please contact admin to assign subjects.
                </p>
              </div>
            ) : Object.keys(attendance).length === 0 ? (
              <div className="bg-white rounded-xl shadow p-8 text-center">
                <p className="text-gray-500 text-lg">No attendance records found</p>
              </div>
            ) : (
              Object.entries(attendance).map(([subjectName, records]) => (
                <div key={subjectName} className="bg-white rounded-xl shadow p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">{subjectName}</h3>
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
                                className={`px-2 py-1 rounded text-xs font-medium ${
                                  (record.status || "present") === "present"
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
                                className={`px-2 py-1 rounded text-xs font-medium ${
                                  record.markedBy === "auto"
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
                <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6">
                  <h2 className="text-2xl font-bold mb-4">Mark Attendance Manually</h2>
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
                <div className="bg-white rounded-xl shadow-xl max-w-4xl w-full p-6 my-8">
                  <h2 className="text-2xl font-bold mb-4">Bulk Mark Attendance</h2>
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
                              `Successfully processed ${data.results.length} attendances${
                                data.errors && data.errors.length > 0
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
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Pending Enrollment Requests</h2>
            {enrollments.filter((e) => e.status === "pending").length === 0 ? (
              <div className="bg-white rounded-xl shadow p-8 text-center">
                <p className="text-gray-500 text-lg">No pending enrollment requests</p>
              </div>
            ) : (
              <div className="space-y-4">
                {enrollments
                  .filter((e) => e.status === "pending")
                  .map((enrollment) => (
                    <div key={enrollment._id} className="bg-white rounded-xl shadow p-6">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="flex items-center gap-4 mb-2">
                            <h3 className="text-xl font-bold text-gray-800">
                              {enrollment.student.name}
                            </h3>
                            <span className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs font-medium">
                              Pending
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 mb-1">
                            Roll Number: {enrollment.student.rollNumber}
                          </p>
                          <p className="text-sm text-gray-600 mb-1">
                            Email: {enrollment.student.email}
                          </p>
                          <p className="text-lg font-semibold text-gray-800 mt-3">
                            Course: {enrollment.subject.name} ({enrollment.subject.code})
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
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
                            className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
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
                            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium"
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
    </div>
  );
}

