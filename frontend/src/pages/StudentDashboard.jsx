import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import StudentSidebar from "../components/StudentSidebar";

export default function StudentDashboard() {
  const [user, setUser] = useState(null);
  const [enrollments, setEnrollments] = useState([]);
  const [availableSubjects, setAvailableSubjects] = useState([]);
  const [courseAttendance, setCourseAttendance] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [activeTab, setActiveTab] = useState("enrolled"); // "enrolled" or "browse"
  const [loading, setLoading] = useState(true);
  const [showCourseAttendance, setShowCourseAttendance] = useState(false);
  const [selectedSubjectForEnroll, setSelectedSubjectForEnroll] = useState(null); // For teacher selection modal
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    const userData = JSON.parse(localStorage.getItem("user") || "{}");

    if (!token || userData.role !== "student") {
      navigate("/login");
      return;
    }

    setUser(userData);
    loadData();
  }, [navigate]);

  const loadData = async () => {
    const token = localStorage.getItem("token");
    setLoading(true);
    try {
      const [enrollmentsRes, subjectsRes] = await Promise.all([
        fetch("http://localhost:5000/api/enrollments/student", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/subjects/with-teachers", {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      if (enrollmentsRes.ok) {
        const enrollmentsData = await enrollmentsRes.json();
        // Filter out any invalid enrollments (those without subject)
        // Note: teacher field is required for new enrollments, but we allow old ones temporarily
        const validEnrollments = enrollmentsData.filter(
          (e) => e && e.subject
        );
        setEnrollments(validEnrollments);
      }

      if (subjectsRes.ok) {
        const subjectsData = await subjectsRes.json();
        // Filter out subjects with no teachers assigned (students can only enroll if teachers are assigned)
        const subjectsWithTeachers = subjectsData.filter(
          (subject) => subject.teachers && subject.teachers.length > 0
        );
        setAvailableSubjects(subjectsWithTeachers);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleEnroll = async (subjectId, teacherId) => {
    const token = localStorage.getItem("token");

    if (!token) {
      alert("You must be logged in to enroll in courses. Please login again.");
      navigate("/login");
      return;
    }

    if (!teacherId) {
      alert("Please select a teacher for this course.");
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/api/enrollments", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ subjectId, teacherId }),
      });

      // Check content type before parsing
      const contentType = res.headers.get("content-type");
      let data;

      if (contentType && contentType.includes("application/json")) {
        try {
          data = await res.json();
        } catch (parseErr) {
          console.error("JSON parse error:", parseErr);
          throw new Error("Failed to parse server response");
        }
      } else {
        // Response is not JSON (likely HTML error page or network error)
        const text = await res.text();
        console.error("Non-JSON response received:", {
          status: res.status,
          statusText: res.statusText,
          contentType: contentType,
          preview: text.substring(0, 200)
        });

        if (res.status === 0) {
          throw new Error("Network error: Unable to connect to server. Please ensure the backend server is running.");
        } else {
          throw new Error(`Server returned non-JSON response: ${res.status} ${res.statusText}`);
        }
      }

      if (res.ok) {
        const message = data.message || "Enrollment request submitted successfully! Waiting for teacher approval.";
        alert(message);
        setSelectedSubjectForEnroll(null); // Close teacher selection modal
        loadData(); // Refresh enrollments
      } else {
        const errorMsg = data.error || data.message || `Server error: ${res.status}`;
        alert(`Enrollment failed: ${errorMsg}`);
      }
    } catch (err) {
      console.error("Enrollment error:", err);

      if (err.message && (err.message.includes("JSON") || err.message.includes("Network") || err.message.includes("fetch"))) {
        alert("Enrollment failed: Unable to connect to server. Please ensure the backend server is running on http://localhost:5000 and try again.");
      } else {
        alert(`Enrollment failed: ${err.message || "Please try again."}`);
      }
    }
  };

  const handleViewAttendance = async (enrollment) => {
    if (enrollment.status !== "approved") {
      alert("You can only view attendance for approved courses.");
      return;
    }

    const token = localStorage.getItem("token");

    // Extract studentId - handle both populated (object with _id) and non-populated (string/ObjectId)
    let studentId;
    if (user.studentId && typeof user.studentId === 'object' && user.studentId._id) {
      studentId = user.studentId._id.toString();
    } else if (user.studentId) {
      studentId = user.studentId.toString();
    } else {
      alert("Student ID not found. Please contact admin.");
      return;
    }

    // Extract subjectId - handle both populated and non-populated
    let subjectId;
    if (enrollment.subject && typeof enrollment.subject === 'object' && enrollment.subject._id) {
      subjectId = enrollment.subject._id.toString();
    } else if (enrollment.subject) {
      subjectId = enrollment.subject.toString();
    } else {
      alert("Subject ID not found.");
      return;
    }

    try {
      const res = await fetch(
        `http://localhost:5000/api/students/${studentId}/attendance/course/${subjectId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      // Check content type before parsing
      const contentType = res.headers.get("content-type");
      let data;

      if (contentType && contentType.includes("application/json")) {
        try {
          data = await res.json();
        } catch (parseErr) {
          console.error("JSON parse error:", parseErr);
          throw new Error("Failed to parse server response");
        }
      } else {
        const text = await res.text();
        console.error("Non-JSON response:", { status: res.status, text: text.substring(0, 200) });
        throw new Error(`Server returned non-JSON response: ${res.status}`);
      }

      if (res.ok) {
        const attendanceData = Array.isArray(data) ? data : [];
        setCourseAttendance(attendanceData);
        setSelectedCourse(enrollment);
        setShowCourseAttendance(true);

        if (attendanceData.length === 0) {
          // Still show the view, but user will see "No attendance records"
          console.log("No attendance records found for this course");
        }
      } else {
        const errorMsg = data.error || data.message || `Failed to load attendance: ${res.status}`;
        alert(errorMsg);
        console.error("Attendance fetch error:", { status: res.status, error: data });
      }
    } catch (err) {
      console.error("Attendance fetch error:", err);
      if (err.message && err.message.includes("JSON")) {
        alert("Failed to load attendance: Server returned invalid response. Please try again.");
      } else if (err.message && err.message.includes("fetch")) {
        alert("Failed to load attendance: Unable to connect to server. Please ensure the backend server is running.");
      } else {
        alert(`Failed to load attendance: ${err.message || "Please try again."}`);
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  const getStatusBadge = (status) => {
    const badges = {
      pending: "bg-yellow-100 text-yellow-700",
      approved: "bg-green-100 text-green-700",
      rejected: "bg-red-100 text-red-700",
    };
    return badges[status] || "bg-gray-100 text-gray-700";
  };

  // Helper function to normalize an ID (handles objects, strings, undefined)
  const normalizeId = (id) => {
    if (!id) return null;
    if (typeof id === 'string') return id;
    if (typeof id === 'object' && id._id) return id._id.toString();
    if (typeof id.toString === 'function') return id.toString();
    return String(id);
  };

  const isEnrolled = (subjectId, teacherId = null) => {
    if (!subjectId || enrollments.length === 0) return false;

    const normalizedSubjectId = normalizeId(subjectId);

    if (teacherId) {
      // Check if enrolled with a specific teacher
      const normalizedTeacherId = normalizeId(teacherId);
      if (!normalizedTeacherId) return false; // Invalid teacherId

      return enrollments.some((e) => {
        if (!e || !e.subject) return false;

        const enrollmentSubjectId = normalizeId(e.subject);
        if (!enrollmentSubjectId) return false;

        const subjectMatch = enrollmentSubjectId === normalizedSubjectId;
        if (!subjectMatch) return false;

        // Check teacher match - teacher field is required for new enrollments
        if (!e.teacher) return false; // No teacher means this enrollment doesn't match
        const enrollmentTeacherId = normalizeId(e.teacher);
        if (!enrollmentTeacherId) return false;

        return enrollmentTeacherId === normalizedTeacherId;
      });
    }

    // Check if enrolled with any teacher for this subject
    return enrollments.some((e) => {
      if (!e || !e.subject) return false;
      const enrollmentSubjectId = normalizeId(e.subject);
      return enrollmentSubjectId === normalizedSubjectId;
    });
  };

  const getEnrollmentStatus = (subjectId, teacherId = null) => {
    if (!subjectId || enrollments.length === 0) return null;

    const normalizedSubjectId = normalizeId(subjectId);

    const enrollment = enrollments.find((e) => {
      if (!e || !e.subject) return false;

      const enrollmentSubjectId = normalizeId(e.subject);
      const subjectMatch = enrollmentSubjectId === normalizedSubjectId;

      if (!subjectMatch) return false;

      if (teacherId) {
        if (!e.teacher) return false; // No teacher means this enrollment doesn't match
        const normalizedTeacherId = normalizeId(teacherId);
        if (!normalizedTeacherId) return false; // Invalid teacherId

        const enrollmentTeacherId = normalizeId(e.teacher);
        if (!enrollmentTeacherId) return false;

        return enrollmentTeacherId === normalizedTeacherId;
      }

      // If no teacherId specified, return true if subject matches (for backward compatibility)
      return true;
    });

    return enrollment ? enrollment.status : null;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-xl text-gray-600 dark:text-gray-400">Loading...</div>
      </div>
    );
  }

  if (showCourseAttendance && selectedCourse) {
    const subject = selectedCourse.subject;
    const subjectName = subject?.name || (typeof subject === 'object' && subject?.name) || "Unknown Subject";
    const subjectCode = subject?.code || (typeof subject === 'object' && subject?.code) || "";

    return (
      <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
        <StudentSidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          enrollmentsCount={enrollments.length}
          availableCoursesCount={availableSubjects.length}
        />

        <main className="flex-1 overflow-y-auto flex flex-col">
          <nav className="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700 h-16 flex items-center justify-between px-6 sticky top-0 z-10">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowCourseAttendance(false)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-sm"
              >
                ← Back to Courses
              </button>
              <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
                Attendance: {subjectName}
              </h2>
            </div>
            <div className="flex items-center gap-4">
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
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6 mb-6">
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
                {subjectName} {subjectCode && `(${subjectCode})`}
              </h2>
              <p className="text-gray-600 dark:text-gray-300">
                {courseAttendance.length} attendance record
                {courseAttendance.length !== 1 ? "s" : ""}
              </p>
            </div>

            {courseAttendance.length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-8 text-center">
                <p className="text-gray-500 dark:text-gray-400 text-lg">No attendance records found</p>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Time
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Marked By
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {courseAttendance
                        .sort((a, b) => new Date(b.date) - new Date(a.date))
                        .map((record, idx) => (
                          <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                              {new Date(record.date).toLocaleDateString("en-US", {
                                weekday: "long",
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                              })}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-300">
                              {new Date(record.date).toLocaleTimeString("en-US", {
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span
                                className={`px-3 py-1 rounded-full text-xs font-medium ${record.status === "present"
                                  ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
                                  : record.status === "absent"
                                    ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400"
                                    : "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400"
                                  }`}
                              >
                                {record.status === "present"
                                  ? "Present"
                                  : record.status === "absent"
                                    ? "Absent"
                                    : "Leave"}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-300">
                              <span
                                className={`px-2 py-1 rounded text-xs font-medium ${record.markedBy === "auto"
                                  ? "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                                  : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
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
            )}
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
      <StudentSidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        enrollmentsCount={enrollments.length}
        availableCoursesCount={availableSubjects.length}
      />

      <main className="flex-1 overflow-y-auto flex flex-col">
        {/* Topbar */}
        <nav className="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700 h-16 flex items-center justify-between px-6 sticky top-0 z-10">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Student Dashboard</h2>
          <div className="flex items-center gap-4">
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

          {/* Enrolled Courses Tab */}
          {activeTab === "enrolled" && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
                My Enrolled Courses
              </h2>
              {enrollments.length === 0 ? (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-8 text-center">
                  <p className="text-gray-500 dark:text-gray-400 text-lg mb-4">
                    You haven't enrolled in any courses yet.
                  </p>
                  <button
                    onClick={() => setActiveTab("browse")}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Browse Courses
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {enrollments.map((enrollment) => {
                    const subject = enrollment.subject;
                    return (
                      <div
                        key={enrollment._id}
                        className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6 hover:shadow-lg transition-shadow"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <h3 className="text-xl font-bold text-gray-800 dark:text-white">
                              {subject.name}
                            </h3>
                            <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                              Code: {subject.code}
                            </p>
                            {enrollment.teacher && (
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                Teacher: {enrollment.teacher.name || enrollment.teacher.email || "N/A"}
                              </p>
                            )}
                          </div>
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusBadge(
                              enrollment.status
                            )}`}
                          >
                            {enrollment.status.charAt(0).toUpperCase() +
                              enrollment.status.slice(1)}
                          </span>
                        </div>

                        <div className="border-t dark:border-gray-700 pt-4 space-y-2">
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Requested:{" "}
                            {new Date(enrollment.requestedAt || enrollment.createdAt).toLocaleDateString()}
                          </p>
                          {enrollment.reviewedAt && (
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              Reviewed:{" "}
                              {new Date(enrollment.reviewedAt).toLocaleDateString()}
                            </p>
                          )}

                          {enrollment.status === "approved" && (
                            <button
                              onClick={() => handleViewAttendance(enrollment)}
                              className="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                              View Attendance
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Browse Courses Tab */}
          {activeTab === "browse" && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">
                Available Courses
              </h2>
              {availableSubjects.length === 0 ? (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-8 text-center">
                  <p className="text-gray-500 dark:text-gray-400 text-lg">
                    No courses with assigned teachers available yet.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {availableSubjects.map((subject) => {
                    if (!subject || !subject._id) return null;

                    return (
                      <div
                        key={subject._id}
                        className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6 hover:shadow-lg transition-shadow"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <h3 className="text-xl font-bold text-gray-800 dark:text-white">
                              {subject.name}
                            </h3>
                            <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                              Code: {subject.code}
                            </p>
                            {subject.teachers && subject.teachers.length > 0 && (
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                {subject.teachers.length === 1
                                  ? `Teacher: ${subject.teachers[0].name}`
                                  : `${subject.teachers.length} teachers available`}
                              </p>
                            )}
                          </div>
                        </div>

                        {subject.teachers && subject.teachers.length > 0 ? (
                          <div className="border-t dark:border-gray-700 pt-4 space-y-2">
                            {subject.teachers.map((teacher) => {
                              if (!teacher || !teacher._id) return null;

                              const teacherEnrolled = isEnrolled(subject._id, teacher._id);
                              const teacherStatus = getEnrollmentStatus(subject._id, teacher._id);

                              return (
                                <div
                                  key={teacher._id}
                                  className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg border dark:border-gray-600"
                                >
                                  <div className="flex justify-between items-center mb-2">
                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                      {teacher.name || teacher.email || "Unknown Teacher"}
                                    </span>
                                    {teacherEnrolled && teacherStatus && (
                                      <span
                                        className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(
                                          teacherStatus
                                        )}`}
                                      >
                                        {teacherStatus.charAt(0).toUpperCase() + teacherStatus.slice(1)}
                                      </span>
                                    )}
                                  </div>

                                  {!teacherEnrolled ? (
                                    <button
                                      onClick={() => {
                                        if (subject._id && teacher._id) {
                                          handleEnroll(subject._id, teacher._id);
                                        }
                                      }}
                                      className="w-full px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
                                    >
                                      Enroll with {(teacher.name || teacher.email || "Teacher").split(' ')[0]}
                                    </button>
                                  ) : (
                                    <div className="text-center">
                                      {teacherStatus === "approved" && (
                                        <button
                                          onClick={() => {
                                            const enrollment = enrollments.find((e) => {
                                              if (!e || !e.subject) return false;
                                              const enrollmentSubjectId = normalizeId(e.subject);
                                              const currentSubjectId = normalizeId(subject._id);
                                              const subjectMatch = enrollmentSubjectId === currentSubjectId;

                                              if (!subjectMatch) return false;

                                              if (!e.teacher) return false;
                                              const enrollmentTeacherId = normalizeId(e.teacher);
                                              const currentTeacherId = normalizeId(teacher._id);
                                              return enrollmentTeacherId === currentTeacherId;
                                            });
                                            if (enrollment) {
                                              handleViewAttendance(enrollment);
                                            }
                                          }}
                                          className="w-full px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
                                        >
                                          View Attendance
                                        </button>
                                      )}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="border-t dark:border-gray-700 pt-4 text-center">
                            <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                              No teachers assigned to this course yet.
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
