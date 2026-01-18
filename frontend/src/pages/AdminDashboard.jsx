import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import AdminSidebarSimple from "../components/AdminSidebarSimple";

export default function AdminDashboard() {
  const [user, setUser] = useState(null);
  const [students, setStudents] = useState([]);
  const [teachers, setTeachers] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [settings, setSettings] = useState(null);
  const [showRegisterTeacher, setShowRegisterTeacher] = useState(false);
  const [showAddSubject, setShowAddSubject] = useState(false);
  const [showAssignSubjects, setShowAssignSubjects] = useState(false);
  const [selectedTeacher, setSelectedTeacher] = useState(null);
  const [teacherForm, setTeacherForm] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [subjectForm, setSubjectForm] = useState({ name: "", code: "" });
  const [allAttendance, setAllAttendance] = useState([]);
  const [showAttendanceManagement, setShowAttendanceManagement] = useState(false);
  const [editingAttendance, setEditingAttendance] = useState(null);
  const [editForm, setEditForm] = useState({ studentId: "", subjectId: "", date: "", markedBy: "auto", status: "present" });
  const [studentSearch, setStudentSearch] = useState("");
  const [filteredStudents, setFilteredStudents] = useState([]);
  const [attendanceSearch, setAttendanceSearch] = useState("");
  const [filteredAttendance, setFilteredAttendance] = useState([]);
  const [resettingPassword, setResettingPassword] = useState(null); // {type: "student"|"teacher", id: ""}
  const [newPassword, setNewPassword] = useState("");
  const [activeSection, setActiveSection] = useState("overview");
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    const userData = JSON.parse(localStorage.getItem("user") || "{}");

    if (!token || userData.role !== "admin") {
      navigate("/login");
      return;
    }

    setUser(userData);
    loadData();
  }, [navigate]);

  const loadData = async () => {
    const token = localStorage.getItem("token");
    try {
      const [studentsRes, teachersRes, subjectsRes, settingsRes, attendanceRes] = await Promise.all([
        fetch("http://localhost:5000/api/students", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/teachers", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/subjects", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/settings", {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch("http://localhost:5000/api/attendance/all", {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      const studentsData = await studentsRes.json();
      setStudents(studentsData);
      setFilteredStudents(studentsData);
      setTeachers(await teachersRes.json());
      setSubjects(await subjectsRes.json());
      setSettings(await settingsRes.json());
      if (attendanceRes.ok) {
        const attendanceData = await attendanceRes.json();
        setAllAttendance(attendanceData);
        setFilteredAttendance(attendanceData);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Search students by name or roll number
  useEffect(() => {
    if (!studentSearch.trim()) {
      setFilteredStudents(students);
    } else {
      const searchLower = studentSearch.toLowerCase().trim();
      const filtered = students.filter(
        (student) =>
          student.name?.toLowerCase().includes(searchLower) ||
          student.rollNumber?.toLowerCase().includes(searchLower) ||
          student.email?.toLowerCase().includes(searchLower)
      );
      setFilteredStudents(filtered);
    }
  }, [studentSearch, students]);

  // Search attendance records
  useEffect(() => {
    if (!attendanceSearch.trim()) {
      setFilteredAttendance(allAttendance);
    } else {
      const searchLower = attendanceSearch.toLowerCase().trim();
      const filtered = allAttendance.filter(
        (record) =>
          record.student?.name?.toLowerCase().includes(searchLower) ||
          record.student?.rollNumber?.toLowerCase().includes(searchLower) ||
          record.subject?.name?.toLowerCase().includes(searchLower) ||
          record.subject?.code?.toLowerCase().includes(searchLower) ||
          (record.status || "present").toLowerCase().includes(searchLower) ||
          record.markedBy?.toLowerCase().includes(searchLower) ||
          new Date(record.date).toLocaleDateString("en-US").toLowerCase().includes(searchLower)
      );
      setFilteredAttendance(filtered);
    }
  }, [attendanceSearch, allAttendance]);

  const handleResetStudentPassword = async (studentId) => {
    if (!newPassword || newPassword.length < 6) {
      alert("Password must be at least 6 characters");
      return;
    }

    const token = localStorage.getItem("token");
    try {
      const res = await fetch(`http://localhost:5000/api/students/${studentId}/reset-password`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ newPassword }),
      });

      const data = await res.json();
      if (res.ok) {
        alert("Student password reset successfully!");
        setResettingPassword(null);
        setNewPassword("");
      } else {
        alert(data.error || "Failed to reset password");
      }
    } catch (err) {
      console.error("Reset password error:", err);
      alert("Failed to reset password");
    }
  };

  const handleResetTeacherPassword = async (teacherId) => {
    if (!newPassword || newPassword.length < 6) {
      alert("Password must be at least 6 characters");
      return;
    }

    const token = localStorage.getItem("token");
    try {
      const res = await fetch(`http://localhost:5000/api/teachers/${teacherId}/reset-password`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ newPassword }),
      });

      const data = await res.json();
      if (res.ok) {
        alert("Teacher password reset successfully!");
        setResettingPassword(null);
        setNewPassword("");
      } else {
        alert(data.error || "Failed to reset password");
      }
    } catch (err) {
      console.error("Reset password error:", err);
      alert("Failed to reset password");
    }
  };

  const handleDeleteStudent = async (studentId, studentName) => {
    if (!window.confirm(`Are you sure you want to delete student "${studentName}"?\n\nThis will permanently delete:\n- Student profile\n- Associated user account\n- All attendance records\n- All enrollment records\n- Video file and dataset frames\n\nThis action cannot be undone!`)) {
      return;
    }

    const token = localStorage.getItem("token");
    try {
      const res = await fetch(`http://localhost:5000/api/students/${studentId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();
      if (res.ok) {
        alert(data.message || "Student deleted successfully!");
        loadData(); // Refresh the student list
      } else {
        alert(data.error || "Failed to delete student");
      }
    } catch (err) {
      console.error("Delete student error:", err);
      alert("Failed to delete student. Please try again.");
    }
  };


  const handleRegisterTeacher = async (e) => {
    e.preventDefault();

    // Validate form
    if (!teacherForm.name || !teacherForm.email || !teacherForm.password) {
      alert("Please fill in all fields");
      return;
    }

    const token = localStorage.getItem("token");
    if (!token) {
      alert("Not authenticated. Please login again.");
      navigate("/login");
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/api/teachers/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(teacherForm),
      });

      // Read response as text first, then try to parse as JSON
      const responseText = await res.text();
      let data;

      try {
        data = JSON.parse(responseText);
      } catch (parseErr) {
        console.error("Failed to parse response as JSON:", responseText);
        alert(`Registration failed: ${res.status} ${res.statusText}\nResponse: ${responseText}`);
        return;
      }

      if (res.ok) {
        alert("Teacher registered successfully!");
        setShowRegisterTeacher(false);
        setTeacherForm({ name: "", email: "", password: "" });
        loadData();
      } else {
        console.error("Registration error:", data);
        alert(data.error || `Registration failed: ${res.status} ${res.statusText}`);
      }
    } catch (err) {
      console.error("Registration error:", err);
      alert(`Registration failed: ${err.message}. Please check console for details.`);
    }
  };

  const handleAddSubject = async (e) => {
    e.preventDefault();
    const token = localStorage.getItem("token");
    try {
      const res = await fetch("http://localhost:5000/api/subjects", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(subjectForm),
      });

      if (res.ok) {
        alert("Subject added successfully!");
        setShowAddSubject(false);
        setSubjectForm({ name: "", code: "" });
        loadData();
      } else {
        const data = await res.json();
        alert(data.error || "Failed to add subject");
      }
    } catch (err) {
      alert("Failed to add subject");
    }
  };

  const toggleManualAttendance = async () => {
    const token = localStorage.getItem("token");
    try {
      const res = await fetch("http://localhost:5000/api/settings", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          allowManualAttendance: !settings.allowManualAttendance,
        }),
      });

      if (res.ok) {
        const updated = await res.json();
        setSettings(updated);
      }
    } catch (err) {
      alert("Failed to update settings");
    }
  };

  const handleAssignSubjects = (teacher) => {
    setSelectedTeacher(teacher);
    setShowAssignSubjects(true);
  };

  const handleSaveSubjectAssignment = async () => {
    if (!selectedTeacher) return;

    const token = localStorage.getItem("token");
    const selectedSubjectIds = subjects
      .filter((subj) => {
        const checkbox = document.getElementById(`subject-${subj._id}`);
        return checkbox?.checked;
      })
      .map((subj) => subj._id);

    try {
      const res = await fetch(
        `http://localhost:5000/api/teachers/${selectedTeacher._id}/assign-subjects`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ subjectIds: selectedSubjectIds }),
        }
      );

      if (res.ok) {
        alert("Subjects assigned successfully!");
        setShowAssignSubjects(false);
        setSelectedTeacher(null);
        loadData();
      } else {
        const data = await res.json();
        alert(data.error || "Failed to assign subjects");
      }
    } catch (err) {
      alert("Failed to assign subjects");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
      <AdminSidebarSimple 
        activeSection={activeSection} 
        onSectionChange={setActiveSection}
      />
      
      <main className="flex-1 overflow-y-auto flex flex-col">
        {/* Topbar */}
        <nav className="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700 h-16 flex items-center justify-between px-6 sticky top-0 z-10">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Admin Dashboard</h2>
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
            <p className="text-gray-500 dark:text-gray-400 text-sm">Total Students</p>
            <h2 className="text-4xl font-bold text-gray-800 dark:text-white">{students.length}</h2>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
            <p className="text-gray-500 dark:text-gray-400 text-sm">Total Subjects</p>
            <h2 className="text-4xl font-bold text-gray-800 dark:text-white">{subjects.length}</h2>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
            <p className="text-gray-500 dark:text-gray-400 text-sm">Manual Attendance</p>
            <div className="flex items-center justify-between mt-2">
              <span className={`text-lg font-semibold ${settings?.allowManualAttendance ? "text-green-600 dark:text-green-400" : "text-gray-400 dark:text-gray-500"}`}>
                {settings?.allowManualAttendance ? "Enabled" : "Disabled"}
              </span>
              <button
                onClick={toggleManualAttendance}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${settings?.allowManualAttendance
                  ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50"
                  : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-900/50"
                  }`}
              >
                {settings?.allowManualAttendance ? "Disable" : "Enable"}
              </button>
            </div>
          </div>
        </div>

        <div className="flex gap-4 mb-6 flex-wrap">
          <button
            onClick={() => setShowRegisterTeacher(true)}
            className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium transition-colors"
          >
            Register Teacher
          </button>
          <button
            onClick={() => setShowAddSubject(true)}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium transition-colors"
          >
            Add Subject
          </button>
          <button
            onClick={() => {
              setAttendanceSearch(""); // Reset search when opening
              setShowAttendanceManagement(true);
              loadData(); // Refresh attendance when opening
            }}
            className="px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-medium transition-colors"
          >
            Manage Attendance
          </button>
        </div>

        {/* Register Teacher Modal */}
        {showRegisterTeacher && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-md w-full p-6">
              <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Register New Teacher</h2>
              <form onSubmit={handleRegisterTeacher} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Name</label>
                  <input
                    type="text"
                    required
                    value={teacherForm.name}
                    onChange={(e) =>
                      setTeacherForm({ ...teacherForm, name: e.target.value })
                    }
                    className="w-full px-4 py-2 border rounded-lg"
                    placeholder="Enter teacher name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Email</label>
                  <input
                    type="email"
                    required
                    value={teacherForm.email}
                    onChange={(e) =>
                      setTeacherForm({ ...teacherForm, email: e.target.value })
                    }
                    className="w-full px-4 py-2 border rounded-lg"
                    placeholder="Enter teacher email"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Password</label>
                  <input
                    type="password"
                    required
                    value={teacherForm.password}
                    onChange={(e) =>
                      setTeacherForm({ ...teacherForm, password: e.target.value })
                    }
                    className="w-full px-4 py-2 border rounded-lg"
                    placeholder="Enter password"
                  />
                </div>
                <div className="flex gap-4 pt-4">
                  <button
                    type="submit"
                    className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                  >
                    Register Teacher
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowRegisterTeacher(false);
                      setTeacherForm({ name: "", email: "", password: "" });
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

        {/* Add Subject Modal */}
        {showAddSubject && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-md w-full p-6">
              <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Add Subject</h2>
              <form onSubmit={handleAddSubject} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Subject Name</label>
                  <input
                    type="text"
                    required
                    value={subjectForm.name}
                    onChange={(e) =>
                      setSubjectForm({ ...subjectForm, name: e.target.value })
                    }
                    className="w-full px-4 py-2 border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Subject Code</label>
                  <input
                    type="text"
                    required
                    value={subjectForm.code}
                    onChange={(e) =>
                      setSubjectForm({ ...subjectForm, code: e.target.value })
                    }
                    className="w-full px-4 py-2 border rounded-lg"
                  />
                </div>
                <div className="flex gap-4 pt-4">
                  <button
                    type="submit"
                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                  >
                    Add Subject
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowAddSubject(false)}
                    className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Teachers List */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Teachers</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Name</th>
                  <th className="text-left p-2">Email</th>
                  <th className="text-left p-2">Assigned Subjects</th>
                  <th className="text-left p-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {teachers.map((teacher) => (
                  <tr key={teacher._id} className="border-b">
                    <td className="p-2">{teacher.name}</td>
                    <td className="p-2">{teacher.email}</td>
                    <td className="p-2">
                      {teacher.assignedSubjects && teacher.assignedSubjects.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {teacher.assignedSubjects.map((subj) => (
                            <span
                              key={subj._id}
                              className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs"
                            >
                              {subj.name}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-gray-400 text-sm">No subjects assigned</span>
                      )}
                    </td>
                    <td className="p-2">
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleAssignSubjects(teacher)}
                          className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm"
                        >
                          Assign Subjects
                        </button>
                        <button
                          onClick={() => setResettingPassword({ type: "teacher", id: teacher._id })}
                          className="px-3 py-1 bg-orange-600 text-white rounded hover:bg-orange-700 text-sm"
                        >
                          Reset Password
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Students List */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border dark:border-gray-700 p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Students ({filteredStudents.length})</h2>
            <div className="flex items-center gap-4">
              <input
                type="text"
                placeholder="Search by name, roll number, or email..."
                value={studentSearch}
                onChange={(e) => setStudentSearch(e.target.value)}
                className="px-4 py-2 border dark:border-gray-600 rounded-lg w-64 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Name</th>
                  <th className="text-left p-2">Roll Number</th>
                  <th className="text-left p-2">Email</th>
                  <th className="text-left p-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredStudents.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="p-8 text-center text-gray-500">
                      {studentSearch ? "No students found matching your search." : "No students registered yet."}
                    </td>
                  </tr>
                ) : (
                  filteredStudents.map((student) => (
                    <tr key={student._id} className="border-b hover:bg-gray-50">
                      <td className="p-2">{student.name}</td>
                      <td className="p-2">{student.rollNumber}</td>
                      <td className="p-2">{student.email}</td>
                      <td className="p-2">
                        <div className="flex gap-2">
                          <button
                            onClick={() => setResettingPassword({ type: "student", id: student._id })}
                            className="px-3 py-1 bg-orange-600 text-white rounded hover:bg-orange-700 text-sm"
                          >
                            Reset Password
                          </button>
                          <button
                            onClick={() => handleDeleteStudent(student._id, student.name)}
                            className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Password Reset Modal */}
        {resettingPassword && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-md w-full p-6">
              <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
                Reset {resettingPassword.type === "student" ? "Student" : "Teacher"} Password
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">New Password</label>
                  <input
                    type="password"
                    required
                    minLength={6}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Enter new password (min 6 characters)"
                    className="w-full px-4 py-2 border rounded-lg"
                  />
                </div>
                <div className="flex gap-4 pt-4">
                  <button
                    onClick={() => {
                      if (resettingPassword.type === "student") {
                        handleResetStudentPassword(resettingPassword.id);
                      } else {
                        handleResetTeacherPassword(resettingPassword.id);
                      }
                    }}
                    disabled={!newPassword || newPassword.length < 6}
                    className="flex-1 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    Reset Password
                  </button>
                  <button
                    onClick={() => {
                      setResettingPassword(null);
                      setNewPassword("");
                    }}
                    className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Assign Subjects Modal */}
        {showAssignSubjects && selectedTeacher && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6">
              <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
                Assign Subjects to {selectedTeacher.name}
              </h2>
              <div className="space-y-3 mb-6">
                {subjects.length === 0 ? (
                  <p className="text-gray-500">No subjects available. Please add subjects first.</p>
                ) : (
                  subjects.map((subject) => {
                    const isAssigned =
                      selectedTeacher.assignedSubjects?.some(
                        (subj) => subj._id === subject._id
                      ) || false;
                    return (
                      <label
                        key={subject._id}
                        className="flex items-center p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          id={`subject-${subject._id}`}
                          defaultChecked={isAssigned}
                          className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500"
                        />
                        <div className="ml-3">
                          <p className="font-medium text-gray-800">{subject.name}</p>
                          <p className="text-sm text-gray-500">{subject.code}</p>
                        </div>
                      </label>
                    );
                  })
                )}
              </div>
              <div className="flex gap-4 pt-4">
                <button
                  onClick={handleSaveSubjectAssignment}
                  className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                >
                  Save Assignment
                </button>
                <button
                  onClick={() => {
                    setShowAssignSubjects(false);
                    setSelectedTeacher(null);
                  }}
                  className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Attendance Management Modal */}
        {showAttendanceManagement && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-6xl w-full max-h-[90vh] overflow-y-auto p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Attendance Management</h2>
                <button
                  onClick={() => {
                    setShowAttendanceManagement(false);
                    setEditingAttendance(null);
                    setAttendanceSearch("");
                  }}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                >
                  Close
                </button>
              </div>

              {/* Search Bar */}
              <div className="mb-4">
                <input
                  type="text"
                  placeholder="Search by student name, roll number, subject, status, date..."
                  value={attendanceSearch}
                  onChange={(e) => setAttendanceSearch(e.target.value)}
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
                {attendanceSearch && (
                  <p className="text-sm text-gray-500 mt-1">
                    Showing {filteredAttendance.length} of {allAttendance.length} records
                  </p>
                )}
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Student</th>
                      <th className="text-left p-2">Roll Number</th>
                      <th className="text-left p-2">Subject</th>
                      <th className="text-left p-2">Date</th>
                      <th className="text-left p-2">Time</th>
                      <th className="text-left p-2">Status</th>
                      <th className="text-left p-2">Marked By</th>
                      <th className="text-left p-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredAttendance.length === 0 ? (
                      <tr>
                        <td colSpan="8" className="p-4 text-center text-gray-500">
                          {attendanceSearch ? "No attendance records found matching your search." : "No attendance records found"}
                        </td>
                      </tr>
                    ) : (
                      filteredAttendance.map((record) => (
                        <tr key={record._id} className="border-b hover:bg-gray-50">
                          <td className="p-2">{record.student?.name || "N/A"}</td>
                          <td className="p-2">{record.student?.rollNumber || "N/A"}</td>
                          <td className="p-2">
                            {record.subject?.name || "N/A"} ({record.subject?.code || "N/A"})
                          </td>
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
                          <td className="p-2">
                            <div className="flex gap-2">
                              <button
                                onClick={() => {
                                  setEditingAttendance(record);
                                  const date = new Date(record.date);
                                  const dateStr = date.toISOString().split('T')[0];
                                  setEditForm({
                                    studentId: record.student?._id || "",
                                    subjectId: record.subject?._id || "",
                                    date: dateStr,
                                    markedBy: record.markedBy || "auto",
                                    status: record.status || "present",
                                  });
                                }}
                                className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                              >
                                Edit
                              </button>
                              <button
                                onClick={async () => {
                                  if (!window.confirm("Are you sure you want to delete this attendance record?")) {
                                    return;
                                  }
                                  const token = localStorage.getItem("token");
                                  try {
                                    const res = await fetch(
                                      `http://localhost:5000/api/attendance/${record._id}`,
                                      {
                                        method: "DELETE",
                                        headers: {
                                          Authorization: `Bearer ${token}`,
                                        },
                                      }
                                    );
                                    if (res.ok) {
                                      alert("Attendance record deleted successfully");
                                      loadData(); // This will refresh allAttendance and filteredAttendance
                                      // Search will automatically update via useEffect
                                    } else {
                                      const data = await res.json();
                                      alert(data.error || "Failed to delete attendance");
                                    }
                                  } catch (err) {
                                    alert("Failed to delete attendance");
                                  }
                                }}
                                className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Edit Attendance Modal */}
              {editingAttendance && (
                <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4">
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 max-w-md w-full p-6">
                    <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Edit Attendance Record</h3>
                    <form
                      onSubmit={async (e) => {
                        e.preventDefault();
                        const token = localStorage.getItem("token");
                        try {
                          const res = await fetch(
                            `http://localhost:5000/api/attendance/${editingAttendance._id}`,
                            {
                              method: "PUT",
                              headers: {
                                "Content-Type": "application/json",
                                Authorization: `Bearer ${token}`,
                              },
                              body: JSON.stringify(editForm),
                            }
                          );
                          const data = await res.json();
                          if (res.ok) {
                            alert("Attendance updated successfully!");
                            setEditingAttendance(null);
                            loadData(); // This will refresh allAttendance and filteredAttendance
                            // Search will automatically update via useEffect
                          } else {
                            alert(data.error || "Failed to update attendance");
                          }
                        } catch (err) {
                          alert("Failed to update attendance");
                        }
                      }}
                      className="space-y-4"
                    >
                      <div>
                        <label className="block text-sm font-medium mb-2">Student</label>
                        <select
                          required
                          value={editForm.studentId}
                          onChange={(e) =>
                            setEditForm({ ...editForm, studentId: e.target.value })
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
                          value={editForm.subjectId}
                          onChange={(e) =>
                            setEditForm({ ...editForm, subjectId: e.target.value })
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
                          value={editForm.date}
                          onChange={(e) =>
                            setEditForm({ ...editForm, date: e.target.value })
                          }
                          className="w-full px-4 py-2 border rounded-lg"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Status</label>
                        <select
                          required
                          value={editForm.status || "present"}
                          onChange={(e) =>
                            setEditForm({ ...editForm, status: e.target.value })
                          }
                          className="w-full px-4 py-2 border rounded-lg"
                        >
                          <option value="present">Present</option>
                          <option value="absent">Absent</option>
                          <option value="leave">Leave</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Marked By</label>
                        <select
                          required
                          value={editForm.markedBy}
                          onChange={(e) =>
                            setEditForm({ ...editForm, markedBy: e.target.value })
                          }
                          className="w-full px-4 py-2 border rounded-lg"
                        >
                          <option value="auto">Auto</option>
                          <option value="manual">Manual</option>
                        </select>
                      </div>
                      <div className="flex gap-4 pt-4">
                        <button
                          type="submit"
                          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                        >
                          Update
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingAttendance(null)}
                          className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                        >
                          Cancel
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        </div>
      </main>
    </div>
  );
}

