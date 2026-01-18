// server.js
import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import multer from "multer";
import dotenv from "dotenv";
import axios from "axios";
import fs from "fs";
import path from "path";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import FormData from "form-data";

import Student from "./models/Student.js";
import Attendance from "./models/Attendance.js";
import User from "./models/User.js";
import Subject from "./models/Subject.js";
import Settings from "./models/Settings.js";
import Enrollment from "./models/Enrollment.js";
import bcrypt from "bcryptjs";

dotenv.config();

const app = express();
app.use(cors());
// Increase JSON body parser limit to handle large image data (base64 encoded images can be large)
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ limit: "10mb", extended: true }));

/* -------------------- FFmpeg Setup -------------------- */
ffmpeg.setFfmpegPath(ffmpegPath);

/* -------------------- Ensure folders exist -------------------- */
["uploads", "frames"].forEach((dir) => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir);
});

/* -------------------- MongoDB Connection -------------------- */
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("Mongo error:", err));

/* -------------------- Multer Setup -------------------- */
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + ".webm");
  },
});

const upload = multer({ storage });

/* -------------------- Convert video to mp4 -------------------- */
function convertToMp4(inputPath) {
  return new Promise((resolve, reject) => {
    // Generate output path by replacing extension with .mp4
    const ext = path.extname(inputPath);
    const outputPath = inputPath.replace(ext, ".mp4");

    console.log(`Converting video: ${inputPath} → ${outputPath}`);

    ffmpeg(inputPath)
      .videoCodec("libx264")
      .audioCodec("aac")
      .output(outputPath)
      .on("start", (commandLine) => {
        console.log(`FFmpeg command: ${commandLine}`);
      })
      .on("progress", (progress) => {
        if (progress.percent) {
          console.log(`Conversion progress: ${Math.round(progress.percent)}%`);
        }
      })
      .on("end", () => {
        console.log(`✅ Video converted successfully: ${outputPath}`);
        resolve(outputPath);
      })
      .on("error", (err) => {
        console.error(`❌ Video conversion error:`, err.message);
        reject(err);
      })
      .run();
  });
}

/* -------------------- Extract ~65 Frames -------------------- */
function extractFrames(videoPath, studentId, targetFrames = 65) {
  return new Promise((resolve, reject) => {
    const framesDir = path.resolve("frames", studentId);
    fs.mkdirSync(framesDir, { recursive: true });

    // ~10s video → 6–7 fps = ~65 frames
    const fps = targetFrames / 10;

    ffmpeg(videoPath)
      .outputOptions(["-vf", `fps=${fps}`])
      .output(path.join(framesDir, "%03d.jpg"))
      .on("end", () => {
        console.log(`✅ Extracted frames for ${studentId}`);
        resolve(framesDir);
      })
      .on("error", reject)
      .run();
  });
}

/* -------------------- Authentication Middleware -------------------- */
const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      return res.status(401).json({ error: "No authorization header" });
    }

    const token = authHeader.split(" ")[1];
    if (!token) {
      return res.status(401).json({ error: "No token provided" });
    }

    // Simple token validation - in production use JWT
    const user = await User.findById(token)
      .populate("studentId")
      .populate("assignedSubjects");
    if (!user) {
      return res.status(401).json({ error: "Invalid token - user not found" });
    }

    req.user = user;
    next();
  } catch (err) {
    console.error("Authentication error:", err);
    // Always return JSON, never HTML
    return res
      .status(401)
      .json({ error: "Authentication failed", details: err.message });
  }
};

/* -------------------- Login -------------------- */
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email })
      .populate("studentId")
      .populate("assignedSubjects");
    if (!user || !(await bcrypt.compare(password, user.password))) {
      return res.status(401).json({ error: "Invalid credentials" });
    }
    const userObj = user.toObject();
    delete userObj.password;
    res.json({ user: userObj, token: user._id.toString() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Register Student (Self-registration only, no auth required) -------------------- */
app.post("/api/students/register", upload.single("video"), async (req, res) => {
  try {
    const { name, rollNumber, email, password, confirmPassword } = req.body;

    if (!name || !rollNumber || !email || !password || !confirmPassword) {
      return res.status(400).json({ error: "All fields required" });
    }

    if (password !== confirmPassword) {
      return res.status(400).json({ error: "Passwords do not match" });
    }

    if (password.length < 6) {
      return res
        .status(400)
        .json({ error: "Password must be at least 6 characters" });
    }

    if (!req.file) {
      return res.status(400).json({ error: "Video is required" });
    }

    // Check if student already exists
    const existingStudent = await Student.findOne({ email });
    if (existingStudent) {
      // Clean up uploaded file
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      return res
        .status(400)
        .json({ error: "Student with this email already registered" });
    }

    // Check if roll number already exists
    const existingRollNumber = await Student.findOne({ rollNumber });
    if (existingRollNumber) {
      // Clean up uploaded file
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      return res
        .status(400)
        .json({ error: "Student with this roll number already exists" });
    }

    // Check if user account already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      // Clean up uploaded file
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      return res.status(400).json({ error: "User account already exists" });
    }

    let mp4Path;
    const originalExt = path.extname(req.file.originalname).toLowerCase();
    const uploadedPath = req.file.path;
    const uploadedExt = path.extname(uploadedPath).toLowerCase();

    // 1️⃣ Convert video to mp4 if needed (handles .webm, .mp4, .mov, .avi, etc.)
    if (uploadedExt === ".mp4" && originalExt === ".mp4") {
      // Already mp4, use as-is
      mp4Path = uploadedPath;
      console.log(`✅ Video is already MP4 format: ${mp4Path}`);
    } else {
      // Convert to mp4 using ffmpeg (handles any video format)
      console.log(`🔄 Converting video from ${uploadedExt} to .mp4...`);
      mp4Path = await convertToMp4(uploadedPath);
      // Clean up original file after conversion
      if (uploadedPath !== mp4Path && fs.existsSync(uploadedPath)) {
        try {
          fs.unlinkSync(uploadedPath);
          console.log(`🗑️ Removed original file: ${uploadedPath}`);
        } catch (cleanupErr) {
          console.warn(
            `⚠️ Could not remove original file: ${cleanupErr.message}`,
          );
        }
      }
    }

    // 2️⃣ Save student
    const student = await Student.create({
      name,
      rollNumber,
      email,
      videoPath: mp4Path,
    });

    // 3️⃣ Create user account for student
    const hashedPassword = await bcrypt.hash(password, 10);
    await User.create({
      email,
      password: hashedPassword,
      role: "student",
      name,
      studentId: student._id,
    });

    // 4️⃣ Extract frames (approximately 65 frames from 10-second video)
    const framesDir = await extractFrames(mp4Path, student._id.toString());

    // 5️⃣ Send ABSOLUTE path to AI server for training
    // Training can take a long time (processing 65 frames with YOLOv8-face + ArcFace)
    // Set timeout to 5 minutes (300000ms) - training is async, so this won't block the response
    axios
      .post(
        "http://127.0.0.1:8000/train",
        {
          studentId: student._id.toString(),
          framesDir: framesDir,
        },
        {
          timeout: 300000, // 5 minutes timeout for training (YOLOv8-face + ArcFace can be slow)
        },
      )
      .then(() => {
        console.log(
          `✅ AI training completed for student: ${student.name} (${student._id})`,
        );
      })
      .catch((err) => {
        if (err.code === "ECONNABORTED" || err.message.includes("timeout")) {
          console.error(
            `⚠️ AI training timed out for student ${student._id} - training may still be in progress`,
          );
        } else {
          console.error(
            `⚠️ AI training failed for student ${student._id}:`,
            err.message,
          );
        }
      });

    res.status(201).json({
      message: "Student registered successfully. AI training started.",
      student: {
        _id: student._id,
        name: student.name,
        rollNumber: student.rollNumber,
        email: student.email,
      },
    });
  } catch (err) {
    console.error("Student registration error:", err);

    // Clean up uploaded file on error
    if (req.file && fs.existsSync(req.file.path)) {
      try {
        fs.unlinkSync(req.file.path);
      } catch (cleanupErr) {
        console.error("Error cleaning up file:", cleanupErr);
      }
    }

    res
      .status(500)
      .json({ error: err.message || "Failed to register student" });
  }
});

/* -------------------- Register Teacher (Admin only) -------------------- */
app.post("/api/teachers/register", authenticate, async (req, res) => {
  try {
    console.log("Teacher registration request:", {
      userRole: req.user?.role,
      body: { ...req.body, password: "[HIDDEN]" },
    });

    if (!req.user || req.user.role !== "admin") {
      return res.status(403).json({ error: "Admin access required" });
    }

    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res
        .status(400)
        .json({ error: "All fields (name, email, password) are required" });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res
        .status(400)
        .json({ error: "User with this email already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const teacher = await User.create({
      email,
      password: hashedPassword,
      role: "teacher",
      name,
      assignedSubjects: [], // Initialize empty array
    });

    console.log("Teacher created successfully:", teacher.email);

    res.status(201).json({
      message: "Teacher registered successfully",
      teacher: { ...teacher.toObject(), password: undefined },
    });
  } catch (err) {
    console.error("Teacher registration error:", err);
    res.status(500).json({
      error: err.message || "Failed to register teacher",
      details: process.env.NODE_ENV === "development" ? err.stack : undefined,
    });
  }
});

/* -------------------- Register Admin (Admin only) -------------------- */
app.post("/api/admins/register", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "admin") {
      return res.status(403).json({ error: "Admin only" });
    }

    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ error: "All fields required" });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res
        .status(400)
        .json({ error: "User with this email already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const admin = await User.create({
      email,
      password: hashedPassword,
      role: "admin",
      name,
    });

    res.status(201).json({
      message: "Admin registered successfully",
      admin: { ...admin.toObject(), password: undefined },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Get All Teachers -------------------- */
app.get("/api/teachers", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "admin") {
      return res.status(403).json({ error: "Admin only" });
    }
    const teachers = await User.find({ role: "teacher" })
      .select("-password")
      .populate("assignedSubjects");
    res.json(teachers);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Assign Subjects to Teacher (Admin only) -------------------- */
app.put(
  "/api/teachers/:teacherId/assign-subjects",
  authenticate,
  async (req, res) => {
    try {
      if (req.user.role !== "admin") {
        return res.status(403).json({ error: "Admin only" });
      }

      const { teacherId } = req.params;
      const { subjectIds } = req.body;

      if (!Array.isArray(subjectIds)) {
        return res.status(400).json({ error: "subjectIds must be an array" });
      }

      const teacher = await User.findById(teacherId);
      if (!teacher || teacher.role !== "teacher") {
        return res.status(404).json({ error: "Teacher not found" });
      }

      teacher.assignedSubjects = subjectIds;
      await teacher.save();

      const updatedTeacher = await User.findById(teacherId)
        .select("-password")
        .populate("assignedSubjects");

      res.json({
        message: "Subjects assigned successfully",
        teacher: updatedTeacher,
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message });
    }
  },
);

/* -------------------- Get Teacher's Assigned Subjects -------------------- */
app.get("/api/teachers/my-subjects", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "teacher") {
      return res.status(403).json({ error: "Teacher only" });
    }

    const teacher = await User.findById(req.user._id)
      .populate("assignedSubjects")
      .select("-password");

    res.json(teacher.assignedSubjects || []);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Get Attendance for Teacher's Subjects -------------------- */
app.get("/api/teachers/attendance", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "teacher") {
      return res.status(403).json({ error: "Teacher only" });
    }

    const teacher = await User.findById(req.user._id).populate(
      "assignedSubjects",
    );
    const subjectIds = teacher.assignedSubjects.map((subj) => subj._id);

    if (subjectIds.length === 0) {
      return res.json({});
    }

    const attendances = await Attendance.find({
      subject: { $in: subjectIds },
    })
      .populate("student")
      .populate("subject")
      .sort({ date: -1 });

    // Group by subject
    const bySubject = {};
    attendances.forEach((att) => {
      const subjName = att.subject?.name || "Unknown";
      if (!bySubject[subjName]) {
        bySubject[subjName] = [];
      }
      bySubject[subjName].push({
        student: {
          _id: att.student._id,
          name: att.student.name,
          rollNumber: att.student.rollNumber,
          email: att.student.email,
        },
        date: att.date,
        markedBy: att.markedBy,
      });
    });

    res.json(bySubject);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Get All Students (with search) -------------------- */
app.get("/api/students", authenticate, async (req, res) => {
  try {
    // Only admin and teachers can view all students
    if (req.user.role !== "admin" && req.user.role !== "teacher") {
      return res.status(403).json({ error: "Access denied" });
    }

    const { search } = req.query;
    let query = {};

    // Search by name or roll number
    if (search && search.trim()) {
      const searchRegex = { $regex: search.trim(), $options: "i" };
      query.$or = [
        { name: searchRegex },
        { rollNumber: searchRegex },
        { email: searchRegex },
      ];
    }

    const students = await Student.find(query)
      .select("-videoPath")
      .sort({ name: 1 });
    res.json(students);
  } catch (err) {
    console.error("Get students error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Reset Student Password (Admin only) -------------------- */
app.put(
  "/api/students/:studentId/reset-password",
  authenticate,
  async (req, res) => {
    try {
      if (req.user.role !== "admin") {
        return res.status(403).json({ error: "Admin only" });
      }

      const { studentId } = req.params;
      const { newPassword } = req.body;

      if (!newPassword || newPassword.length < 6) {
        return res
          .status(400)
          .json({ error: "Password must be at least 6 characters" });
      }

      // Find student
      const student = await Student.findById(studentId);
      if (!student) {
        return res.status(404).json({ error: "Student not found" });
      }

      // Find user account associated with student
      const user = await User.findOne({
        studentId: studentId,
        role: "student",
      });
      if (!user) {
        return res
          .status(404)
          .json({ error: "Student user account not found" });
      }

      // Hash and update password
      const hashedPassword = await bcrypt.hash(newPassword, 10);
      user.password = hashedPassword;
      await user.save();

      res.json({ message: "Student password reset successfully" });
    } catch (err) {
      console.error("Reset student password error:", err);
      res.status(500).json({ error: err.message });
    }
  },
);

/* -------------------- Delete Student (Admin only) -------------------- */
app.delete("/api/students/:studentId", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "admin") {
      return res.status(403).json({ error: "Admin only" });
    }

    const { studentId } = req.params;

    // Find student
    const student = await Student.findById(studentId);
    if (!student) {
      return res.status(404).json({ error: "Student not found" });
    }

    // Store paths before deletion for cleanup
    const videoPath = student.videoPath;
    const framesDir = path.resolve("frames", student._id.toString());

    // Delete associated User account
    const user = await User.findOne({ studentId: studentId, role: "student" });
    if (user) {
      await User.findByIdAndDelete(user._id);
      console.log(`✓ Deleted user account for student ${studentId}`);
    }

    // Delete all Attendance records for this student
    const attendanceCount = await Attendance.countDocuments({
      student: studentId,
    });
    await Attendance.deleteMany({ student: studentId });
    console.log(
      `✓ Deleted ${attendanceCount} attendance records for student ${studentId}`,
    );

    // Delete all Enrollment records for this student
    const enrollmentCount = await Enrollment.countDocuments({
      student: studentId,
    });
    await Enrollment.deleteMany({ student: studentId });
    console.log(
      `✓ Deleted ${enrollmentCount} enrollment records for student ${studentId}`,
    );

    // Delete video file if it exists
    if (videoPath && fs.existsSync(videoPath)) {
      try {
        fs.unlinkSync(videoPath);
        console.log(`✓ Deleted video file: ${videoPath}`);
      } catch (fileErr) {
        console.error(`⚠ Failed to delete video file: ${fileErr.message}`);
      }
    }

    // Delete frames directory if it exists
    if (fs.existsSync(framesDir)) {
      try {
        fs.rmSync(framesDir, { recursive: true, force: true });
        console.log(`✓ Deleted frames directory: ${framesDir}`);
      } catch (dirErr) {
        console.error(`⚠ Failed to delete frames directory: ${dirErr.message}`);
      }
    }

    // Delete student record
    await Student.findByIdAndDelete(studentId);

    console.log(
      `✓ Student ${studentId} (${student.name}) deleted successfully`,
    );

    res.json({
      message: `Student deleted successfully. Removed ${attendanceCount} attendance records and ${enrollmentCount} enrollment records.`,
    });
  } catch (err) {
    console.error("Delete student error:", err);
    res.status(500).json({ error: err.message || "Failed to delete student" });
  }
});

/* -------------------- Reset Teacher Password (Admin only) -------------------- */
app.put(
  "/api/teachers/:teacherId/reset-password",
  authenticate,
  async (req, res) => {
    try {
      if (req.user.role !== "admin") {
        return res.status(403).json({ error: "Admin only" });
      }

      const { teacherId } = req.params;
      const { newPassword } = req.body;

      if (!newPassword || newPassword.length < 6) {
        return res
          .status(400)
          .json({ error: "Password must be at least 6 characters" });
      }

      // Find teacher user
      const teacher = await User.findById(teacherId);
      if (!teacher || teacher.role !== "teacher") {
        return res.status(404).json({ error: "Teacher not found" });
      }

      // Hash and update password
      const hashedPassword = await bcrypt.hash(newPassword, 10);
      teacher.password = hashedPassword;
      await teacher.save();

      res.json({ message: "Teacher password reset successfully" });
    } catch (err) {
      console.error("Reset teacher password error:", err);
      res.status(500).json({ error: err.message });
    }
  },
);

/* -------------------- Get Student Attendance by Subject -------------------- */
app.get(
  "/api/students/:studentId/attendance",
  authenticate,
  async (req, res) => {
    try {
      const { studentId } = req.params;
      const attendances = await Attendance.find({ student: studentId })
        .populate("subject")
        .sort({ date: -1 });

      // Group by subject
      const bySubject = {};
      attendances.forEach((att) => {
        const subjName = att.subject?.name || "Unknown";
        if (!bySubject[subjName]) {
          bySubject[subjName] = [];
        }
        bySubject[subjName].push({
          date: att.date,
          markedBy: att.markedBy,
        });
      });

      res.json(bySubject);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  },
);

/* -------------------- Mark Attendance -------------------- */
app.post("/api/attendance/mark", authenticate, async (req, res) => {
  try {
    const {
      studentId,
      subjectId,
      markedBy = "auto",
      date,
      status = "present",
    } = req.body;

    if (!studentId || !subjectId) {
      return res
        .status(400)
        .json({ error: "Student ID and Subject ID required" });
    }

    // Validate status
    if (!["present", "absent", "leave"].includes(status)) {
      return res
        .status(400)
        .json({ error: "Invalid status. Must be present, absent, or leave" });
    }

    // CRITICAL: Check if student is enrolled and approved in this course
    // Note: For attendance marking, we check if student is enrolled with ANY teacher for this subject
    // This allows any assigned teacher to mark attendance for any enrolled student
    const enrollment = await Enrollment.findOne({
      student: studentId,
      subject: subjectId,
      status: "approved",
    });

    if (!enrollment) {
      return res.status(403).json({
        error: "Student is not enrolled or not approved in this course",
      });
    }

    console.log(
      `✓ Student ${studentId} is enrolled in subject ${subjectId} (with teacher: ${enrollment.teacher})`,
    );

    // Check if manual attendance is allowed
    if (markedBy === "manual") {
      const settings = await Settings.findOne();
      if (!settings || !settings.allowManualAttendance) {
        return res.status(403).json({ error: "Manual attendance not allowed" });
      }
      if (req.user.role !== "teacher" && req.user.role !== "admin") {
        return res
          .status(403)
          .json({ error: "Only teachers can mark manual attendance" });
      }

      // If teacher, verify they are assigned to this subject
      if (req.user.role === "teacher") {
        const subjectIdStr = subjectId.toString();
        const teacher = await User.findById(req.user._id).populate(
          "assignedSubjects",
        );
        const assignedSubjectIds = (teacher.assignedSubjects || []).map(
          (subj) => subj._id?.toString() || subj.toString(),
        );

        const isAssigned =
          assignedSubjectIds.includes(subjectIdStr) ||
          (req.user.assignedSubjects || []).some((assignedId) => {
            const assignedIdStr =
              assignedId._id?.toString() || assignedId.toString();
            return assignedIdStr === subjectIdStr;
          });

        if (!isAssigned) {
          return res
            .status(403)
            .json({ error: "Not assigned to this subject" });
        }
      }
    }

    // Auto-marked attendance is always "present"
    const finalStatus = markedBy === "auto" ? "present" : status;

    // Use provided date or current date
    const attendanceDate = date ? new Date(date) : new Date();

    // Set time to current time if not provided, or keep the provided date/time
    if (!date) {
      attendanceDate.setHours(
        new Date().getHours(),
        new Date().getMinutes(),
        0,
        0,
      );
    }

    // Check for duplicate: same student, subject, and date (ignoring time)
    // This ensures only ONE attendance per day per student per subject
    const checkDate = new Date(attendanceDate);
    checkDate.setHours(0, 0, 0, 0);
    const checkTomorrow = new Date(checkDate);
    checkTomorrow.setDate(checkTomorrow.getDate() + 1);

    const alreadyMarked = await Attendance.findOne({
      student: studentId,
      subject: subjectId,
      date: { $gte: checkDate, $lt: checkTomorrow },
    });

    // For ANY attendance: prevent duplicate marking on the same day
    // ONLY ONE attendance record per student per subject per day
    if (alreadyMarked) {
      if (markedBy === "auto") {
        // Auto-attendance: prevent duplicate - already marked today, don't mark again
        console.log(
          `⚠️ Attendance already marked for student ${studentId} in subject ${subjectId} on ${checkDate.toISOString().split("T")[0]}`,
        );
        return res.json({
          message: "Attendance already marked for this student on this date",
          alreadyMarked: true,
          existingAttendance: {
            _id: alreadyMarked._id,
            date: alreadyMarked.date,
            status: alreadyMarked.status,
            markedBy: alreadyMarked.markedBy,
          },
        });
      } else if (markedBy === "manual") {
        // Manual attendance: update the existing record (allows editing previous attendance)
        alreadyMarked.status = finalStatus;
        alreadyMarked.markedBy = markedBy; // Update markedBy to manual
        alreadyMarked.date = attendanceDate; // Update date if changed
        await alreadyMarked.save();
        console.log(
          `✓ Updated existing attendance for student ${studentId} in subject ${subjectId}`,
        );
        return res.json({
          message: "Attendance updated successfully",
          alreadyMarked: true,
          attendanceId: alreadyMarked._id,
        });
      }
    }

    // Create new attendance record (no duplicate found)
    const newAttendance = await Attendance.create({
      student: studentId,
      subject: subjectId,
      date: attendanceDate,
      markedBy,
      status: finalStatus,
    });

    res.json({
      message: "Attendance marked successfully",
      attendanceId: newAttendance._id,
      alreadyMarked: false,
    });
  } catch (err) {
    console.error("Mark attendance error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Update Attendance (Admin/Teacher) -------------------- */
app.put("/api/attendance/:attendanceId", authenticate, async (req, res) => {
  try {
    const { attendanceId } = req.params;
    const { studentId, subjectId, date, markedBy, status } = req.body;

    // Only admin and teachers can update attendance
    if (req.user.role !== "admin" && req.user.role !== "teacher") {
      return res.status(403).json({ error: "Access denied" });
    }

    // Check if manual attendance is allowed (for teachers)
    if (req.user.role === "teacher" && markedBy === "manual") {
      const settings = await Settings.findOne();
      if (!settings || !settings.allowManualAttendance) {
        return res.status(403).json({ error: "Manual attendance not allowed" });
      }
    }

    const attendance = await Attendance.findById(attendanceId);
    if (!attendance) {
      return res.status(404).json({ error: "Attendance record not found" });
    }

    // Teachers can only edit attendance for their assigned subjects
    if (req.user.role === "teacher") {
      const teacher = await User.findById(req.user._id).populate(
        "assignedSubjects",
      );
      const canEdit = teacher.assignedSubjects.some(
        (subj) => subj._id.toString() === attendance.subject.toString(),
      );
      if (!canEdit && subjectId) {
        // Check if new subject is assigned
        const canEditNew = teacher.assignedSubjects.some(
          (subj) => subj._id.toString() === subjectId,
        );
        if (!canEditNew) {
          return res
            .status(403)
            .json({ error: "Not authorized to edit this attendance" });
        }
      }
    }

    // Validate status if provided
    if (status && !["present", "absent", "leave"].includes(status)) {
      return res
        .status(400)
        .json({ error: "Invalid status. Must be present, absent, or leave" });
    }

    // Update attendance
    if (studentId) attendance.student = studentId;
    if (subjectId) attendance.subject = subjectId;
    if (date) attendance.date = new Date(date);
    if (markedBy) attendance.markedBy = markedBy;
    if (status) attendance.status = status;

    await attendance.save();

    res.json({ message: "Attendance updated successfully", attendance });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Delete Attendance (Admin/Teacher) -------------------- */
app.delete("/api/attendance/:attendanceId", authenticate, async (req, res) => {
  try {
    const { attendanceId } = req.params;

    // Only admin and teachers can delete attendance
    if (req.user.role !== "admin" && req.user.role !== "teacher") {
      return res.status(403).json({ error: "Access denied" });
    }

    const attendance = await Attendance.findById(attendanceId);
    if (!attendance) {
      return res.status(404).json({ error: "Attendance record not found" });
    }

    // Teachers can only delete attendance for their assigned subjects
    if (req.user.role === "teacher") {
      const teacher = await User.findById(req.user._id).populate(
        "assignedSubjects",
      );
      const canDelete = teacher.assignedSubjects.some(
        (subj) => subj._id.toString() === attendance.subject.toString(),
      );
      if (!canDelete) {
        return res
          .status(403)
          .json({ error: "Not authorized to delete this attendance" });
      }
    }

    await Attendance.findByIdAndDelete(attendanceId);

    res.json({ message: "Attendance deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Get All Attendance Records (Admin/Teacher) -------------------- */
app.get("/api/attendance/all", authenticate, async (req, res) => {
  try {
    // Only admin and teachers can view all attendance
    if (req.user.role !== "admin" && req.user.role !== "teacher") {
      return res.status(403).json({ error: "Access denied" });
    }

    let query = {};

    // Teachers can only see attendance for their assigned subjects
    if (req.user.role === "teacher") {
      const teacher = await User.findById(req.user._id).populate(
        "assignedSubjects",
      );
      const subjectIds = teacher.assignedSubjects.map((subj) => subj._id);
      query.subject = { $in: subjectIds };
    }

    const attendance = await Attendance.find(query)
      .populate("student", "name rollNumber email")
      .populate("subject", "name code")
      .sort({ date: -1 });

    res.json(attendance);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Dashboard -------------------- */
app.get("/api/dashboard", async (req, res) => {
  try {
    const students = await Student.countDocuments();
    const attendance = await Attendance.find()
      .populate("student")
      .sort({ date: -1 });

    res.json({ students, attendance });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Live Recognition (with face coordinates) -------------------- */
app.post("/api/attendance/recognize-live", authenticate, async (req, res) => {
  console.log("[Backend] Received recognize-live request");
  try {
    const { imageBase64 } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: "Image required" });
    }

    // base64 → buffer
    const buffer = Buffer.from(
      imageBase64.replace(/^data:image\/\w+;base64,/, ""),
      "base64",
    );

    const formData = new FormData();
    formData.append("frame", buffer, {
      filename: "frame.jpg",
      contentType: "image/jpeg",
    });

    // Send to Flask AI
    // Increased timeout to 30 seconds - YOLOv8-face + ArcFace recognition can take time
    // First inference may take longer as models load into memory
    const aiRes = await axios.post(
      "http://127.0.0.1:8000/recognize-live",
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 30000, // 30 second timeout (increased from 5s for YOLOv8-face + ArcFace)
      },
    );

    console.log(`[Backend] AI Server returned ${aiRes.data.count || 0} results`);

    // Process all results from AI (Batch mode)
    const aiResults = aiRes.data.results || [];
    const processedRecognitions = [];
    const allDetectedFaces = [];

    for (const resItem of aiResults) {
      const { student_id, confidence, bbox, recognized } = resItem;
      const faceBox = { x: bbox[0], y: bbox[1], w: bbox[2], h: bbox[3] };

      allDetectedFaces.push(faceBox);

      if (recognized && student_id) {
        try {
          // Optimized: Fetch only needed fields
          const student = await Student.findById(student_id);
          if (student) {
            processedRecognitions.push({
              student,
              confidence,
              faceBox,
              recognized: true
            });
            console.log(`✓ Batch Recognized: ${student.name} (${student_id}) at ${(confidence * 100).toFixed(1)}%`);
          }
        } catch (dbErr) {
          console.error(`Error fetching student ${student_id}:`, dbErr.message);
        }
      }
    }

    // Return the comprehensive results to frontend
    res.json({
      recognized: processedRecognitions.length > 0,
      recognitions: processedRecognitions,
      faces: allDetectedFaces,
      count: aiResults.length
    });
  } catch (err) {
    console.error("Recognition error:", err.message);

    // Handle different types of errors
    if (err.code === "ECONNREFUSED") {
      return res.status(500).json({
        error:
          "AI server not running. Please start the AI server on port 8000.",
        recognized: false,
        faces: [],
        errorType: "connection_refused",
      });
    }

    if (err.code === "ECONNABORTED" || err.message.includes("timeout")) {
      console.error(
        "AI recognition timed out - this can happen with YOLOv8-face + ArcFace on slower systems",
      );
      return res.status(504).json({
        error:
          "Recognition timeout - AI processing took too long. Try again or check AI server performance.",
        recognized: false,
        faces: [],
        errorType: "timeout",
        suggestion:
          "The YOLOv8-face + ArcFace models may need more time. Consider optimizing or using a faster GPU.",
      });
    }

    res.status(500).json({
      error: "Recognition failed",
      details: err.message,
      recognized: false,
      faces: [],
      errorType: "unknown",
    });
  }
});

/* -------------------- Single Recognition (Backward compatibility) -------------------- */
app.post("/api/attendance/recognize", authenticate, async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).json({ error: "Image required" });

    const buffer = Buffer.from(imageBase64.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const formData = new FormData();
    formData.append("frame", buffer, { filename: "frame.jpg", contentType: "image/jpeg" });

    const aiRes = await axios.post("http://127.0.0.1:8000/recognize-live", formData, {
      headers: formData.getHeaders(),
      timeout: 30000,
    });

    const results = aiRes.data.results || [];
    if (results.length === 0) return res.json({ recognized: false, faces: [] });

    const bestRec = results.filter(r => r.recognized).sort((a, b) => (b.confidence || 0) - (a.confidence || 0))[0];

    if (bestRec) {
      const student = await Student.findById(bestRec.student_id);
      return res.json({
        recognized: true,
        student: student || { _id: bestRec.student_id, name: "Unknown" },
        confidence: bestRec.confidence,
        faceBox: { x: bestRec.bbox[0], y: bestRec.bbox[1], w: bestRec.bbox[2], h: bestRec.bbox[3] },
        faces: results.map(r => ({ x: r.bbox[0], y: r.bbox[1], w: r.bbox[2], h: r.bbox[3] }))
      });
    }

    res.json({
      recognized: false,
      faces: results.map(r => ({ x: r.bbox[0], y: r.bbox[1], w: r.bbox[2], h: r.bbox[3] }))
    });
  } catch (err) {
    console.error("Single recognize error:", err.message);
    res.status(500).json({ error: "Recognition failed", details: err.message });
  }
});

/* -------------------- Subjects -------------------- */
app.get("/api/subjects", authenticate, async (req, res) => {
  try {
    const subjects = await Subject.find();
    res.json(subjects);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/subjects", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "admin") {
      return res.status(403).json({ error: "Admin only" });
    }
    const { name, code } = req.body;
    const subject = await Subject.create({ name, code });
    res.json(subject);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get subjects with their assigned teachers (for student enrollment)
app.get("/api/subjects/with-teachers", authenticate, async (req, res) => {
  try {
    // All authenticated users can view this
    const subjects = await Subject.find().sort({ name: 1 });

    // Get all teachers with their assigned subjects
    const teachers = await User.find({ role: "teacher" })
      .select("name email assignedSubjects")
      .populate("assignedSubjects");

    // Build a map of subject -> teachers
    const subjectTeachersMap = {};

    teachers.forEach((teacher) => {
      if (teacher.assignedSubjects && teacher.assignedSubjects.length > 0) {
        teacher.assignedSubjects.forEach((subject) => {
          const subjectId = subject._id.toString();
          if (!subjectTeachersMap[subjectId]) {
            subjectTeachersMap[subjectId] = {
              subject: subject,
              teachers: [],
            };
          }
          subjectTeachersMap[subjectId].teachers.push({
            _id: teacher._id,
            name: teacher.name,
            email: teacher.email,
          });
        });
      }
    });

    // Format response: list subjects with their teachers
    const subjectsWithTeachers = subjects.map((subject) => {
      const subjectId = subject._id.toString();
      const teachersForSubject = subjectTeachersMap[subjectId]?.teachers || [];

      return {
        _id: subject._id,
        name: subject.name,
        code: subject.code,
        teachers: teachersForSubject,
        hasMultipleTeachers: teachersForSubject.length > 1,
      };
    });

    res.json(subjectsWithTeachers);
  } catch (err) {
    console.error("Get subjects with teachers error:", err);
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Settings -------------------- */
app.get("/api/settings", authenticate, async (req, res) => {
  try {
    // Teachers and admins can view settings
    if (req.user.role !== "admin" && req.user.role !== "teacher") {
      return res.status(403).json({ error: "Access denied" });
    }
    let settings = await Settings.findOne();
    if (!settings) {
      settings = await Settings.create({ allowManualAttendance: false });
    }
    res.json(settings);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put("/api/settings", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "admin") {
      return res.status(403).json({ error: "Admin only" });
    }
    let settings = await Settings.findOne();
    if (!settings) {
      settings = await Settings.create({
        allowManualAttendance: req.body.allowManualAttendance,
      });
    } else {
      settings.allowManualAttendance = req.body.allowManualAttendance;
      await settings.save();
    }
    res.json(settings);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Enrollments -------------------- */
// Student enrolls in a course/subject with a specific teacher
app.post("/api/enrollments", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "student") {
      return res.status(403).json({ error: "Students only" });
    }

    const { subjectId, teacherId } = req.body;
    if (!subjectId) {
      return res.status(400).json({ error: "subjectId is required" });
    }

    if (!teacherId) {
      return res
        .status(400)
        .json({
          error:
            "teacherId is required. Please select a teacher for this course.",
        });
    }

    // Get student ID from user (handle both populated and non-populated)
    let studentId = req.user.studentId;
    if (studentId && typeof studentId === "object" && studentId._id) {
      studentId = studentId._id;
    }

    if (!studentId) {
      return res
        .status(400)
        .json({ error: "Student profile not found. Please contact admin." });
    }

    // Verify student exists
    const student = await Student.findById(studentId);
    if (!student) {
      return res.status(404).json({ error: "Student record not found" });
    }

    // Check if subject exists
    const subject = await Subject.findById(subjectId);
    if (!subject) {
      return res.status(404).json({ error: "Subject not found" });
    }

    // Verify teacher exists and is assigned to this subject
    const teacher = await User.findById(teacherId).populate("assignedSubjects");
    if (!teacher || teacher.role !== "teacher") {
      return res.status(404).json({ error: "Teacher not found" });
    }

    // Check if teacher is assigned to this subject
    const teacherSubjectIds = (teacher.assignedSubjects || []).map((subj) => {
      if (subj && subj._id) {
        return subj._id.toString();
      }
      return subj.toString();
    });

    const subjectIdStr = subjectId.toString();
    const isTeacherAssigned =
      teacherSubjectIds.includes(subjectIdStr) ||
      (teacher.assignedSubjects || []).some((assignedId) => {
        const assignedIdStr =
          assignedId._id?.toString() || assignedId.toString();
        return assignedIdStr === subjectIdStr;
      });

    if (!isTeacherAssigned) {
      return res.status(400).json({
        error: `Teacher ${teacher.name} is not assigned to this subject. Please select a different teacher.`,
      });
    }

    // Check if enrollment already exists (same student, subject, AND teacher)
    const existingEnrollment = await Enrollment.findOne({
      student: studentId,
      subject: subjectId,
      teacher: teacherId,
    });

    if (existingEnrollment) {
      return res.status(400).json({
        error: `Already enrolled or pending enrollment with ${teacher.name}`,
        enrollment: existingEnrollment,
      });
    }

    // Create enrollment request
    const enrollment = await Enrollment.create({
      student: studentId,
      subject: subjectId,
      teacher: teacherId,
      status: "pending",
    });

    const populatedEnrollment = await Enrollment.findById(enrollment._id)
      .populate("student")
      .populate("subject")
      .populate("teacher", "name email");

    console.log(
      `✓ Enrollment request created: Student ${student.name} -> Subject ${subject.name} -> Teacher ${teacher.name}`,
    );

    res.json({
      message: `Enrollment request submitted successfully to ${teacher.name}`,
      enrollment: populatedEnrollment,
    });
  } catch (err) {
    console.error("Enrollment error:", err);

    // Handle duplicate key error (MongoDB unique index)
    if (err.code === 11000) {
      return res.status(400).json({
        error: "Already enrolled or pending enrollment with this teacher",
      });
    }

    res
      .status(500)
      .json({ error: err.message || "Failed to process enrollment request" });
  }
});

// Get enrolled courses for a student
app.get("/api/enrollments/student", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "student") {
      return res.status(403).json({ error: "Students only" });
    }

    // Handle both populated and non-populated studentId
    let studentId = req.user.studentId;
    if (studentId && typeof studentId === "object" && studentId._id) {
      studentId = studentId._id;
    }

    if (!studentId) {
      return res.status(400).json({ error: "Student profile not found" });
    }

    const enrollments = await Enrollment.find({ student: studentId })
      .populate("subject")
      .populate("teacher", "name email")
      .populate("reviewedBy", "name email")
      .sort({ createdAt: -1 });

    res.json(enrollments);
  } catch (err) {
    console.error("Get student enrollments error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get all enrollments (for admin/teacher)
app.get("/api/enrollments", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "admin" && req.user.role !== "teacher") {
      return res.status(403).json({ error: "Access denied" });
    }

    const { status, subjectId } = req.query;
    const query = {};

    if (status) {
      query.status = status;
    }

    // If teacher, only show enrollments specifically assigned to them (teacher field matches)
    if (req.user.role === "teacher") {
      query.teacher = req.user._id; // Only show enrollments where this teacher is the assigned teacher
    }

    if (subjectId) {
      query.subject = subjectId;
    }

    const enrollments = await Enrollment.find(query)
      .populate("student")
      .populate("subject")
      .populate("teacher", "name email")
      .populate("reviewedBy", "name email")
      .sort({ createdAt: -1 });

    res.json(enrollments);
  } catch (err) {
    console.error("Get enrollments error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get students enrolled in a specific course (for teacher) - only shows students enrolled with THIS teacher
app.get(
  "/api/enrollments/course/:subjectId",
  authenticate,
  async (req, res) => {
    try {
      if (req.user.role !== "teacher" && req.user.role !== "admin") {
        return res.status(403).json({ error: "Access denied" });
      }

      const { subjectId } = req.params;

      // Build query for approved enrollments
      const query = {
        subject: subjectId,
        status: "approved",
      };

      // If teacher, only show students enrolled specifically with THIS teacher
      if (req.user.role === "teacher") {
        // Verify teacher is assigned to this subject first
        const teacher = await User.findById(req.user._id).populate(
          "assignedSubjects",
        );

        if (
          !teacher ||
          !teacher.assignedSubjects ||
          teacher.assignedSubjects.length === 0
        ) {
          return res.status(403).json({ error: "No subjects assigned to you" });
        }

        const assignedSubjectIds = teacher.assignedSubjects.map((subj) => {
          if (subj && subj._id) {
            return subj._id.toString();
          }
          return subj.toString();
        });

        const subjectIdStr = subjectId.toString();
        const isAssigned = assignedSubjectIds.includes(subjectIdStr);

        if (!isAssigned) {
          return res
            .status(403)
            .json({ error: "Not assigned to this subject" });
        }

        // Only show enrollments where teacher field matches this teacher
        query.teacher = req.user._id;
      }

      const enrollments = await Enrollment.find(query)
        .populate("student")
        .populate("subject")
        .populate("teacher", "name email")
        .sort({ createdAt: -1 });

      // Extract just the student information
      const students = enrollments.map((enrollment) => enrollment.student);

      res.json(students);
    } catch (err) {
      console.error("Get course enrollments error:", err);
      res.status(500).json({ error: err.message });
    }
  },
);

// Teacher approves/rejects enrollment
app.put("/api/enrollments/:enrollmentId", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "teacher" && req.user.role !== "admin") {
      return res.status(403).json({ error: "Teachers/Admins only" });
    }

    const { enrollmentId } = req.params;
    const { status } = req.body;

    if (!["approved", "rejected"].includes(status)) {
      return res
        .status(400)
        .json({ error: "Status must be 'approved' or 'rejected'" });
    }

    const enrollment = await Enrollment.findById(enrollmentId)
      .populate("subject")
      .populate("teacher", "name email");

    if (!enrollment) {
      return res.status(404).json({ error: "Enrollment not found" });
    }

    // If teacher, verify this enrollment is specifically assigned to them
    if (req.user.role === "teacher") {
      // Check if this enrollment's teacher field matches the current teacher
      let enrollmentTeacherId;
      if (
        enrollment.teacher &&
        typeof enrollment.teacher === "object" &&
        enrollment.teacher._id
      ) {
        enrollmentTeacherId = enrollment.teacher._id.toString();
      } else {
        enrollmentTeacherId =
          enrollment.teacher?.toString() || enrollment.teacher;
      }
      const currentTeacherId = req.user._id.toString();

      if (enrollmentTeacherId !== currentTeacherId) {
        console.error(
          `❌ Teacher ${req.user._id} (${req.user.email}) cannot approve enrollment ${enrollmentId}`,
        );
        console.error(
          `   Enrollment is assigned to teacher: ${enrollmentTeacherId}`,
        );
        console.error(`   Current teacher: ${currentTeacherId}`);

        const teacherName = enrollment.teacher?.name || "Unknown teacher";

        return res.status(403).json({
          error: `This enrollment request is assigned to ${teacherName}. Only that teacher can approve or reject it.`,
        });
      }

      console.log(
        `✓ Teacher ${req.user._id} is the assigned teacher for this enrollment, approval allowed`,
      );
    }

    enrollment.status = status;
    enrollment.reviewedAt = new Date();
    enrollment.reviewedBy = req.user._id;
    await enrollment.save();

    const updatedEnrollment = await Enrollment.findById(enrollmentId)
      .populate("student")
      .populate("subject")
      .populate("teacher", "name email")
      .populate("reviewedBy", "name email");

    res.json({
      message: `Enrollment ${status}`,
      enrollment: updatedEnrollment,
    });
  } catch (err) {
    console.error("Enrollment approval error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get attendance for a specific course (student view)
app.get(
  "/api/students/:studentId/attendance/course/:subjectId",
  authenticate,
  async (req, res) => {
    try {
      const { studentId, subjectId } = req.params;

      // Verify student can only view their own attendance (unless admin/teacher)
      if (req.user.role === "student") {
        // Handle populated studentId (object with _id) or non-populated (ObjectId/string)
        let userStudentId;
        if (
          req.user.studentId &&
          typeof req.user.studentId === "object" &&
          req.user.studentId._id
        ) {
          // Populated: extract _id
          userStudentId = req.user.studentId._id.toString();
        } else if (req.user.studentId) {
          // Not populated: already ObjectId or string
          userStudentId = req.user.studentId.toString();
        }

        if (!userStudentId || userStudentId !== studentId) {
          console.error(
            `Access denied: user.studentId=${userStudentId}, requested studentId=${studentId}`,
          );
          return res
            .status(403)
            .json({
              error: "Access denied. You can only view your own attendance.",
            });
        }
      }

      // Check if student is enrolled and approved (only for students, not for admin/teacher)
      if (req.user.role === "student") {
        const enrollment = await Enrollment.findOne({
          student: studentId,
          subject: subjectId,
          status: "approved",
        });

        if (!enrollment) {
          return res
            .status(403)
            .json({
              error: "You are not enrolled or not approved in this course",
            });
        }
      }

      // Fetch attendance records
      const attendanceRecords = await Attendance.find({
        student: studentId,
        subject: subjectId,
      })
        .populate("subject", "name code")
        .sort({ date: -1 });

      console.log(
        `✓ Returning ${attendanceRecords.length} attendance records for student ${studentId}, subject ${subjectId}`,
      );

      res.json(attendanceRecords);
    } catch (err) {
      console.error("Get attendance by course error:", err);
      res
        .status(500)
        .json({ error: err.message || "Failed to fetch attendance" });
    }
  },
);

// Bulk mark attendance for enrolled students in a course (teacher)
app.post("/api/attendance/bulk", authenticate, async (req, res) => {
  try {
    if (req.user.role !== "teacher") {
      return res.status(403).json({ error: "Teachers only" });
    }

    const settings = await Settings.findOne();
    if (!settings?.allowManualAttendance) {
      return res
        .status(403)
        .json({ error: "Manual attendance not allowed by admin" });
    }

    const { subjectId, date, attendances } = req.body; // attendances: [{studentId, status}]

    if (!subjectId || !date || !Array.isArray(attendances)) {
      return res
        .status(400)
        .json({ error: "subjectId, date, and attendances array required" });
    }

    // Verify teacher is assigned to this subject
    if (
      !req.user.assignedSubjects ||
      !req.user.assignedSubjects.includes(subjectId)
    ) {
      return res.status(403).json({ error: "Not assigned to this subject" });
    }

    // Verify subject exists
    const subject = await Subject.findById(subjectId);
    if (!subject) {
      return res.status(404).json({ error: "Subject not found" });
    }

    const attendanceDate = new Date(date);
    const today = new Date(attendanceDate);
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const results = [];
    const errors = [];

    for (const att of attendances) {
      const { studentId, status } = att;

      // Verify student is enrolled and approved (with any teacher for this subject)
      const enrollment = await Enrollment.findOne({
        student: studentId,
        subject: subjectId,
        status: "approved",
      });

      if (!enrollment) {
        errors.push({
          studentId,
          error: "Student not enrolled or not approved in this course",
        });
        continue;
      }

      try {
        // Check for duplicate attendance
        const existing = await Attendance.findOne({
          student: studentId,
          subject: subjectId,
          date: { $gte: today, $lt: tomorrow },
        });

        if (existing) {
          // Update existing attendance
          existing.status = status || "present";
          existing.markedBy = "manual";
          await existing.save();
          results.push({
            studentId,
            action: "updated",
            attendanceId: existing._id,
          });
        } else {
          // Create new attendance
          const attendance = await Attendance.create({
            student: studentId,
            subject: subjectId,
            date: attendanceDate,
            markedBy: "manual",
            status: status || "present",
          });
          results.push({
            studentId,
            action: "created",
            attendanceId: attendance._id,
          });
        }
      } catch (err) {
        errors.push({ studentId, error: err.message });
      }
    }

    res.json({
      message: `Processed ${results.length} attendances`,
      results,
      errors: errors.length > 0 ? errors : undefined,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/* -------------------- Error Handling Middleware -------------------- */
// Catch-all error handler - always return JSON
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(err.status || 500).json({
    error: err.message || "Internal server error",
    details: process.env.NODE_ENV === "development" ? err.stack : undefined,
  });
});

// 404 handler - always return JSON
app.use((req, res) => {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.path}` });
});

/* -------------------- Server -------------------- */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
  console.log(`📋 Available routes:`);
  console.log(`   POST /api/auth/login`);
  console.log(`   POST /api/students/register (self-registration)`);
  console.log(`   GET  /api/students (with ?search=)`);
  console.log(`   PUT  /api/students/:id/reset-password`);
  console.log(`   DELETE /api/students/:id`);
  console.log(`   POST /api/teachers/register`);
  console.log(`   PUT  /api/teachers/:id/reset-password`);
  console.log(`   POST /api/subjects`);
  console.log(`   GET  /api/subjects`);
  console.log(`   GET  /api/teachers`);
  console.log(`   POST /api/attendance/mark`);
  console.log(`   POST /api/attendance/bulk`);
  console.log(`   POST /api/attendance/recognize-live`);
  console.log(`   POST /api/enrollments`);
  console.log(`   GET  /api/enrollments`);
  console.log(`   PUT  /api/enrollments/:id`);
});
