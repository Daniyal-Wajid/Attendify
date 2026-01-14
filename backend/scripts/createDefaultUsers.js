// scripts/createDefaultUsers.js
// Run this script once to create default admin and teacher users
// node scripts/createDefaultUsers.js

import mongoose from "mongoose";
import bcrypt from "bcryptjs";
import dotenv from "dotenv";
import User from "../models/User.js";

dotenv.config();

async function createDefaultUsers() {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected to MongoDB");

    // Create default admin
    const adminEmail = "admin@school.com";
    const adminExists = await User.findOne({ email: adminEmail });
    if (!adminExists) {
      const adminPassword = await bcrypt.hash("admin123", 10);
      await User.create({
        email: adminEmail,
        password: adminPassword,
        role: "admin",
        name: "Admin User",
      });
      console.log("✅ Admin user created:");
      console.log("   Email: admin@school.com");
      console.log("   Password: admin123");
    } else {
      console.log("ℹ️  Admin user already exists");
    }

    // Create default teacher
    const teacherEmail = "teacher@school.com";
    const teacherExists = await User.findOne({ email: teacherEmail });
    if (!teacherExists) {
      const teacherPassword = await bcrypt.hash("teacher123", 10);
      await User.create({
        email: teacherEmail,
        password: teacherPassword,
        role: "teacher",
        name: "Teacher User",
      });
      console.log("✅ Teacher user created:");
      console.log("   Email: teacher@school.com");
      console.log("   Password: teacher123");
    } else {
      console.log("ℹ️  Teacher user already exists");
    }

    console.log("\n✅ Default users setup complete!");
    process.exit(0);
  } catch (err) {
    console.error("Error:", err);
    process.exit(1);
  }
}

createDefaultUsers();

