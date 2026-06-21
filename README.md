# Attendify

## Overview

Attendify is an AI-powered attendance management platform that automates student attendance using facial recognition and computer vision technologies.

Traditional attendance systems are time-consuming, prone to human error, and vulnerable to proxy attendance. Attendify addresses these challenges by combining machine learning, facial recognition, and real-time camera processing to identify students automatically and record attendance instantly.

The platform provides a complete web-based solution for managing students, teachers, courses, enrollments, and attendance records through an intuitive dashboard.

---

## Features

### AI-Powered Facial Recognition

* Automatic student identification
* Real-time face detection from live camera feeds
* Multi-angle face training for improved accuracy
* Recognition from trained facial datasets
* Attendance marking without manual intervention

### Automated Attendance Tracking

* Real-time attendance recording
* Instant presence verification
* Attendance history management
* Course-specific attendance tracking

### Student Management

* Student registration portal
* Video-based face enrollment
* Profile management
* Enrollment tracking

### Teacher Management

* Teacher registration and management
* Course assignment
* Attendance monitoring

### Course Management

* Create and manage courses
* Student enrollments
* Attendance records by course

### Dashboard & Analytics

* Attendance overview
* Student statistics
* Subject management
* Course enrollment monitoring

---

## Problems Addressed

Traditional attendance systems often face:

### Time Consumption

Manual roll calls interrupt lectures and consume valuable class time.

### Proxy Attendance

Students can mark attendance on behalf of absent classmates.

### Poor Scalability

Managing attendance across multiple classes becomes increasingly difficult.

### Human Errors

Manual attendance recording can lead to inaccurate records.

### Security Concerns

Traditional systems lack reliable identity verification.

---

## How Attendify Works

```text
Student Registration
         │
         ▼
Video Capture
(10-second facial recording)
         │
         ▼
Frame Extraction
         │
         ▼
AI Model Training
         │
         ▼
Live Camera Detection
         │
         ▼
Face Recognition
         │
         ▼
Automatic Attendance Marking
         │
         ▼
MongoDB Storage
```

---

## System Modules

### Authentication Module

* Secure login system
* Session management
* Role-based access

### Student Module

* Register students
* Upload facial data
* View attendance history
* Manage profile information

### Teacher Module

* Manage courses
* Monitor attendance
* View enrolled students

### Attendance Module

* Automated attendance recording
* Manual attendance support
* Attendance history
* Course attendance management

### AI Recognition Module

* Face detection
* Face encoding generation
* Facial recognition matching
* Live camera monitoring

---

## Technology Stack

### Frontend

* React.js
* Tailwind CSS
* PostCSS
* JavaScript

### Backend

* Node.js
* Express.js
* REST APIs

### Database

* MongoDB

### Artificial Intelligence

* Python
* OpenCV
* Face Recognition
* Computer Vision

### Integrations

* Webcam Camera Feed
* REST API Communication

---

## Project Structure

```text
Attendify/
│
├── ai/                    # AI recognition and training modules
│
├── backend/               # Node.js & Express backend
│
├── frontend/              # React frontend application
│
├── Attendify.pptx         # Project presentation
│
└── README.md
```

---

## Key Functionalities

### Student Registration

Students are enrolled by recording a short video clip containing multiple facial angles. The captured frames are used to train the recognition model.

### Face Recognition Training

The AI model processes extracted video frames and generates facial encodings used for future identification.

### Live Attendance Detection

The system continuously monitors the camera feed and identifies students in real time.

### Automatic Attendance Marking

Once a student is recognized, attendance is instantly recorded and stored in the database.

### Attendance Management

Administrators and teachers can review attendance records through a centralized dashboard.

---

## Benefits

### High Recognition Accuracy

Multi-angle facial training improves identification accuracy across different orientations and lighting conditions.

### Real-Time Attendance

Attendance is recorded immediately when a student is detected.

### Reduced Administrative Work

Eliminates manual roll calls and repetitive attendance management tasks.

### Improved Security

Prevents proxy attendance through facial verification.

### Scalability

Designed to support multiple classrooms, courses, and student groups.

---

## Future Enhancements

* Mobile application support
* Liveness detection to prevent spoofing
* Cloud deployment
* Advanced attendance analytics
* Automated reporting
* Multi-camera classroom support
* Real-time notifications
* Biometric verification enhancements

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Attendify.git
cd Attendify
```

### Backend Setup

```bash
cd backend
npm install
npm start
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### AI Module Setup

```bash
cd ai
pip install -r requirements.txt
python main.py
```

---

## Results

Attendify successfully demonstrates how Artificial Intelligence and Computer Vision can modernize attendance management by:

* Automating attendance processes
* Increasing attendance accuracy
* Eliminating proxy attendance
* Reducing administrative overhead
* Providing real-time attendance monitoring
* Improving data reliability and security

