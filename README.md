# 🏸 Badminton Video Scoring App

An **Expo + React Native** mobile application that takes **badminton match videos** as input and produces **automated scoring and analytics** using computer vision and machine learning.

This project aims to bridge **sports + software + AI**, focusing on offline/online video processing to detect rallies, shots, player actions, and scoring events in badminton matches.

---

## 🚀 Vision

Badminton scoring is fast-paced and difficult to track manually. This app aims to:

* Accept **video input** (recorded or uploaded)
* Analyze gameplay using **computer vision models**
* Automatically **detect shots, rallies, and points**
* Generate **match scores, statistics, and insights**
* Serve as a foundation for advanced analytics (player performance, shot types, heatmaps)

---

## ✨ Key Features (Planned & In Progress)

### 📹 Video Input

* Upload video from device gallery
* (Future) Record video directly inside the app

### 🧠 AI-Powered Analysis

* Shuttle detection & tracking (TrackNet / YOLO-based)
* Player detection & pose estimation
* Shot detection (smash, clear, drop, net shot)
* Rally segmentation
* Contact detection between racket & shuttle

### 🧮 Scoring Engine

* Automatic point detection
* Rally-based score updates
* Match progression tracking
* Support for singles & doubles (future)

### 📊 Output & Insights

* Final match score
* Rally count & duration
* Shot distribution
* Player movement analysis (future)

---

## 🧱 Tech Stack

### Frontend (This Repository)

* **Expo (SDK 54)**
* **React Native**
* **Expo Router**
* **TypeScript**
* **Expo Image Picker** – video selection
* **Expo Image** – efficient rendering

### Backend / ML (Separate Service)

* **FastAPI** (Python)
* **OpenCV** for video frame processing
* **YOLOv8** – player & shuttle detection
* **TrackNet** – shuttle tracking
* **ST-GCN / Temporal Models** – action & shot detection

> ⚠️ This repository focuses on the **mobile application layer**. The ML pipeline runs as a separate backend service.

---

## 📁 Project Structure

```text
app/
 ├── (tabs)/
 │   ├── index.tsx        # Home screen
 │   ├── upload.tsx       # Video upload & preview
 │   └── results.tsx      # Scoring & analytics output
 ├── _layout.tsx          # App layout & routing

components/
 ├── ThemedView.tsx
 ├── ThemedText.tsx

assets/
 ├── images/

scripts/
 └── reset-project.js

package.json
app.json
README.md
```

---

## 🛠️ Installation & Setup

### 1️⃣ Prerequisites

* Node.js (>= 18 recommended)
* npm or yarn
* Expo CLI (local)

```bash
npm install -g expo
```

---

### 2️⃣ Install Dependencies

```bash
npm install
```

---

### 3️⃣ Start the App

```bash
npm start
```

Run on specific platforms:

```bash
npm run android
npm run ios
npm run web
```

---

## 📱 How the App Works (High Level)

1. User selects a **badminton match video**
2. Video is uploaded to the backend API
3. Backend:

   * Extracts frames
   * Detects players & shuttle
   * Tracks rallies & shots
   * Computes score logic
4. Processed results are returned to the app
5. App displays:

   * Match score
   * Rally stats
   * Visual summaries

---

## 🔌 Backend API (Planned Interface)

Example:

```http
POST /analyze-video
Content-Type: multipart/form-data

video=<match.mp4>
```

Response:

```json
{
  "score": { "playerA": 21, "playerB": 18 },
  "rallies": 39,
  "shots": {
    "smash": 12,
    "drop": 8,
    "clear": 15
  }
}
```

---

## 🧠 ML Roadmap

* [ ] Shuttle detection with TrackNet
* [ ] Player pose estimation (YOLOv8-Pose)
* [ ] Shot classification using temporal models
* [ ] Rally segmentation logic
* [ ] Robust scoring rules for badminton
* [ ] Offline video processing support

---

## 🧪 Current Status

🚧 **Work in Progress**

* App scaffolding complete
* Video picker integration in progress
* Backend experimentation with YOLO & TrackNet

---

## 🌱 Future Enhancements

* Real-time scoring
* Match replay with overlays
* Cloud processing & job status tracking
* Player comparison dashboards
* Coach & training mode

---

## 🤝 Contributions

Contributions, ideas, and feedback are welcome!

* Fork the repo
* Create a feature branch
* Submit a pull request

---

## 📜 License

This project is currently under development and not licensed for commercial use.

---

## 👤 Author

**Chirag Chandrashekar**
AI / ML Engineer | Sports Analytics Enthusiast

---

If you love **sports + computer vision + AI**, this project is for you 🏸🤖
