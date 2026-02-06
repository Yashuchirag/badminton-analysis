# 🏸 Badminton Video Scoring App

# Video Preview

https://github.com/user-attachments/assets/030b32bc-8911-4777-a746-f64145b064ea

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

* Upload video from device gallery / Record video directly inside the app 

### 🧠 AI-Powered Analysis

* Shuttle detection & tracking (TrackNet and YOLO-based)
* Rally segmentation

### 🧮 Scoring Engine

* Automatic point detection
* Rally-based score updates
* Match progression tracking
* Support for singles & doubles (future)

### 📊 Output & Insights

* Player movement analysis
* Shuttle tracking
* Rally count & duration (future)
* Shot distribution (future)
* Final match score (future)


---

## 🧱 Tech Stack

### Frontend (This Repository)

* **Expo (SDK 54)**
* **React Native**
* **Expo Router**
* **TypeScript**
* **Expo Video Picker** – video selection
* **Expo Video Player** – video playback

### Backend / ML (Separate Service)

* **FastAPI** (Python)
* **OpenCV** for video frame processing
* **YOLOv8** – player & shuttle detection
* **TrackNet** – shuttle tracking

> ⚠️ This repository focuses on the **mobile application layer**. The ML pipeline runs as a separate backend service.

---

## 📁 Project Structure

```text
app/
 ├── index.tsx        # Home screen
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

```
Video
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
* Video picker integration completed
* Backend experimentation with YOLO & TrackNet completed
* Court mapping pending
* Shot detection pending
* Rally segmentation pending
* Scoring logic pending

---

## 🌱 Future Enhancements

* Real-time scoring
* Match replay with overlays
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
