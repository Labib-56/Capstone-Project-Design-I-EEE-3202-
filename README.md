# EEE 3202 — Capstone Project Design I
## About the Course
**EEE 3202 — Capstone Project Design I**  
Department of Electrical & Electronic Engineering  
Rajshahi University of Engineering & Technology (RUET)

Capstone Project Design I is a core course in the undergraduate EEE curriculum at RUET 
where students independently identify a real-world engineering problem and develop a 
complete hardware-software solution from scratch. The project spans problem definition, 
literature review, system design, implementation, testing, and final presentation — 
simulating a real professional engineering project lifecycle. Students attend this course throughout the 3rd year even semester and submit the project at the end of 3rd year even semester. The theme of project reflects the knowledge gained from 1st year odd semester to 3rd year odd semester.

---

## What I Built
### Real-Time Edge-Based Facial Recognition System Using Deep Learning on Raspberry Pi 4
A standalone, offline facial recognition system that runs entirely on a **Raspberry Pi 4** 
using a quantized deep learning pipeline — no cloud, no internet, no external AI accelerator.

### The Problem
Most facial recognition systems depend on cloud computing, which raises concerns about:
- **Data privacy** — biometric data sent to external servers
- **Network latency** — real-time performance depends on internet speed
- **Cost** — recurring cloud API fees

Traditional edge solutions (like Dlib/HOG) are too slow on low-cost hardware, 
often delivering below 1 FPS on a Raspberry Pi.

### My Solution
Implemented a **YuNet + SFace** pipeline via OpenCV's DNN module:
- **YuNet** — a quantized CNN for fast face detection (~180ms on ARM CPU)
- **SFace** — generates 128-dimensional face embeddings for recognition
- **Cosine Similarity Matching** against a local `.pickle` database

### Key Features
- ✅ Real-time performance: **3–4 FPS** on native ARM CPU
- ✅ Fully **offline** — no cloud dependency
- ✅ **One-Shot enrollment** — register a new user in under 60 seconds without retraining
- ✅ Detects faces up to **±30° yaw** (side profiles)
- ✅ Simultaneous tracking of registered and unregistered (Unknown) faces
- ✅ Total hardware cost: **BDT 16,600** (~USD 150)

---

## Tech Stack
| Component | Detail |
|---|---|
| Hardware | Raspberry Pi 4 Model B (4GB) |
| Camera | Pi Camera V1.3 (5MP, OV5647) |
| OS | Raspberry Pi OS Bookworm (Debian 12, 64-bit) |
| Language | Python 3.11 |
| CV Library | OpenCV 4.6.0 (DNN Module) |
| Camera Driver | Picamera2 |
| Face Detector | YuNet (2022mar ONNX) |
| Face Recognizer | SFace (2021dec ONNX) |

---

## Results
| Metric | Target | Achieved |
|---|---|---|
| Frame Rate | ≥ 3 FPS | 3–4 FPS ✅ |
| Inference Latency | < 300ms | ~250ms ✅ |
| True Positive Rate | > 95% | > 98% ✅ |
| False Acceptance Rate | < 1% | < 0.5% ✅ |

---

## Project Info
- **Student:** Labib Marwan Hoque (Roll: 2101064)
- **Supervisor:** Md. Mayenul Islam, Assistant Professor, EEE, RUET
- **Showcased:** January 12, 2026
