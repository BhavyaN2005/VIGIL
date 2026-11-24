# VIGIL: Explainable AI Network Intrusion Detection System (XAI-NIDS)
## Comprehensive Technical Context & Architectural Reference

---

### 1. Executive Summary & Core Mission
**VIGIL** is an enterprise-grade, high-performance Network Intrusion Detection System (NIDS) powered by machine learning, sequential deep learning, and unsupervised anomaly detection, engineered with real-time **Explainable AI (XAI)**. Traditional signature-based intrusion detection systems (Snort, Suricata, Bro/Zeek) fall short against zero-day exploits, polymorphic network malware, and high-velocity volumetric attacks across cloud, enterprise, and IoT perimeters. Furthermore, "black box" deep learning models trigger opaque alerts that overwhelm Security Operations Center (SOC) tier-1/tier-2 analysts with false alarms without contextual rationale.

VIGIL resolves this dilemma by pairing multi-model detection (Random Forest, XGBoost, LSTM, and KitNET Autoencoders) with real-time feature attribution (SHAP values, feature importance waterfalls, temporal flow statistics), wrapped in an ultra-clean **Swiss-Grid Cyber Architecture** interface designed for high-density, mission-critical threat operations.

---

### 2. Dataset Benchmarks & Telemetry Engineering

#### 2.1 Kitsune Network Attack Dataset (Primary High-Density Feature Set)
* **Feature Topology:** 115 damped incremental statistical features (`Feature_0` through `Feature_114`) extracted per packet flow across 5 temporal decay windows ($\lambda = [100\text{ms}, 500\text{ms}, 1.5\text{s}, 10\text{s}, 1\text{min}]$).
* **Statistical Dimensions:**
  * **1D Frame Statistics (Weight, Mean, Variance/Std):** Packet arrival jitter, flow duration, packet length distribution.
  * **2D Channel Covariance & Correlation:** Covariance between source IP/MAC and destination IP/MAC, cross-stream packet size variance.
* **Attack Scenario Vectors Evaluated:**
  * `ARP Man-in-the-Middle (MitM)`: Layer 2 spoofing and poison packets (100.0% detection recall).
  * `Fuzzing`: High-frequency malformed packet injection across unexpected port ranges (99.93% accuracy).
  * `Mirai Botnet`: Distributed scanner and Telnet/SSH credential brute-force telemetry (96.30% detection).
  * `Active Wiretap & OS Fingerprinting`: High-rate SYN scanning and stealth probe packets.
  * `SSDP DoS / SYN Flooding`: High-volume layer 4 denial-of-service packet bursts.

#### 2.2 CIC-IDS2017 Benchmark Flow Engine
* **Flow Statistics:** 78 directional network flow attributes (Flow Duration, Fwd/Bwd Packet Length Mean, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, FIN/SYN/RST/PSH/ACK/URG flag counts, Down/Up Ratio, Average Packet Size).
* **Attack Profiles:** PortScan, DDoS LOIC/HOIC, DoS GoldenEye/Slowloris, Infiltration, Web Attacks (SQLi, XSS, Brute Force).

---

### 3. Model Architecture & Detection Stack

```
Incoming Network Flow (PCAP / Live Socket / CSV Batch)
                       │
                       ▼
       Feature Extraction & Sanitization (115 Kitsune / 78 CIC-IDS2017)
                       │
                       ▼
       Pre-computed Feature Scaling (StandardScaler `scaler_final.pkl`)
                       │
       ┌───────────────┴───────────────────────────────┐
       ▼                                               ▼
Supervised Ensemble Classifiers                 Unsupervised Anomaly Core
├── Random Forest (`random_forest_final.pkl`)    └── KitNET Deep Autoencoder
│   • 100 Trees, Gini Impurity                      • Bottleneck Reconstruction Error
│   • 99.8% Overall Precision                       • Dynamic Adaptive Threshold ($R_{err} > \tau$)
├── XGBoost Classifier                           Sequential Temporal Model
│   • Depth 6, Subsample 0.8                     └── LSTM Recurrent Network
│   • Fast gradient-boosted decision trees           • 64 Hidden Units, Dropout 0.2
       │                                               • Long-range temporal pattern analysis
       └───────────────┬───────────────────────────────┘
                       │
                       ▼
        Inference Aggregation & Alert Scoring Engine
                       │
                       ▼
          Explainable AI (XAI) Attribution
          ├── TreeSHAP (Local Waterfall & Force Plots)
          ├── Top-5 Malicious Feature Deviations
          └── Analyst Mitigation Recommendation Engine
```

---

### 4. Visual Direction: "Swiss-Grid Cyber Architecture"

* **Aesthetic Philosophy:** Inspired by mid-century Swiss graphic design (Josef Müller-Brockmann, Massimo Vignelli) combined with modern high-contrast dark military/aerospace security operations consoles. No generic neon purple or saturated blobs; every line, label, and grid cell serves a precise spatial and telemetry purpose.
* **Color Palette & Design Tokens:**
  * **Primary Canvas:** Deep Carbon Obsidian (`#07090E`)
  * **Elevated Glass Surface:** Midnight Slate (`#0D121D` at 82% opacity with `backdrop-blur-md`)
  * **Hairlines:** Crisp `1px` geometric borders in `rgba(255, 255, 255, 0.10)`
  * **Primary Telemetry Accent:** Electric Cyan (`#00E5FF`) & Cobalt Glow (`#0284C7`)
  * **Kinetic Focal Spark:** Luminous Vermilion-Coral (`#FF4D2D`) for hero caustic core, real-time threat pulses, and high-priority anomalies
  * **Severity Indicators:** Low (`#10B981`), Medium (`#F59E0B`), High (`#FF4D2D`), Critical (`#EF4444`)
* **Typography:**
  * Display & Headings: `Space Grotesk` (tracking `-0.03em`)
  * Interface & Body: `Inter`
  * Telemetry & Monospace Data: `JetBrains Mono` (`tabular-nums`)

---

### 5. System Monorepo Topology
```
Vigil/
├── backend/                  # FastAPI REST & WebSocket Backend
│   ├── app/
│   │   ├── core/             # Model registry, risk scoring, MITRE ATT&CK, CTI, async SQLite DB
│   │   ├── routers/          # API endpoints (health, predict, stream, alerts, forecast, topology, metrics)
│   │   ├── services/         # Inference orchestration, alert persistence, stream simulation
│   │   ├── schemas/          # Flow, alert, health Pydantic data schemas
│   │   ├── models/           # SQLAlchemy Alert DB entity model
│   │   ├── config.py         # Application settings & environment path resolution
│   │   └── main.py           # FastAPI application factory & lifespan manager
│   ├── tests/                # Automated pytest API test suite (23 tests)
│   ├── requirements.txt      # Python backend dependencies
│   └── vigil.db              # Local SQLite threat database
│
├── frontend/                 # High-Performance Swiss-Grid Presentation Layer
│   ├── index.html            # Landing Page, Architectural Overview & Model Benchmarks
│   ├── console.html          # VIGIL Threat Operations & Spatial Attack Forecasting Console
│   ├── dashboard.html        # Backward-compatibility alias
│   ├── components/           # Standalone UI components (glass capsule)
│   ├── js/                   # liquid-glass-engine.js (WebGL Three.js 3D caustic liquid glass shader)
│   └── vercel.json           # Frontend direct Vercel deployment configuration
│
├── ml/                       # Machine Learning & Explainable AI Pipeline
│   ├── models/               # PyTorch deep learning architectures (autoencoder_net.py, lstm_net.py)
│   ├── artifacts/            # Trained model weights, scalers, label map, thresholds
│   ├── preprocess.py         # 78-feature engineering, sanitization & scaling
│   ├── train.py              # Supervised & unsupervised training pipelines
│   ├── explain.py            # TreeExplainer & SHAP feature attribution
│   ├── config.py             # Hyperparameters, data paths & canonical taxonomy
│   └── tests/                # Automated pytest ML unit tests (12 tests)
│
├── data/                     # Benchmark Telemetry Datasets
│   ├── sample/sample_flows.csv  # Balanced CIC-IDS2017 & Kitsune flow telemetry
│   └── generate_sample.py    # Synthetic wire-speed flow generation utility
│
├── vercel.json               # Monorepo root Vercel URL rewrites
├── pytest.ini                # Pytest configuration (backend/tests & ml/tests)
├── PROJECT_CONTEXT.md        # Technical architecture documentation
└── DATASET.md                # Dataset topology documentation
```
