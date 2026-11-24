# VIGIL: Dataset Reference & Preprocessing Specification
## Benchmark: CIC-IDS2017 (Canadian Institute for Cybersecurity)

---

### 1. Dataset Profile & Hardware Allocation Strategy
* **Full Dataset:** ~2.83 Million network flow records across 8 CSV files (Monday to Friday captures), ~78 numeric flow features.
* **Hardware Strategy (Local 8 GB RAM vs. Cloud):**
  * **Local Development & PyTest:** Powered by `data/sample/sample_flows.csv` (~5,000 stratified flow rows) for zero-OOM local testing, instant test runs, and frontend mock feeds.
  * **Production / Benchmark Training:** CLI arguments (`--data-dir`, `--output-dir`, `--sample-mode`) enable running the exact same `ml/train.py` pipeline seamlessly on Kaggle / Google Colab (16GB+ RAM / T4 GPU).

---

### 2. Label Consolidation Taxonomy (7 Canonical Attack Classes)

Raw CIC-IDS2017 multiclass labels are mapped to 7 clean, operational NIDS categories:

| Target Class | Raw CIC-IDS2017 Labels Merged | Threat Description |
| :--- | :--- | :--- |
| **`Benign`** | `BENIGN` | Legitimate background enterprise and web traffic |
| **`DoS`** | `DoS Hulk`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest` | Application/transport layer denial-of-service starvation |
| **`DDoS`** | `DDoS` | High-volume distributed packet floods |
| **`PortScan`** | `PortScan` | Horizontal/vertical reconnaissance and port discovery |
| **`Brute Force`** | `FTP-Patator`, `SSH-Patator` | Automated credential guessing on authentication services |
| **`Web Attack`** | `Web Attack – Brute Force`, `Web Attack – XSS`, `Web Attack – Sql Injection` (including UTF-8 / hyphen variations) | Common OWASP Top 10 exploits targeting web endpoints |
| **`Bot`** | `Bot` | Command & Control (C2) heartbeat and zombie communications (e.g. Ares) |

#### Excluded Classes (Academic Limitations Section):
* **`Heartbleed`** (~11 records in full capture)
* **`Infiltration`** (~36 records in full capture)
* *Rationale:* Extreme rarity causes catastrophic stratification failure and degenerate decision tree splits. Dropped and logged explicitly for thesis defense and methodology transparency.

---

### 3. Data Cleaning & Feature Pipeline
1. **Header Sanitization:** Strip leading and trailing whitespace from all column names (`df.columns = df.columns.str.strip()`).
2. **Infinite & NaN Handling:** Replace all `inf` and `-inf` with `np.nan`, followed by dropping null rows and exact duplicates.
3. **Data Leakage Prevention:** Split 80% train / 20% test using stratified sampling on the consolidated target class. `StandardScaler` is fitted strictly on the training partition and serialized to `ml/artifacts/scaler.joblib`.
4. **Target Encoding:** Mapped via `ml/artifacts/label_map.json` preserving integer index $\leftrightarrow$ class name parity.
