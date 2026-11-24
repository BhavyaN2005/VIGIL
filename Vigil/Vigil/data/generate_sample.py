"""
Generate realistic, authentic sample flows modeled on the CIC-IDS2017 dataset.
Includes:
- Authentic 78 directional flow attributes
- Raw labels with messy headers (leading spaces)
- 7 canonical classes + rare classes (Heartbleed, Infiltration)
- Intentional edge cases (inf, NaN, duplicate rows) for preprocess verification
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

FEATURE_NAMES = [
    " Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
    "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Max",
    " Fwd Packet Length Min", " Fwd Packet Length Mean", " Fwd Packet Length Std",
    "Bwd Packet Length Max", " Bwd Packet Length Min", " Bwd Packet Length Mean",
    " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean",
    " Flow IAT Std", " Flow IAT Max", " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean",
    " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean",
    " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags",
    " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length",
    "Fwd Packets/s", " Bwd Packets/s", " Min Packet Length", " Max Packet Length",
    " Packet Length Mean", " Packet Length Std", " Packet Length Variance",
    "FIN Flag Count", " SYN Flag Count", " RST Flag Count", " PSH Flag Count",
    " ACK Flag Count", " URG Flag Count", " CWE Flag Count", " ECE Flag Count",
    " Down/Up Ratio", " Average Packet Size", " Avg Fwd Segment Size",
    " Avg Bwd Segment Size", " Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    " Subflow Fwd Bytes", " Subflow Bwd Packets", " Subflow Bwd Bytes",
    "Init_Win_bytes_forward", " Init_Win_bytes_backward", " act_data_pkt_fwd",
    " min_seg_size_forward", "Active Mean", " Active Std", " Active Max",
    " Active Min", "Idle Mean", " Idle Std", " Idle Max", " Idle Min"
]

def generate_sample_dataset(n_samples=5200, output_path=None):
    classes_distribution = {
        "BENIGN": 3000,
        "DoS Hulk": 600,
        "DoS GoldenEye": 150,
        "DoS slowloris": 150,
        "DDoS": 450,
        "PortScan": 400,
        "FTP-Patator": 120,
        "SSH-Patator": 80,
        "Web Attack – Brute Force": 70,
        "Web Attack - XSS": 50,
        "Web Attack – Sql Injection": 30,
        "Bot": 60,
        "Heartbleed": 12,      # Rare class for drop test
        "Infiltration": 28,    # Rare class for drop test
    }
    
    rows = []
    labels = []
    
    for label, count in classes_distribution.items():
        for _ in range(count):
            row = np.random.exponential(scale=10.0, size=len(FEATURE_NAMES))
            
            # Domain-specific modifications based on attack type
            if "BENIGN" in label:
                row[0] = np.random.choice([80, 443, 53, 8080, 22]) # Dest port
                row[1] = np.random.uniform(500, 50000)             # Duration
                row[44] = np.random.choice([0, 1], p=[0.7, 0.3])   # SYN
                row[47] = 1                                        # ACK
            elif "DoS" in label:
                row[0] = 80
                row[1] = np.random.uniform(100000, 5000000)
                row[2] = np.random.randint(50, 1000)              # High Fwd packets
                row[3] = np.random.randint(0, 5)                  # Low Bwd packets
                row[14] = np.random.uniform(1000, 200000)         # Bytes/s
            elif "DDoS" in label:
                row[0] = 80
                row[1] = np.random.uniform(5000, 500000)
                row[15] = np.random.uniform(50000, 1000000)       # Packets/s
                row[36] = np.random.uniform(10000, 200000)
            elif "PortScan" in label:
                row[0] = np.random.randint(1, 65535)              # Varied port
                row[1] = np.random.uniform(50, 500)               # Tiny duration
                row[44] = 1                                       # SYN high
                row[47] = 0                                       # ACK 0
                row[38] = 0
                row[39] = 64
            elif "Patator" in label:
                row[0] = 22 if "SSH" in label else 21
                row[1] = np.random.uniform(1000, 20000)
                row[46] = np.random.randint(5, 50)                # PSH flags
            elif "Web Attack" in label:
                row[0] = np.random.choice([80, 443, 8080])
                row[4] = np.random.uniform(5000, 80000)           # Fwd length high
                row[46] = 1
            elif "Bot" in label:
                row[0] = np.random.choice([8080, 6667, 4444])
                row[74] = np.random.uniform(50000, 200000)        # Idle mean regular
            elif label == "Heartbleed":
                row[0] = 443
                row[4] = 65535
            elif label == "Infiltration":
                row[0] = 4444
                row[1] = np.random.uniform(50000, 100000)
                
            rows.append(row)
            labels.append(label)
            
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df[" Label"] = labels  # Messy header with leading space
    
    # Inject dirty test rows for preprocessing validation
    # 1. Row with inf
    df.loc[10, "Flow Bytes/s"] = np.inf
    # 2. Row with -inf
    df.loc[20, " Flow Packets/s"] = -np.inf
    # 3. Row with NaN
    df.loc[30, " Total Fwd Packets"] = np.nan
    # 4. Duplicate rows
    duplicate_rows = df.iloc[50:55].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Generated sample dataset: {df.shape} at {out}")
        
    return df

if __name__ == "__main__":
    generate_sample_dataset(output_path=Path(__file__).parent.parent / "data" / "sample" / "sample_flows.csv")
