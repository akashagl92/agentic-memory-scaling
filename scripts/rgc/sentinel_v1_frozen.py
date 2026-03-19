#!/usr/bin/env python3
import os
import json
import re
from datetime import datetime

# L0 Gated Sentinel Keywords (V1.0 - Technical Context)
# This file is frozen for research reproducibility.
SENTINEL_PATTERNS = [
    r"ADR \d+",
    r"Decision:",
    r"Pattern:",
    r"Rule:",
    r"Goal:",
    r"Status: Hardened",
    r"Status: Candidate"
]

def scan_file(filepath):
    signals = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            matches = []
            for pattern in SENTINEL_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    matches.append(pattern)
            
            if matches:
                title = content.split('\n')[0].strip('# ')
                signals.append({
                    "source": filepath,
                    "title": title,
                    "patterns_matched": matches,
                    "timestamp": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
    except Exception as e:
        print(f"Error scanning {filepath}: {e}")
    return signals

def main():
    print("[*] RGC Research Sentinel (V1.0 Frozen) starting...")
    signals = []
    # Simplified scan for reproducibility
    for root, _, files in os.walk(".pai"):
        for file in files:
            if file.endswith(".md"):
                signals.extend(scan_file(os.path.join(root, file)))

    print(f"[*] Identified {len(signals)} signals using V1.0 policy.")

if __name__ == "__main__":
    main()
