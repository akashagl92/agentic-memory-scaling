import os, requests, json, time, sys

# Load env safely
env = {}
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                env[key] = val

# Use Fallback Key for maximum reliability
api_key = env.get('GOOGLE_API_KEY_FALLBACK')
if not api_key:
    api_key = env.get('GOOGLE_API_KEY')

MODEL = 'gemini-3-flash-preview'
OUT_FILE = 'test/benchmarks/live_calibration_G3.0-FLASH_diagnostic_CONTIGUOUS.json'

# The standard 20-iteration matrix
MATRIX = []
for mode in ['warm', 'cold']:
    for turns in [5000, 10000]:
        for _ in range(5):
            MATRIX.append({
                "turns": turns,
                "tokens": turns * 25,
                "mode": mode
            })

results = []
print(f"Starting Contiguous {MODEL} Sweep (N=20)...")

for i, config in enumerate(MATRIX):
    turns = config["turns"]
    tokens = config["tokens"]
    mode = config["mode"]
    
    # Payload construction
    haystack = "Developer Log Entry: " + "A" * (turns * 25)
    prompt_text = haystack + "\n\nTask: Say exactly 'ok'."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
    
    print(f"[{i+1}/20] {turns}t {mode}... ", end="")
    sys.stdout.flush()
    
    start = time.time()
    try:
        res = requests.post(url, json={'contents': [{'parts': [{'text': prompt_text}]}]}, headers={'Content-Type': 'application/json'}, timeout=180)
        latency = time.time() - start
        
        if res.status_code == 200:
            print(f"✅ {latency:.2f}s")
            results.append({
                "success": True,
                "latency": latency,
                "depth": 0.5,
                "status": 200,
                "turns": turns,
                "tokens": tokens,
                "mode": mode
            })
        else:
            print(f"❌ {res.status_code}")
            if res.status_code == 429:
                print("Quota limit reached. stopping.")
                break
    except Exception as e:
        print(f"⚠️ Exception: {e}")
    
    # Atomic save
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    time.sleep(5) # Steady heartbeat to stay under RPM/TPM limits

print(f"\nContiguous Sweep Complete! Data saved to {OUT_FILE}")
