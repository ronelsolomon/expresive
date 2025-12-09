import os
import requests
import time

AUDIO_FILE = "test_audio_en.wav"
API_KEY = os.environ.get("WHISPER_API_KEY")  # Get from environment
if not API_KEY:
    raise EnvironmentError("❌ Environment variable WHISPER_API_KEY not set.")

HEADERS = {
    "x-api-key": API_KEY
}

# Replace with your actual IPs/domains
AZURE_URL = "https://curify-whisperx-btfhgrecdbf4ezf2.westus2-01.azurewebsites.net"
GCP_URL = "http://34.150.188.176:8000"

def check_health(api_name, base_url):
    health_url = f"{base_url}/health"
    try:
        print(f"🔎 Checking /health for {api_name}...")
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            print(f"✅ {api_name} is healthy.")
            return True
        else:
            print(f"❌ {api_name} /health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to reach {api_name} /health: {e}")
        return False

def benchmark(api_name, base_url):
    if not check_health(api_name, base_url):
        return float("inf")

    transcribe_url = f"{base_url}/transcribe"
    print(f"\n▶️ Sending transcription request to {api_name}...")

    try:
        with open(AUDIO_FILE, "rb") as f:
            files = {
                "file": ("test_audio_en.wav", f, "audio/wav")
            }

            start_time = time.time()
            response = requests.post(transcribe_url, headers=HEADERS, files=files)
            elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            line_count = len(result.get("lines", []))
            print(f"✅ {api_name} succeeded in {elapsed:.2f}s — {line_count} lines transcribed.")
        else:
            print(f"❌ {api_name} failed: {response.status_code} - {response.text}")

        return elapsed

    except Exception as e:
        print(f"❌ Error connecting to {api_name}: {e}")
        return float("inf")


def main():
    print("📊 Benchmarking WhisperX APIs with test audio...\n")

    azure_time = benchmark("Azure", AZURE_URL)
    gcp_time = benchmark("GCP", GCP_URL)

    print("\n📈 Comparison:")
    print(f"Azure: {azure_time:.2f} seconds")
    print(f"GCP:   {gcp_time:.2f} seconds")

    if azure_time < gcp_time:
        print("🏆 Azure is faster.")
    elif gcp_time < azure_time:
        print("🏆 GCP is faster.")
    else:
        print("⚖️  Both are equally fast.")

if __name__ == "__main__":
    main()