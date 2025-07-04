import subprocess
import time
import os

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Start hypothesis_agent
    hypothesis_path = os.path.join(base_path, "hypothesis_agent", "app.py")
    p1 = subprocess.Popen(["python", hypothesis_path])
    time.sleep(2)

    # Start media_info_agent
    media_path = os.path.join(base_path, "media_info_agent", "app.py")
    p2 = subprocess.Popen(["python", media_path])

    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()
