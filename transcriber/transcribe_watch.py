import os
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

RECORDS_DIR = Path(os.getenv("RECORDS_DIR", "/records"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pt")

class Mp3Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if file_path.suffix.lower() != ".mp3":
            return

        process_mp3(file_path)

def is_file_stable(file_path: Path, wait_seconds: int = 2, retries: int = 10) -> bool:
    last_size = -1

    for _ in range(retries):
        if not file_path.exists():
            return False

        current_size = file_path.stat().st_size
        if current_size == last_size and current_size > 0:
            return True

        last_size = current_size
        time.sleep(wait_seconds)

    return False

def process_mp3(file_path: Path):
    txt_path = file_path.with_suffix(".txt")

    if txt_path.exists():
        print(f"[SKIP] Transcrição já existe: {txt_path.name}")
        return

    print(f"[INFO] Novo MP3 detectado: {file_path.name}")

    if not is_file_stable(file_path):
        print(f"[WARN] Arquivo não estabilizou: {file_path.name}")
        return

    cmd = [
        "whisper",
        str(file_path),
        "--model", WHISPER_MODEL,
        "--language", LANGUAGE,
        "--output_format", "txt",
        "--output_dir", str(RECORDS_DIR),
    ]

    try:
        print(f"[INFO] Transcrevendo: {file_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[OK] Transcrição concluída: {file_path.name}")
        if result.stdout.strip():
            print(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        print(f"[ERRO] Falha ao transcrever: {file_path.name}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)

def transcribe_existing_files():
    for mp3_file in sorted(RECORDS_DIR.glob("*.mp3")):
        process_mp3(mp3_file)

if __name__ == "__main__":
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] Monitorando pasta: {RECORDS_DIR}")
    print("[INIT] Transcrevendo arquivos já existentes...")
    transcribe_existing_files()

    event_handler = Mp3Handler()
    observer = Observer()
    observer.schedule(event_handler, str(RECORDS_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
