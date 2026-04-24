import os
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from faster_whisper import WhisperModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

RECORDS_DIR = Path(os.getenv("RECORDS_DIR", "/records"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pt")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "2"))

CALL_BACKEND_UPLOAD_URL = os.getenv(
    "CALL_BACKEND_UPLOAD_URL",
    "http://192.168.0.55:8789/api/call/recordings",
)
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "")

KEEP_LOCAL_FILES = os.getenv("KEEP_LOCAL_FILES", "false").lower() == "true"
UPLOAD_TIMEOUT_SECONDS = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "120"))
TRANSCRIBE_MIXED_AUDIO = os.getenv("TRANSCRIBE_MIXED_AUDIO", "false").lower() == "true"

model = WhisperModel(
    WHISPER_MODEL,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=WHISPER_CPU_THREADS,
    num_workers=WHISPER_NUM_WORKERS,
)


class ReadyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        ready_path = Path(event.src_path)
        if ready_path.suffix.lower() != ".ready":
            return

        base_wav = ready_path.with_suffix(".wav")
        process_call(base_wav, ready_path)


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def wait_for_file(path: Path, timeout: int = 10) -> bool:
    last_size = -1

    for _ in range(timeout):
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                return True
            last_size = size

        time.sleep(1)

    return path.exists() and path.stat().st_size > 0


def normalize_wav(input_path: Path) -> Path:
    normalized_path = input_path.with_name(f"{input_path.stem}.normalized.wav")

    print(f"[INFO] Normalizando WAV: {input_path.name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(normalized_path),
    ]

    run_cmd(cmd)
    return normalized_path


def clean_text(text: str) -> str:
    if not text:
        return ""

    replacements = {
        " tá ": " está ",
        " pra ": " para ",
        " né ": " ",
    }

    padded = f" {text} "

    for old, new in replacements.items():
        padded = padded.replace(old, new)

    return " ".join(padded.split()).strip()


def transcribe_file(input_path: Path) -> str:
    print(f"[INFO] Transcrevendo: {input_path.name} com modelo {WHISPER_MODEL}")

    segments, info = model.transcribe(
        str(input_path),
        language=LANGUAGE,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 150,
        },
        condition_on_previous_text=False,
        initial_prompt=(
            "Transcrição de ligação telefônica em português do Brasil. "
            "Contexto: sistema de telefonia com Asterisk, NVOIP e ramais. "
            "Corrigir nomes para: Tracevia, NVOIP, Asterisk, módulo call, atendimento."
        ),
    )

    parts = []

    for segment in segments:
        txt = segment.text.strip()
        if txt:
            parts.append(txt)

    final_text = " ".join(parts).strip()
    final_text = clean_text(final_text)

    print(
        f"[INFO] Idioma detectado: {info.language} | "
        f"prob={info.language_probability:.3f}"
    )

    return final_text


def save_text_file(path: Path, text: str) -> None:
    path.write_text(text or "", encoding="utf-8")
    print(f"[OK] TXT salvo: {path.name}")


def delete_if_exists(path: Path | None) -> None:
    try:
        if path and path.exists():
            path.unlink()
            print(f"[INFO] Arquivo removido: {path.name}")
    except Exception as exc:
        print(f"[WARN] Não foi possível remover {path.name}: {exc}")


def transcribe_single_audio(audio_path: Path) -> tuple[str, Path | None]:
    if not audio_path.exists():
        print(f"[WARN] Arquivo não encontrado: {audio_path.name}")
        return "", None

    if audio_path.stat().st_size == 0:
        print(f"[WARN] Arquivo vazio: {audio_path.name}")
        return "", None

    normalized_path = normalize_wav(audio_path)

    if not normalized_path.exists() or normalized_path.stat().st_size == 0:
        print(f"[ERRO] WAV normalizado ficou vazio: {normalized_path.name}")
        return "", normalized_path

    text = transcribe_file(normalized_path)
    return text, normalized_path


def get_audio_duration_seconds(audio_path: Path) -> float:
    if not audio_path.exists():
        return 0.0

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        result = run_cmd(cmd)
        output = (result.stdout or "").strip()
        return float(output)
    except Exception as exc:
        print(f"[WARN] Não foi possível obter duração de {audio_path.name}: {exc}")
        return 0.0


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


def parse_call_metadata_from_filename(base_wav: Path) -> dict:
    stem = base_wav.stem
    parts = stem.split("-")

    if len(parts) < 5:
        print(f"[WARN] Nome de arquivo fora do padrão esperado: {base_wav.name}")
        return {
            "callUniqueId": stem,
            "caller": "",
            "destination": "",
            "extension": None,
        }

    call_unique_id = parts[2]
    caller = parts[3]
    destination = parts[4]
    extension = None

    return {
        "callUniqueId": call_unique_id,
        "caller": caller,
        "destination": destination,
        "extension": extension,
    }


def upload_to_call_backend(
    call_unique_id: str,
    caller: str,
    destination: str,
    extension: int | None,
    duration_seconds: float,
    duration_formatted: str,
    mp3_path: Path,
    txt_path: Path,
) -> bool:
    if not mp3_path.exists():
        print(f"[ERRO] MP3 não encontrado para upload: {mp3_path.name}")
        return False

    if not txt_path.exists():
        print(f"[ERRO] TXT não encontrado para upload: {txt_path.name}")
        return False

    payload = {
        "callUniqueId": call_unique_id,
        "caller": caller,
        "destination": destination,
        "durationSeconds": str(int(round(duration_seconds))),
        "durationFormatted": duration_formatted,
    }

    if extension is not None:
        payload["extension"] = str(extension)

    try:
        with open(mp3_path, "rb") as audio_file, open(txt_path, "rb") as transcript_file:
            response = requests.post(
                CALL_BACKEND_UPLOAD_URL,
                headers={
                    "X-Gateway-Secret": GATEWAY_SECRET,
                },
                data=payload,
                files={
                    "audio": (mp3_path.name, audio_file, "audio/mpeg"),
                    "transcript": (txt_path.name, transcript_file, "text/plain"),
                },
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )

        if 200 <= response.status_code < 300:
            print(f"[OK] Upload concluído para backend CALL: {call_unique_id}")
            return True

        print(
            f"[ERRO] Upload falhou. "
            f"status={response.status_code} body={response.text}"
        )
        return False

    except Exception as exc:
        print(f"[ERRO] Falha ao enviar para backend CALL: {exc}")
        return False


def process_call(base_wav: Path, ready_path: Path | None = None):
    time.sleep(1)

    if not base_wav.exists():
        print(f"[WARN] WAV base não encontrado: {base_wav.name}")
        if ready_path and ready_path.exists():
            ready_path.unlink()
        return

    if ".normalized" in base_wav.name:
        return

    final_txt = base_wav.with_suffix(".txt")
    if final_txt.exists():
        print(f"[SKIP] Transcrição já existe: {final_txt.name}")
        if ready_path and ready_path.exists():
            ready_path.unlink()
        return

    rx_wav = base_wav.with_name(f"{base_wav.stem}.rx.wav")
    tx_wav = base_wav.with_name(f"{base_wav.stem}.tx.wav")
    mp3_path = base_wav.with_suffix(".mp3")

    wait_for_file(base_wav, 10)

    if rx_wav.exists():
        wait_for_file(rx_wav, 10)

    if tx_wav.exists():
        wait_for_file(tx_wav, 10)

    if mp3_path.exists():
        wait_for_file(mp3_path, 10)

    mixed_normalized = None
    rx_normalized = None
    tx_normalized = None

    try:
        print(f"[INFO] Processando chamada base: {base_wav.name}")

        metadata = parse_call_metadata_from_filename(base_wav)
        call_unique_id = metadata["callUniqueId"]
        caller = metadata["caller"]
        destination = metadata["destination"]
        extension = metadata["extension"]

        mixed_text = ""

        if TRANSCRIBE_MIXED_AUDIO:
            mixed_text, mixed_normalized = transcribe_single_audio(base_wav)

        ligador_text = ""
        atendente_text = ""

        tasks = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            if rx_wav.exists():
                tasks["ligador"] = executor.submit(transcribe_single_audio, rx_wav)

            if tx_wav.exists():
                tasks["atendente"] = executor.submit(transcribe_single_audio, tx_wav)

            if "ligador" in tasks:
                ligador_text, rx_normalized = tasks["ligador"].result()

            if "atendente" in tasks:
                atendente_text, tx_normalized = tasks["atendente"].result()

        duration_seconds = get_audio_duration_seconds(mp3_path if mp3_path.exists() else base_wav)
        duration_formatted = format_duration(duration_seconds)

        final_content = []
        final_content.append("TRANSCRIÇÃO DA CHAMADA")
        final_content.append("")
        final_content.append(f"ID DA CHAMADA: {call_unique_id}")
        final_content.append(f"LIGADOR: {caller or '[não identificado]'}")
        final_content.append(f"DESTINO: {destination or '[não identificado]'}")
        final_content.append(f"DURAÇÃO: {duration_formatted}")
        final_content.append("")

        if TRANSCRIBE_MIXED_AUDIO:
            final_content.append("=== ÁUDIO MISTO ===")
            final_content.append(mixed_text if mixed_text else "[sem transcrição no áudio misto]")
            final_content.append("")

        final_content.append("=== LIGADOR ===")
        final_content.append(ligador_text if ligador_text else "[sem fala do ligador]")
        final_content.append("")

        final_content.append("=== ATENDENTE ===")
        final_content.append(atendente_text if atendente_text else "[sem fala do atendente]")
        final_content.append("")

        save_text_file(final_txt, "\n".join(final_content).strip() + "\n")

        uploaded = upload_to_call_backend(
            call_unique_id=call_unique_id,
            caller=caller,
            destination=destination,
            extension=extension,
            duration_seconds=duration_seconds,
            duration_formatted=duration_formatted,
            mp3_path=mp3_path,
            txt_path=final_txt,
        )

        if uploaded:
            print(f"[OK] Chamada {call_unique_id} enviada ao backend.")
        else:
            print(f"[WARN] Chamada {call_unique_id} NÃO foi enviada ao backend.")

    except subprocess.CalledProcessError as exc:
        print(f"[ERRO] Falha ao processar: {base_wav.name}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)

    except Exception as exc:
        print(f"[ERRO] Exceção inesperada ao processar {base_wav.name}: {exc}")

    finally:
        if ready_path and ready_path.exists():
            ready_path.unlink()
            print(f"[INFO] Arquivo ready removido: {ready_path.name}")

        delete_if_exists(mixed_normalized)
        delete_if_exists(rx_normalized)
        delete_if_exists(tx_normalized)

        delete_if_exists(base_wav)
        delete_if_exists(rx_wav)
        delete_if_exists(tx_wav)

        if KEEP_LOCAL_FILES:
            print("[INFO] KEEP_LOCAL_FILES=true, mantendo MP3 e TXT locais.")
        else:
            print("[INFO] Mantendo somente MP3 e TXT locais.")


if __name__ == "__main__":
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] Monitorando pasta: {RECORDS_DIR}")
    print(f"[INIT] Modelo Whisper: {WHISPER_MODEL}")
    print(f"[INIT] Idioma: {LANGUAGE}")
    print(f"[INIT] Device: {DEVICE}")
    print(f"[INIT] Compute type: {COMPUTE_TYPE}")
    print(f"[INIT] CPU threads: {WHISPER_CPU_THREADS}")
    print(f"[INIT] Num workers: {WHISPER_NUM_WORKERS}")
    print(f"[INIT] Upload backend URL: {CALL_BACKEND_UPLOAD_URL}")
    print(f"[INIT] KEEP_LOCAL_FILES: {KEEP_LOCAL_FILES}")
    print(f"[INIT] TRANSCRIBE_MIXED_AUDIO: {TRANSCRIBE_MIXED_AUDIO}")
    print("[INIT] Aguardando arquivos .ready...")

    event_handler = ReadyHandler()
    observer = Observer()
    observer.schedule(event_handler, str(RECORDS_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
