"""
audio.py  -  Modulo de sintesis de voz (Microsoft Edge TTS - Neural)

Genera audio de alta calidad en espanol usando las voces neurales de
Microsoft Edge TTS. No requiere GPU ni modelos locales pesados.
Requiere conexion a internet.

Dependencias:
    pip install edge-tts          (ya incluido si usas una .venv reciente)

Uso como modulo:
    from audio import _generar_audio, SPEAKER_MUJER, SPEAKER_HOMBRE

Uso directo:
    python audio.py ruta/al/texto.txt [salida.mp3] [mujer|hombre]
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).parent

# ── Configuracion de voces ───────────────────────────────────────────────────
# Voces neurales de alta calidad de Microsoft Edge TTS
SPEAKER_MUJER  = "es-ES-ElviraNeural"   # mujer,  espanol de Espana
SPEAKER_HOMBRE = "es-ES-AlvaroNeural"   # hombre, espanol de Espana
SPEAKER        = SPEAKER_HOMBRE         # alias de compatibilidad

# Ajustes de prosodia (Edge TTS SSML rates/pitch/volume)
# Formato: "+10%", "-5%", "+0%"  |  para pitch: "+5Hz", "-3Hz", "+0Hz"
VOICE_RATE_MUJER   = "+0%"    # velocidad de habla  (mujer)
VOICE_RATE_HOMBRE  = "-5%"    # un poco mas lento para voz masculina
VOICE_PITCH_MUJER  = "+0Hz"   # tono base
VOICE_PITCH_HOMBRE = "-5Hz"   # tono ligeramente mas grave


# ── Helpers internos ─────────────────────────────────────────────────────────

def _leer_txt(txt_path: str) -> str:
    """Lee el .txt y devuelve el contenido como una sola cadena."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lineas = [l.strip() for l in f if l.strip()]
    return " ".join(lineas)


def _run_coro(coro):
    """Ejecuta una corrutina asyncio de forma segura desde un hilo sincrono."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Si asyncio.run falla (loop ya existente en el hilo), crear uno nuevo
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ── Generacion principal ─────────────────────────────────────────────────────

def _generar_audio(
    txt_path: str,
    mp3_path: str,
    progress_cb: Callable[[float, str], None] | None = None,
    speaker: str = SPEAKER_HOMBRE,
) -> bool:
    """
    Genera el .mp3 usando Microsoft Edge TTS (Neural).

    Edge TTS emite directamente MP3; no se necesita ffmpeg ni GPU.
    Si el texto es vacio o no existe el .txt, devuelve False.
    """
    try:
        import edge_tts
    except ImportError:
        print("[Audio] edge-tts no esta instalado. Ejecuta: pip install edge-tts")
        if progress_cb:
            progress_cb(-1.0, "Error: instala edge-tts")
        return False

    try:
        if not os.path.exists(txt_path):
            print(f"[Audio] No existe el archivo: {txt_path}")
            return False

        texto = _leer_txt(txt_path)
        if not texto:
            print("[Audio] El .txt esta vacio, no se genera audio.")
            return False

        # Seleccionar ajustes de prosodia segun la voz
        rate  = VOICE_RATE_MUJER   if speaker == SPEAKER_MUJER else VOICE_RATE_HOMBRE
        pitch = VOICE_PITCH_MUJER  if speaker == SPEAKER_MUJER else VOICE_PITCH_HOMBRE

        print(f"[Audio] Sintetizando con {speaker}: {texto[:80]}{'...' if len(texto)>80 else ''}")

        if progress_cb:
            progress_cb(0.1, f"Conectando con Edge TTS...")

        async def _synthesize():
            communicate = edge_tts.Communicate(
                text=texto,
                voice=speaker,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(mp3_path)

        if progress_cb:
            progress_cb(0.35, f"Generando voz ({speaker})...")

        _run_coro(_synthesize())

        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            print(f"[Audio] MP3 guardado -> {mp3_path}")
            if progress_cb:
                progress_cb(1.0, "Audio listo!")
            return True
        else:
            raise RuntimeError("El archivo MP3 generado esta vacio o no existe.")

    except Exception as exc:
        print(f"[Audio] Error: {exc}")
        if progress_cb:
            progress_cb(-1.0, f"Error: {exc}")
        return False


def txt_a_mp3_async(
    txt_path: str,
    mp3_path: str,
    on_progress: Callable[[float, str], None] | None = None,
    on_done: Callable[[bool, str], None] | None = None,
    speaker: str = SPEAKER_HOMBRE,
) -> threading.Thread:
    """
    Lanza la generacion de audio en un hilo de fondo.

    Parametros:
        txt_path    - ruta al .txt generado por EasyOCR
        mp3_path    - ruta de salida del .mp3
        on_progress - callback(fraccion: float, msg: str)  0.0-1.0
                      fraccion=-1 indica error
        on_done     - callback(exito: bool, ruta_mp3: str)
        speaker     - SPEAKER_MUJER o SPEAKER_HOMBRE

    Devuelve el Thread iniciado.
    """
    def _worker():
        ok = _generar_audio(txt_path, mp3_path, on_progress, speaker)
        if on_done:
            on_done(ok, mp3_path)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def reproducir_mp3(mp3_path: str) -> None:
    """Reproduce el archivo de audio usando pygame, playsound o startfile."""
    if not os.path.exists(mp3_path):
        print(f"[Audio] Archivo no encontrado: {mp3_path}")
        return
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        print(f"[Audio] Reproduciendo (pygame): {mp3_path}")
        return
    except Exception:
        pass
    try:
        from playsound import playsound
        threading.Thread(target=playsound, args=(mp3_path,), daemon=True).start()
        print(f"[Audio] Reproduciendo (playsound): {mp3_path}")
        return
    except Exception:
        pass
    try:
        os.startfile(mp3_path)
        print(f"[Audio] Abriendo con aplicacion predeterminada: {mp3_path}")
    except Exception as exc:
        print(f"[Audio] No se pudo reproducir: {exc}")


# ── Ejecucion directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python audio.py ruta/al/texto.txt [salida.mp3] [mujer|hombre]")
        sys.exit(1)

    _txt  = sys.argv[1]
    _mp3  = sys.argv[2] if len(sys.argv) > 2 else _txt.replace(".txt", ".mp3")
    _voz  = (SPEAKER_MUJER
             if len(sys.argv) > 3 and sys.argv[3].lower() == "mujer"
             else SPEAKER_HOMBRE)

    def _cb(frac, msg):
        bar = int(frac * 30) if frac >= 0 else 0
        print(f"\r[{'#'*bar}{' '*(30-bar)}] {msg}          ", end="", flush=True)

    print(f"Convirtiendo: {_txt}  ->  {_mp3}  (voz: {_voz})")
    ok = _generar_audio(_txt, _mp3, _cb, _voz)
    print()
    print("Listo!" if ok else "Fallo la generacion de audio.")
    sys.exit(0 if ok else 1)

    _txt  = sys.argv[1]
    _mp3  = sys.argv[2] if len(sys.argv) > 2 else _txt.replace(".txt", ".mp3")
    _voz  = (SPEAKER_MUJER
             if len(sys.argv) > 3 and sys.argv[3].lower() == "mujer"
             else SPEAKER_HOMBRE)

    def _cb(frac, msg):
        bar = int(frac * 30) if frac >= 0 else 0
        print(f"\r[{'#'*bar}{' '*(30-bar)}] {msg}          ", end="", flush=True)

    print(f"Convirtiendo: {_txt}  ->  {_mp3}  (voz: {_voz})")
    ok = _generar_audio(_txt, _mp3, _cb, _voz)
    print()
    print("Listo!" if ok else "Fallo la generacion de audio.")
    sys.exit(0 if ok else 1)