"""
Reconocimiento de gestos de mano con MediaPipe + Kinect Xbox 360 (SDK v1.8)

Requisitos:
    pip install mediapipe opencv-python numpy

Drivers necesarios:
    - KinectRuntime-v1.8
    - KinectSDK-v1.8
    - KinectDeveloperToolkit-v1.8
"""

import ctypes
import ctypes.wintypes as wt
import os
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Gesture Recognizer
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import GestureRecognizerResult

# Configuracion
BASE_DIR           = Path(__file__).parent
TASK_FILE          = str(BASE_DIR / "gesture_recognizer.task")
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
PITCH_BYTES        = FRAME_WIDTH * 4 # BGRA : 4 bytes por pixel
FRAME_SIZE         = PITCH_BYTES * FRAME_HEIGHT
FONT               = cv2.FONT_HERSHEY_SIMPLEX

COLOR_GREEN        = (0, 255, 0)
COLOR_YELLOW       = (0, 255, 255)
COLOR_WHITE        = (255, 255, 255)
COLOR_BLACK        = (0, 0, 0)

PENCIL_THICKNESS   = 4
PENCIL_COLORS = [
    (0,   0, 255, "Rojo"),    # BGR
    (0, 255,   0, "Verde"),
    (255, 0,   0, "Azul"),
]

CAMERA_INDEX        = None # None = auto-detect via DirectShow
                           # Cambia a un numero (ej. 1) para forzar el indice
KINECT_USB_LOCATION = "Port_#0003.Hub_#0002"  # ubicacion USB en Administrador de dispositivos

# Botones UI (interaccion mediante hover del dedo indice)
DWELL_TIME_S   = 1.5 # segundos de hover para activar
BTN_COOLDOWN_S = 1.5
_BTN_W         = 162
_BTN_H         = 48
_BTN_X         = 12
_COLOR_BTN_W   = 50
_COLOR_BTN_GAP = 6
_BTN_DEFS      = [
    {"id": "toggle",  "y": 12},
    {"id": "clear",   "y": 12 + _BTN_H + 10},
    {"id": "color_0", "y": 12 + 2 * (_BTN_H + 10), "x": _BTN_X,                                   "w": _COLOR_BTN_W},
    {"id": "color_1", "y": 12 + 2 * (_BTN_H + 10), "x": _BTN_X + _COLOR_BTN_W + _COLOR_BTN_GAP,   "w": _COLOR_BTN_W},
    {"id": "color_2", "y": 12 + 2 * (_BTN_H + 10), "x": _BTN_X + 2*(_COLOR_BTN_W+_COLOR_BTN_GAP), "w": _COLOR_BTN_W},
    {"id": "quit",    "y": 12 + 3 * (_BTN_H + 10)},
]

# Constantes NUI API (Kinect SDK v1.8)
NUI_INITIALIZE_FLAG_USES_COLOR  = 0x00000080
NUI_IMAGE_TYPE_COLOR            = 0
NUI_IMAGE_RESOLUTION_640x480    = 2
NUI_IMAGE_STREAM_FLAG_DEFAULT   = 0x00000000
POOL_SIZE                       = 2
INVALID_HANDLE_VALUE            = ctypes.c_void_p(-1).value

# Estado compartido entre hilos
latest_frame:  np.ndarray | None = None
latest_result: GestureRecognizerResult | None = None
frame_lock     = threading.Lock()
result_lock    = threading.Lock()
running        = True


# Callback de MediaPipe (modo LIVE_STREAM)
def on_gesture_result(
    result: GestureRecognizerResult,
    output_image: mp.Image,
    timestamp_ms: int,
) -> None:
    global latest_result
    with result_lock:
        latest_result = result


# Inicializacion del Gesture Recognizer
def build_gesture_recognizer() -> mp_vision.GestureRecognizer:
    base_options = mp_python.BaseOptions(model_asset_path=TASK_FILE)
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_gesture_result,
    )
    return mp_vision.GestureRecognizer.create_from_options(options)


# Backend 1: Kinect NUI API directa (Kinect10.dll)
class KinectNUI:
    """
    Accede al stream de color de la Kinect Xbox 360 a traves de la NUI API
    del SDK v1.8, llamando directamente a Kinect10.dll via ctypes.
    """

    def __init__(self):
        # Cargar la DLL del Kinect SDK
        dll_path = os.path.join(
            os.environ.get("WINDIR", r"C:\Windows"), "System32", "Kinect10.dll"
        )
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"Kinect10.dll no encontrado en {dll_path}. "
                "Verifica que KinectSDK-v1.8 este instalado."
            )
        self._dll = ctypes.windll.LoadLibrary(dll_path)
        self._setup_prototypes()

        # Inicializar el runtime NUI
        hr = self._dll.NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR)
        if hr < 0:
            raise RuntimeError(f"NuiInitialize fallo: HRESULT=0x{hr & 0xFFFFFFFF:08X}")

        # Evento Win32 para notificacion de frame listo
        self._event = ctypes.windll.kernel32.CreateEventW(None, True, False, None)
        if self._event == INVALID_HANDLE_VALUE:
            raise RuntimeError("No se pudo crear el evento Win32.")

        # Abrir el stream de color
        self._stream = ctypes.c_void_p()
        hr = self._dll.NuiImageStreamOpen(
            NUI_IMAGE_TYPE_COLOR,
            NUI_IMAGE_RESOLUTION_640x480,
            NUI_IMAGE_STREAM_FLAG_DEFAULT,
            POOL_SIZE,
            self._event,
            ctypes.byref(self._stream),
        )
        if hr < 0:
            raise RuntimeError(f"NuiImageStreamOpen fallo: HRESULT=0x{hr & 0xFFFFFFFF:08X}")

        print("[Kinect] NUI stream de color abierto (Kinect10.dll).")

        # Hilo de lectura
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _setup_prototypes(self):
        dll = self._dll
        dll.NuiInitialize.restype  = ctypes.HRESULT
        dll.NuiInitialize.argtypes = [wt.DWORD]
        dll.NuiShutdown.restype    = None
        dll.NuiShutdown.argtypes   = []
        dll.NuiImageStreamOpen.restype  = ctypes.HRESULT
        dll.NuiImageStreamOpen.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            wt.DWORD,
            wt.DWORD,
            wt.HANDLE,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.NuiImageStreamGetNextFrame.restype  = ctypes.HRESULT
        dll.NuiImageStreamGetNextFrame.argtypes = [
            ctypes.c_void_p, wt.DWORD, ctypes.c_void_p
        ]
        dll.NuiImageStreamReleaseFrame.restype  = ctypes.HRESULT
        dll.NuiImageStreamReleaseFrame.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p
        ]

    def _read_loop(self):
        """
        Lee frames del stream de la Kinect usando la vtable de INuiFrameTexture.

        NUI_IMAGE_FRAME layout (x64, SDK v1.8):
          offset 0  : liTimeStamp (LARGE_INTEGER, 8 bytes)
          offset 8  : dwFrameNumber (DWORD, 4 bytes)
          offset 12 : eImageType (int, 4 bytes)
          offset 16 : eResolution (int, 4 bytes)
          offset 20 : padding (4 bytes)
          offset 24 : pFrameTexture (INuiFrameTexture*, 8 bytes)
        """
        global latest_frame, running

        FRAME_STRUCT_SIZE = 64
        frame_buf  = (ctypes.c_byte * FRAME_STRUCT_SIZE)()
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT  = 0x00000102
        kernel32      = ctypes.windll.kernel32

        while running:
            wait_ret = kernel32.WaitForSingleObject(self._event, 500)
            if wait_ret == WAIT_TIMEOUT:
                continue
            if wait_ret != WAIT_OBJECT_0:
                break

            hr = self._dll.NuiImageStreamGetNextFrame(
                self._stream, 0, ctypes.cast(frame_buf, ctypes.c_void_p)
            )
            if hr < 0:
                kernel32.ResetEvent(self._event)
                continue

            # Puntero a INuiFrameTexture en offset 24
            texture_ptr = ctypes.c_void_p.from_buffer(frame_buf, 24).value

            if texture_ptr:
                # vtable de INuiFrameTexture:
                # 0:QueryInterface  1:AddRef  2:Release
                # 3:BufferLen       4:Pitch
                # 5:LockRect        6:GetLevelDesc   7:UnlockRect
                LOCKED_RECT_SIZE = 16
                locked_rect = (ctypes.c_byte * LOCKED_RECT_SIZE)()
                vtable = ctypes.cast(texture_ptr, ctypes.POINTER(ctypes.c_void_p))

                LockRect = ctypes.WINFUNCTYPE(
                    ctypes.HRESULT,
                    ctypes.c_void_p, ctypes.c_uint,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                )(vtable[5])

                hr2 = LockRect(
                    texture_ptr, 0,
                    ctypes.cast(locked_rect, ctypes.c_void_p),
                    None, 0,
                )
                if hr2 >= 0:
                    # LOCKED_RECT: Pitch(int,4) + padding(4) + pBits(ptr,8)
                    pitch    = ctypes.c_int.from_buffer(locked_rect, 0).value
                    bits_ptr = ctypes.c_void_p.from_buffer(locked_rect, 8).value

                    if bits_ptr and pitch > 0:
                        arr  = (ctypes.c_ubyte * FRAME_SIZE).from_address(bits_ptr)
                        bgra = np.frombuffer(arr, dtype=np.uint8).reshape(
                            (FRAME_HEIGHT, FRAME_WIDTH, 4)
                        ).copy()
                        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                        with frame_lock:
                            latest_frame = bgr

                    UnlockRect = ctypes.WINFUNCTYPE(
                        ctypes.HRESULT, ctypes.c_void_p, ctypes.c_uint,
                    )(vtable[7])
                    UnlockRect(texture_ptr, 0)

            self._dll.NuiImageStreamReleaseFrame(
                self._stream, ctypes.cast(frame_buf, ctypes.c_void_p)
            )
            kernel32.ResetEvent(self._event)

    def release(self):
        global running
        running = False
        time.sleep(0.3)
        ctypes.windll.kernel32.CloseHandle(self._event)
        self._dll.NuiShutdown()
        print("[Kinect] NUI liberado.")


# ── Helpers de enumeracion DirectShow para identificar la camara Kinect ─────────

def _com_release(ptr: ctypes.c_void_p) -> None:
    """Llama IUnknown::Release() sobre un puntero COM."""
    if ptr and ptr.value:
        vtbl = ctypes.cast(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])(ptr)


def _enumerate_dshow_cameras() -> list[tuple[int, str, str]]:
    """
    Enumera dispositivos de captura de video DirectShow via COM.
    Retorna [(indice, friendly_name, device_path), ...].
    El indice corresponde al que usa cv2.VideoCapture(idx, cv2.CAP_DSHOW).
    """
    results: list[tuple[int, str, str]] = []
    try:
        ole32    = ctypes.windll.ole32
        oleaut32 = ctypes.windll.oleaut32

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        def _make_guid(s: str) -> GUID:
            s = s.strip("{}")
            p = s.split("-")
            d4 = p[3] + p[4]
            return GUID(
                int(p[0], 16), int(p[1], 16), int(p[2], 16),
                (ctypes.c_ubyte * 8)(*[int(d4[i:i+2], 16) for i in range(0, 16, 2)]),
            )

        CLSID_SysDevEnum   = _make_guid("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
        IID_ICreateDevEnum = _make_guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
        CLSID_VideoInput   = _make_guid("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")
        IID_IPropertyBag   = _make_guid("{55272A00-42CB-11CE-8135-00AA004BB851}")

        ole32.CoInitialize(None)

        pCreateDevEnum = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(CLSID_SysDevEnum), None, 1,
            ctypes.byref(IID_ICreateDevEnum),
            ctypes.byref(pCreateDevEnum),
        )
        if hr < 0 or not pCreateDevEnum.value:
            return results

        vtbl0 = ctypes.cast(
            ctypes.cast(pCreateDevEnum, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )
        CreateClassEnum = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p,
            ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong,
        )(vtbl0[3])

        pEnum = ctypes.c_void_p()
        hr = CreateClassEnum(
            pCreateDevEnum, ctypes.byref(CLSID_VideoInput), ctypes.byref(pEnum), 0,
        )
        if hr < 0 or not pEnum.value:
            _com_release(pCreateDevEnum)
            return results

        vtbl1 = ctypes.cast(
            ctypes.cast(pEnum, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )
        EnumNext = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p,
            ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ulong),
        )(vtbl1[3])

        class VARIANT(ctypes.Structure):
            _fields_ = [
                ("vt",  ctypes.c_ushort),
                ("r1",  ctypes.c_ushort),
                ("r2",  ctypes.c_ushort),
                ("r3",  ctypes.c_ushort),
                ("val", ctypes.c_ulonglong),
            ]

        VT_BSTR = 8
        cam_idx = 0

        while True:
            pMoniker = ctypes.c_void_p()
            fetched  = ctypes.c_ulong(0)
            hr = EnumNext(pEnum, 1, ctypes.byref(pMoniker), ctypes.byref(fetched))
            if hr != 0 or fetched.value == 0:
                break
            if not pMoniker.value:
                cam_idx += 1
                continue

            vtblm = ctypes.cast(
                ctypes.cast(pMoniker, ctypes.POINTER(ctypes.c_void_p))[0],
                ctypes.POINTER(ctypes.c_void_p),
            )

            pBindCtx = ctypes.c_void_p()
            ole32.CreateBindCtx(0, ctypes.byref(pBindCtx))

            # IMoniker::BindToStorage esta en el indice 9 de la vtable
            BindToStorage = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p),
            )(vtblm[9])

            pPropBag = ctypes.c_void_p()
            hr2 = BindToStorage(
                pMoniker, pBindCtx, None,
                ctypes.byref(IID_IPropertyBag),
                ctypes.byref(pPropBag),
            )

            friendly_name = ""
            device_path   = ""

            if hr2 >= 0 and pPropBag.value:
                vtblp = ctypes.cast(
                    ctypes.cast(pPropBag, ctypes.POINTER(ctypes.c_void_p))[0],
                    ctypes.POINTER(ctypes.c_void_p),
                )
                PropBagRead = ctypes.WINFUNCTYPE(
                    ctypes.HRESULT, ctypes.c_void_p,
                    ctypes.c_wchar_p, ctypes.POINTER(VARIANT), ctypes.c_void_p,
                )(vtblp[3])

                for prop_name in ("FriendlyName", "DevicePath"):
                    v = VARIANT()
                    hr3 = PropBagRead(pPropBag, prop_name, ctypes.byref(v), None)
                    if hr3 >= 0 and v.vt == VT_BSTR and v.val:
                        text = ctypes.wstring_at(v.val)
                        if prop_name == "FriendlyName":
                            friendly_name = text
                        else:
                            device_path = text
                        oleaut32.VariantClear(ctypes.byref(v))

                _com_release(pPropBag)

            if pBindCtx.value:
                _com_release(pBindCtx)
            _com_release(pMoniker)

            results.append((cam_idx, friendly_name, device_path))
            cam_idx += 1

        _com_release(pEnum)
        _com_release(pCreateDevEnum)

    except Exception as exc:
        print(f"[AVISO] Error en enumeracion DirectShow: {exc}")
    return results


def _find_kinect_camera_index() -> int | None:
    """
    Detecta automaticamente el indice DirectShow de la camara Kinect.
    Estrategias (en orden): nombre NUI/Kinect, VID:PID 045E:02BB, location hint.
    Imprime todos los dispositivos encontrados para facilitar depuracion.
    """
    devices = _enumerate_dshow_cameras()
    if not devices:
        print("[AVISO] No se enumeraron dispositivos DirectShow.")
        return None

    print("[Kinect] Dispositivos de video DirectShow detectados:")
    for idx, name, path in devices:
        print(f"  [{idx}] {name!r}")

    # 1. Nombre caracteristico del sensor de color Kinect
    for idx, name, path in devices:
        if any(kw in name.lower() for kw in ("nui", "kinect", "xbox nui")):
            print(f"[Kinect] Identificada por nombre -> indice {idx}: {name!r}")
            return idx

    # 2. VID:PID del sensor de color Kinect v1 (VID_045E&PID_02BB)
    for idx, name, path in devices:
        if "vid_045e" in path.lower() and "pid_02bb" in path.lower():
            print(f"[Kinect] Identificada por VID/PID -> indice {idx}: {name!r}")
            return idx

    # 3. USB location hint en DevicePath
    hint = KINECT_USB_LOCATION.lower().replace(".", "_").replace("#", "_")
    for idx, name, path in devices:
        if hint in path.lower():
            print(f"[Kinect] Identificada por location -> indice {idx}: {name!r}")
            return idx

    print("[AVISO] No se pudo identificar la Kinect automaticamente en DirectShow.")
    return None


# Backend 2: OpenCV DirectShow (fallback si NUI falla)
class KinectOpenCV:
    """
    Fallback: abre la camara de color de la Kinect a traves de DirectShow
    (cv2.VideoCapture). Prueba los indices 0-9 o el indice forzado por CAMERA_INDEX.
    """

    def __init__(self, index: int | None = None):
        self._cap = None
        if index is None:
            index = _find_kinect_camera_index()
        scan_range = [index] if index is not None else range(10)
        # Backends a intentar en orden: DSHOW, MSMF, auto
        backends = [
            (cv2.CAP_DSHOW,  "DirectShow"),
            (cv2.CAP_MSMF,   "MSMF"),
            (cv2.CAP_ANY,    "auto"),
        ]
        found: list[int] = []

        print("[Camara] Escaneando dispositivos de video...")
        for idx in scan_range:
            for backend, bname in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                # Reintentar lectura hasta 5 veces (la Kinect tarda en entregar el 1er frame)
                ret = False
                for _ in range(5):
                    ret, _ = cap.read()
                    if ret:
                        break
                    time.sleep(0.1)
                if ret:
                    if idx not in found:
                        found.append(idx)
                    if self._cap is None:
                        self._cap = cap
                        print(f"[Camara] Indice {idx} ({bname}) -> OK (seleccionado)")
                    else:
                        print(f"[Camara] Indice {idx} ({bname}) -> OK")
                        cap.release()
                    break # backend OK para este indice, pasar al siguiente
                else:
                    print(f"[Camara] Indice {idx} ({bname}) -> abre pero no lee frames")
                    cap.release()

        if not found:
            print("[Camara] No se detecto ningun dispositivo en los indices probados.")
        else:
            print(f"[Camara] Dispositivos encontrados en indices: {found}")
            if len(found) > 1:
                print("[Camara] Si la Kinect no es el indice seleccionado, "
                      f"cambia CAMERA_INDEX a otro valor de {found}.")

        if self._cap is None:
            raise RuntimeError(
                "No se encontro camara. Verifica que la Kinect este conectada "
                "y los drivers instalados."
            )
        print("[Kinect] Camara lista (OpenCV).")
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        global latest_frame, running
        while running:
            ret, frame = self._cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
            time.sleep(0.008)

    def release(self):
        global running
        running = False
        time.sleep(0.2)
        self._cap.release()
        print("[Kinect] VideoCapture liberado.")


# Conexiones de la mano (MediaPipe Hands, 21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # pulgar
    (0,5),(5,6),(6,7),(7,8),       # indice
    (0,9),(9,10),(10,11),(11,12),  # medio
    (0,13),(13,14),(14,15),(15,16),# anular
    (0,17),(17,18),(18,19),(19,20),# menique
    (5,9),(9,13),(13,17),          # nudillos
]

# Colores por zona del dedo (BGR)
_FINGERTIP_IDS    = {4, 8, 12, 16, 20}
_PALM_IDS         = {0, 1, 5, 9, 13, 17}
_LM_COLOR_FINGERTIP = (0, 217, 255)
_LM_COLOR_FINGER    = (0, 217, 255)
_LM_COLOR_PALM      = (255, 138, 0)
_CONN_COLOR         = (200, 200, 200)


def draw_results(frame: np.ndarray, result: GestureRecognizerResult) -> np.ndarray:
    if result is None:
        return frame

    h, w, _ = frame.shape

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], _CONN_COLOR, 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(pts):
            dot_color = _LM_COLOR_FINGERTIP if i in _FINGERTIP_IDS else (
                        _LM_COLOR_PALM      if i in _PALM_IDS       else _LM_COLOR_FINGER)
            cv2.circle(frame, (px, py), 5, dot_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, COLOR_BLACK, 1, cv2.LINE_AA)

        # Etiqueta del gesto sobre la muneca (landmark 0)
        gesture_name, score = "?", 0.0
        if result.gestures and idx < len(result.gestures):
            top = result.gestures[idx][0]
            gesture_name, score = top.category_name, top.score

        wx, wy = pts[0]
        label  = f"{gesture_name} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        lx, ly = wx - 4, wy - 30
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 8, ly + 4), COLOR_BLACK, -1)
        cv2.putText(frame, label, (lx + 4, ly), FONT, 0.8, COLOR_YELLOW, 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"Manos: {len(result.hand_landmarks)}",
        (10, 30), FONT, 0.7, COLOR_GREEN, 2, cv2.LINE_AA,
    )
    return frame


# Deteccion de postura: solo el indice extendido (los demas doblados)
def _is_index_only_up(hand_landmarks) -> bool:
    """
    Devuelve True si el dedo indice esta extendido y el resto (medio, anular,
    menique) estan doblados. Usa coordenadas Y normalizadas de MediaPipe
    (0 = arriba de la imagen, 1 = abajo).
    """
    index_up   = hand_landmarks[8].y  < hand_landmarks[6].y   # tip < PIP
    middle_dn  = hand_landmarks[12].y > hand_landmarks[10].y  # tip > PIP
    ring_dn    = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_dn   = hand_landmarks[20].y > hand_landmarks[18].y
    return index_up and middle_dn and ring_dn and pinky_dn


# Helpers de la UI de botones
def _finger_on_button(cx: int, cy: int, btn: dict) -> bool:
    """Devuelve True si el punto (cx, cy) esta dentro del area del boton."""
    bx = btn.get("x", _BTN_X)
    bw = btn.get("w", _BTN_W)
    return bx <= cx <= bx + bw and btn["y"] <= cy <= btn["y"] + _BTN_H


def draw_ui_buttons(
    frame: np.ndarray,
    drawing_mode: bool,
    hover_prog: dict,
    color_idx: int = 0,
) -> np.ndarray:
    """Dibuja los botones UI sobre el frame con indicador de progreso de hover."""
    labels = {
        "toggle": "Lapiz: ON " if drawing_mode else "Lapiz: OFF",
        "clear":  "Limpiar",
        "quit":   "Salir",
    }
    for btn in _BTN_DEFS:
        bid  = btn["id"]
        bx   = btn.get("x", _BTN_X)
        bw   = btn.get("w", _BTN_W)
        x1, y1 = bx, btn["y"]
        x2, y2 = x1 + bw, y1 + _BTN_H
        prog   = hover_prog.get(bid, 0.0)

        # Botones de color: fondo solido del propio color
        if bid.startswith("color_"):
            cidx     = int(bid[-1])
            dot_color = PENCIL_COLORS[cidx][:3]
            selected  = (cidx == color_idx)

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), dot_color, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            border_clr   = COLOR_WHITE if selected else (80, 80, 80)
            border_thick = 3 if selected else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_clr, border_thick, cv2.LINE_AA)

            name = PENCIL_COLORS[cidx][3]
            (tw, th), _ = cv2.getTextSize(name, FONT, 0.42, 1)
            tx = x1 + (bw - tw) // 2
            ty = y1 + (_BTN_H + th) // 2
            cv2.putText(frame, name, (tx, ty), FONT, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

            if prog > 0:
                center = (x1 + bw // 2, y1 + _BTN_H // 2)
                cv2.circle(frame, center, 11, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.ellipse(frame, center, (11, 11), -90, 0, int(360 * prog),
                            (60, 230, 60), 3, cv2.LINE_AA)
            continue

        # Botones normales
        overlay = frame.copy()
        bg = (50, 80, 130) if prog > 0 else (35, 35, 35)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        border = (80, 190, 255) if prog > 0 else (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)

        label = labels[bid]
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        tx = x1 + (bw - tw) // 2
        ty = y1 + (_BTN_H + th) // 2
        cv2.putText(frame, label, (tx, ty), FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

        if prog > 0:
            center = (x2 - 18, y1 + _BTN_H // 2)
            cv2.circle(frame, center, 11, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.ellipse(
                frame, center, (11, 11), -90,
                0, int(360 * prog),
                (60, 230, 60), 3, cv2.LINE_AA,
            )

    return frame


# Bucle principal
def main():
    global running

    print("=" * 60)
    print("  Gestos con MediaPipe + Kinect Xbox 360 | SDK v1.8")
    print("  Presiona 'q' para salir")
    print("=" * 60)

    # Intentar NUI API; si falla, usar OpenCV
    kinect = None
    backend_label = ""
    try:
        kinect = KinectNUI()
        backend_label = "Kinect SDK v1.8 (NUI)"
    except Exception as e:
        print(f"[AVISO] NUI API no disponible: {e}")
        print("[INFO] Intentando fallback OpenCV/DirectShow...")
        try:
            kinect = KinectOpenCV(CAMERA_INDEX)
            backend_label = "OpenCV/DirectShow"
        except Exception as e2:
            print(f"[ERROR] No se pudo abrir la Kinect: {e2}")
            return

    print(f"[MediaPipe] Cargando modelo: {TASK_FILE}")
    recognizer = build_gesture_recognizer()
    print("[MediaPipe] Listo.")

    timestamp_ms   = 0
    frame_interval = int(1000 / 30)

    # Estado del lapiz
    drawing_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    prev_index_pts: dict = {} # hand_label -> (x, y)
    drawing_mode   = True
    color_idx      = 0  # indice en PENCIL_COLORS

    # Estado de los botones UI
    hover_start:    dict = {} # btn_id -> timestamp inicio hover
    hover_prog:     dict = {} # btn_id -> fraccion 0.0..1.0
    last_activated: dict = {} # btn_id -> timestamp de la ultima activacion

    cv2.namedWindow("Kinect - Gestos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kinect - Gestos", FRAME_WIDTH, FRAME_HEIGHT)

    try:
        while True:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if frame is None:
                cv2.waitKey(10)
                continue

            frame = cv2.flip(frame, 1) # modo espejo (flip horizontal)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            recognizer.recognize_async(mp_img, timestamp_ms)
            timestamp_ms += frame_interval

            with result_lock:
                snap = latest_result

            # Posicion del indice de cada mano en este frame
            finger_pts = [] # lista de (cx, cy) de cada mano detectada
            if snap is not None:
                current_labels: set = set()
                for idx, hand_landmarks in enumerate(snap.hand_landmarks):
                    hand_label = (
                        snap.handedness[idx][0].category_name
                        if snap.handedness and idx < len(snap.handedness)
                        else str(idx)
                    )
                    current_labels.add(hand_label)
                    lm = hand_landmarks[8] # punta del indice (landmark 8)
                    cx = int(lm.x * FRAME_WIDTH)
                    cy = int(lm.y * FRAME_HEIGHT)
                    finger_pts.append((cx, cy))

                    # Si el dedo esta sobre un boton, no dibuja (evita trazos)
                    on_btn = any(_finger_on_button(cx, cy, b) for b in _BTN_DEFS)
                    if on_btn:
                        prev_index_pts.pop(hand_label, None)
                    elif drawing_mode and _is_index_only_up(hand_landmarks):
                        cur_color = PENCIL_COLORS[color_idx][:3]
                        prev_pt = prev_index_pts.get(hand_label)
                        if prev_pt is not None:
                            cv2.line(drawing_canvas, prev_pt, (cx, cy),
                                     cur_color, PENCIL_THICKNESS, cv2.LINE_AA)
                        cv2.circle(drawing_canvas, (cx, cy),
                                   PENCIL_THICKNESS // 2, cur_color, -1, cv2.LINE_AA)
                        prev_index_pts[hand_label] = (cx, cy)
                    else:
                        prev_index_pts.pop(hand_label, None)

                for label in list(prev_index_pts.keys()):
                    if label not in current_labels:
                        del prev_index_pts[label]
            else:
                prev_index_pts.clear()

            # Logica de hover sobre botones
            now = time.time()
            action = None
            for btn in _BTN_DEFS:
                bid      = btn["id"]
                hovering = any(_finger_on_button(cx, cy, btn) for cx, cy in finger_pts)
                if hovering:
                    if bid not in hover_start:
                        hover_start[bid] = now
                    elapsed  = now - hover_start[bid]
                    hover_prog[bid] = min(elapsed / DWELL_TIME_S, 1.0)
                    last_act = last_activated.get(bid, 0.0)
                    if elapsed >= DWELL_TIME_S and (now - last_act) > BTN_COOLDOWN_S:
                        action = bid
                        last_activated[bid] = now
                        hover_start.pop(bid, None)
                        hover_prog[bid] = 0.0
                else:
                    hover_start.pop(bid, None)
                    hover_prog[bid] = 0.0

            # Ejecutar accion del boton activado
            quit_flag = False
            if action == "toggle":
                drawing_mode = not drawing_mode
                prev_index_pts.clear()
            elif action == "clear":
                drawing_canvas[:] = 0
                prev_index_pts.clear()
            elif action in ("color_0", "color_1", "color_2"):
                color_idx = int(action[-1])
            elif action == "quit":
                quit_flag = True

            output = draw_results(frame.copy(), snap)
            output = cv2.add(output, drawing_canvas)
            output = draw_ui_buttons(output, drawing_mode, hover_prog, color_idx)

            cv2.putText(
                output,
                f"Backend: {backend_label}",
                (10, FRAME_HEIGHT - 10), FONT, 0.45, COLOR_WHITE, 1, cv2.LINE_AA,
            )
            cv2.imshow("Kinect - Gestos", output)

            if cv2.waitKey(1) & 0xFF == 27 or quit_flag: # ESC de emergencia
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        kinect.release()
        recognizer.close()
        cv2.destroyAllWindows()
        print("[INFO] Cerrado correctamente.")


if __name__ == "__main__":
    main()
