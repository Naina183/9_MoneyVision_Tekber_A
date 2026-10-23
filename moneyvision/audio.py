# audio.py
# bagian suara: TTS untuk membacakan hasil ke user, STT untuk input harga menggunakan suara
# pilihan tts:
#   pyttsx3 -> offline
#   gtts    -> pakai google
#   none    -> tidak ada suara, hanya untuk print ke terminal (buat testing)

import os
import queue
import tempfile
import threading

from .money import parse_nominal


class Speaker:
    def __init__(self, backend="auto", rate=160, verbose=True):
        self.verbose = verbose
        self.rate = rate  # kecepatan berbicara
        self.backend = self._pilih_backend(backend)
        self._q = queue.Queue()
        # tts dijalankan di thread terpisah agar kamera tidak freeze ketika berbicara
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @staticmethod
    def _pilih_backend(backend):
        if backend != "auto":
            return backend
        # jika auto: gunakan pyttsx3 bila sudah terinstall, jika belum hanya tampil teks
        try:
            import pyttsx3  # hanya untuk mengecek apakah sudah terinstall
            return "pyttsx3"
        except Exception:
            return "none"

    def say(self, text, interrupt=True):
        if self.verbose:
            print(f"[SUARA] {text}", flush=True)
        if self.backend == "none":
            return
        # pesan lama yang belum sempat dibacakan dibuang agar yang terdengar info terbaru
        if interrupt:
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put(text)

    def close(self):
        self._q.put(None)  # None sebagai tanda untuk menghentikan thread

    def _worker(self):
        if self.backend == "pyttsx3":
            self._jalan_pyttsx3()
        elif self.backend == "gtts":
            self._jalan_gtts()

    def _jalan_pyttsx3(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            # cari suara bahasa indonesia dari daftar suara yang ada di laptop
            suara_indo = None
            for v in engine.getProperty("voices"):
                info = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
                if "indones" in info or "andika" in info or "id-id" in info or "id_id" in info:
                    suara_indo = v.id
                    break
            if suara_indo:
                engine.setProperty("voice", suara_indo)
            else:
                print("[!] suara bahasa indonesia tidak ditemukan, coba jalankan dengan --tts gtts")
        except Exception as e:
            print(f"[!] pyttsx3 error ({e}), hasil hanya ditampilkan di terminal")
            self.backend = "none"
            return

        while True:
            text = self._q.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def _jalan_gtts(self):
        try:
            from gtts import gTTS
        except Exception as e:
            print(f"[!] gtts belum terinstall ({e}), hasil hanya ditampilkan di terminal")
            self.backend = "none"
            return

        play = _pemutar_mp3()
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                # gtts menyimpan suara ke file mp3 terlebih dahulu, lalu diputar
                path = os.path.join(tempfile.gettempdir(), "moneyvision_tts.mp3")
                gTTS(text=text, lang="id").save(path)
                play(path)
            except Exception as e:
                print(f"[!] gtts gagal: {e}")  # biasanya karena tidak ada internet


# untuk memutar mp3 hasil gtts, coba pygame dulu, jika tidak ada pakai playsound
def _pemutar_mp3():
    try:
        import pygame
        pygame.mixer.init()

        def play(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # tunggu sampai selesai berbicara
                pygame.time.wait(50)
            pygame.mixer.music.unload()
        return play
    except Exception:
        pass

    try:
        from playsound import playsound
        return lambda path: playsound(path)
    except Exception:
        return lambda path: print("[!] tidak ada pemutar audio, install pygame terlebih dahulu")


# input harga menggunakan suara (tombol V di aplikasi)
# membutuhkan SpeechRecognition + PyAudio + internet (google speech, bahasa indonesia)
def listen_nominal(timeout=5.0):
    import speech_recognition as sr
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        rec.adjust_for_ambient_noise(source, duration=0.5)  # menyesuaikan dengan suara bising di sekitar
        audio = rec.listen(source, timeout=timeout, phrase_time_limit=6)
    text = rec.recognize_google(audio, language="id-ID")
    # contoh: "dua puluh lima ribu" -> 25000
    return parse_nominal(text), text