"""Pipeline completa: foto -> righi -> simboli -> MIDI -> audio.

    risultato = foto_a_audio('foto.jpg', percorso_modello='best.pt')
    if risultato.ok:
        frequenza, onda = risultato.audio          # pronto per Gradio
    print(risultato.messaggio)                      # sempre un messaggio comprensibile per l'utente

La funzione NON solleva errori: se qualcosa non va, 'ok' è False e 'messaggio' spiega cosa fare.
In ogni caso il risultato contiene quello che è stato calcolato fino a quel punto (righi, simboli, ...).
"""
import os
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

_CACHE_MODELLI = {}          # il modello si carica una volta sola, non a ogni foto


@dataclass
class Risultato:
    ok: bool = False
    messaggio: str = ''
    righi: list = field(default_factory=list)       # immagini dei righi ritagliati
    simboli: list = field(default_factory=list)     # una lista di simboli per ogni rigo
    midi: object = None                             # PrettyMIDI del brano
    audio: object = None                            # (frequenza, array int16)
    avvisi: list = field(default_factory=list)      # simboli saltati, righi senza note...
    tempi: dict = field(default_factory=dict)       # secondi impiegati da ogni fase

    def testo_simboli(self):
        """Una riga di testo per ogni rigo: serve per mostrare cosa ha capito il modello."""
        if not self.simboli:
            return "(nessun simbolo)"
        return "\n".join(f"Rigo {i}: " + " ".join(s) for i, s in enumerate(self.simboli, 1))


# ---------------------------------------------------------------- modello
def carica_modello_una_volta(percorso, device='cpu'):
    """Restituisce (modello, dizionario numero->simbolo). Dalla seconda chiamata in poi è istantaneo."""
    chiave = (os.path.abspath(str(percorso)), device)
    if chiave not in _CACHE_MODELLI:
        import model as _modello
        _CACHE_MODELLI[chiave] = _modello.carica_modello(percorso, device)
    return _CACHE_MODELLI[chiave]


# ---------------------------------------------------------------- foto
def _prepara_foto(foto, rgb=True):
    """Percorso, bytes, immagine PIL o array -> percorso oppure array BGR (quello che si aspetta OpenCV).

    rgb: per gli array a 3 colori. Gradio e PIL danno RGB (True); cv2.imread dà BGR (False).
    Solleva un errore se l'immagine non è utilizzabile.
    """
    import cv2
    if foto is None:
        raise ValueError("nessuna foto")
    if isinstance(foto, (str, os.PathLike)):
        if not os.path.exists(foto):
            raise ValueError(f"file non trovato: {foto}")
        return str(foto)
    if isinstance(foto, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(foto, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("i dati non sono un'immagine")
        return img
    if hasattr(foto, 'convert') and not isinstance(foto, np.ndarray):      # immagine PIL
        foto = np.array(foto.convert('RGB'))
        rgb = True
    if isinstance(foto, np.ndarray):
        img = foto
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]                                              # tolgo la trasparenza
        if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
            raise ValueError("formato dell'immagine non supportato")
        if img.dtype != np.uint8:
            img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        if img.ndim == 3 and rgb:
            img = img[..., ::-1]
        if min(img.shape[:2]) < 50:
            raise ValueError("immagine troppo piccola")
        return np.ascontiguousarray(img)
    raise ValueError("tipo di immagine non supportato")


# ---------------------------------------------------------------- pipeline
def foto_a_audio(foto, percorso_modello=None, modello=None, tempo=100, device='cpu', rgb=True,
                 applica_armatura=False, gracenote='salta'):
    """Foto di una pagina -> Risultato (righi, simboli, MIDI, audio, messaggio, tempi).

    percorso_modello: percorso di best.pt (il modello si carica una sola volta). Oppure passa
                      direttamente 'modello' = (modello, id2tok) già caricato.
    tempo: battiti al minuto del brano.
    """
    ris = Risultato()
    inizio = time.perf_counter()
    try:
        # 0) modello
        t0 = time.perf_counter()
        try:
            if modello is None:
                if percorso_modello is None:
                    raise ValueError("percorso del modello non indicato")
                modello = carica_modello_una_volta(percorso_modello, device)
            rete, id2tok = modello
        except Exception as e:
            ris.messaggio = f"Non riesco a caricare il modello ({e}). Il problema è nell'app, non nella tua foto."
            return ris
        ris.tempi['caricamento_modello'] = time.perf_counter() - t0

        # 1) la foto
        try:
            immagine = _prepara_foto(foto, rgb)
        except Exception as e:
            ris.messaggio = ("Non riesco a leggere la foto: controlla che sia un'immagine valida "
                             f"(JPG o PNG) e non troppo piccola. ({e})")
            return ris

        # 2) righi
        from preprocessing import foto_a_righi
        t0 = time.perf_counter()
        try:
            ris.righi = foto_a_righi(immagine)
        except Exception as e:
            ris.messaggio = f"Non riesco a elaborare la foto ({e}). Prova con un'altra foto."
            return ris
        ris.tempi['preprocessing'] = time.perf_counter() - t0
        if not ris.righi:
            ris.messaggio = ("Non ho trovato nessun pentagramma nella foto. Prova con più luce, "
                             "avvicinandoti allo spartito e tenendo il telefono parallelo al foglio. "
                             "Funziona con melodie scritte su un solo pentagramma.")
            return ris

        # 3) simboli (il modello)
        from model import riconosci_righi
        t0 = time.perf_counter()
        ris.simboli = riconosci_righi(ris.righi, rete, id2tok, device)
        ris.tempi['modello'] = time.perf_counter() - t0
        if not any(ris.simboli):
            ris.messaggio = (f"Ho trovato {len(ris.righi)} rigo/righi ma il modello non è riuscito a leggere "
                             "nessun simbolo. Prova con una foto più nitida o con uno spartito più semplice.")
            return ris

        # 4) un MIDI per rigo, poi unione
        from midi_convert import tokens_to_midi
        from audio import staves_to_midi, midi_to_audio, ErroreAudio
        t0 = time.perf_counter()
        midi_righi = [tokens_to_midi(s, tempo=tempo, applica_armatura=applica_armatura,
                                     gracenote=gracenote, avvisi=ris.avvisi) for s in ris.simboli]
        try:
            ris.midi = staves_to_midi(midi_righi, avvisi=ris.avvisi)
        except ErroreAudio:
            n_simboli = sum(len(s) for s in ris.simboli)
            ris.messaggio = (f"Ho letto {len(ris.righi)} rigo/righi ({n_simboli} simboli) ma non contengono "
                             "nessuna nota (solo chiavi o stanghette). Prova con una foto più nitida.")
            return ris
        ris.tempi['midi'] = time.perf_counter() - t0

        # 5) audio
        t0 = time.perf_counter()
        try:
            ris.audio = midi_to_audio(ris.midi)
        except ErroreAudio as e:
            ris.messaggio = str(e)
            return ris
        ris.tempi['audio'] = time.perf_counter() - t0

        n_note = sum(len(s.notes) for s in ris.midi.instruments)
        ris.ok = True
        ris.messaggio = f"Ho riconosciuto {len(ris.righi)} rigo/righi e {n_note} note."
        if ris.avvisi:
            ris.messaggio += f" Alcuni simboli sono stati saltati ({len(ris.avvisi)} avvisi)."
        return ris
    except Exception as e:                      # rete di sicurezza: l'app non deve mai bloccarsi
        ris.ok = False
        ris.messaggio = "Qualcosa è andato storto durante l'elaborazione. Riprova con un'altra foto."
        ris.avvisi.append("Errore imprevisto: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        return ris
    finally:
        ris.tempi['totale'] = time.perf_counter() - inizio


# ---------------------------------------------------------------- misura dei tempi
def misura_tempi(foto, percorso_modello, ripetizioni=3, **opzioni):
    """Esegue la pipeline più volte e restituisce i tempi medi per fase (il caricamento del modello è escluso)."""
    carica_modello_una_volta(percorso_modello, opzioni.get('device', 'cpu'))     # fuori dalla misura
    raccolti = {}
    for _ in range(ripetizioni):
        r = foto_a_audio(foto, percorso_modello, **opzioni)
        if not r.ok:
            raise RuntimeError("La pipeline non è riuscita: " + r.messaggio)
        for chiave, valore in r.tempi.items():
            if chiave != 'caricamento_modello':
                raccolti.setdefault(chiave, []).append(valore)
    return {chiave: sum(v) / len(v) for chiave, v in raccolti.items()}
