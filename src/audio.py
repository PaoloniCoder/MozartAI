"""Audio e unione dei righi.

    pm_brano = staves_to_midi([pm_rigo1, pm_rigo2])      # più righi -> un unico brano
    frequenza, onda = midi_to_audio(pm_brano)            # pronto per Gradio o per IPython.display.Audio
"""
import numpy as np


class ErroreAudio(ValueError):
    """Errore con un messaggio pensato per essere mostrato all'utente (non un guasto del programma)."""


def _note_da_pm(pm):
    """Tutte le note di un oggetto PrettyMIDI, escluse le percussioni."""
    note = []
    for strumento in getattr(pm, 'instruments', []) or []:
        if getattr(strumento, 'is_drum', False):
            continue
        note.extend(strumento.notes)
    return note


# ---------------------------------------------------------------- MIDI -> suono
def midi_to_audio(pm, fs=22050, volume=0.8, armoniche=(1.0, 0.35, 0.15), attacco=0.01, rilascio=0.06, coda=0.3):
    """PrettyMIDI -> (frequenza di campionamento, array di interi a 16 bit).

    Come evitare la distorsione:
    - ogni nota ha un inviluppo (attacco e rilascio graduali): senza, ogni inizio/fine di nota fa un "click";
    - le armoniche sopra la metà della frequenza di campionamento vengono saltate (altrimenti fanno rumore);
    - alla fine il volume viene riportato a 'volume' (0-1) del massimo: niente saturazione.
    """
    if not 0 < volume <= 1:
        raise ErroreAudio("Il volume deve essere un numero tra 0 (escluso) e 1.")
    note = [n for n in _note_da_pm(pm) if n.end > n.start]
    if not note:
        raise ErroreAudio("Il MIDI non contiene note: non c'è niente da riprodurre.")

    n_rel = max(1, int(rilascio * fs))
    lunghezza = int(np.ceil((max(n.end for n in note) + coda) * fs)) + n_rel + 2
    out = np.zeros(lunghezza, dtype=np.float64)

    for n in note:
        freq = 440.0 * 2.0 ** ((n.pitch - 69) / 12.0)
        i0 = int(round(n.start * fs))
        n_nota = max(int(round((n.end - n.start) * fs)), 2)
        t = np.arange(n_nota + n_rel) / fs
        onda = np.zeros_like(t)
        for k, ampiezza in enumerate(armoniche):
            if freq * (k + 1) < 0.45 * fs:                       # niente armoniche sopra il limite di Nyquist
                onda += ampiezza * np.sin(2 * np.pi * freq * (k + 1) * t)
        # inviluppo: attacco graduale, leggero calo durante la nota, rilascio graduale
        env = np.ones(n_nota + n_rel)
        n_att = max(1, min(int(attacco * fs), n_nota // 2))
        env[:n_nota] = np.linspace(1.0, 0.7, n_nota)
        env[:n_att] *= np.linspace(0.0, 1.0, n_att, endpoint=False)
        env[n_nota:] = np.linspace(env[n_nota - 1], 0.0, n_rel)
        velocita = getattr(n, 'velocity', 90) / 127.0
        fine_i = min(i0 + n_nota + n_rel, len(out))
        out[i0:fine_i] += (onda * env * velocita)[:fine_i - i0]

    picco = np.max(np.abs(out))
    if picco <= 0 or not np.isfinite(picco):
        raise ErroreAudio("Non è stato possibile generare il suono.")
    out *= volume / picco                                         # margine di sicurezza: mai oltre 'volume'
    dissolvenza = min(int(0.005 * fs), len(out) // 2)             # 5 ms agli estremi: niente click
    if dissolvenza > 1:
        rampa = np.linspace(0.0, 1.0, dissolvenza)
        out[:dissolvenza] *= rampa
        out[-dissolvenza:] *= rampa[::-1]
    return fs, np.round(out * 32767).astype(np.int16)


# ---------------------------------------------------------------- più righi -> un brano
def staves_to_midi(righi, pausa=0.0, avvisi=None):
    """Lista di PrettyMIDI (uno per rigo) -> un unico PrettyMIDI: ogni rigo parte quando finisce il precedente.

    pausa: secondi di silenzio tra un rigo e il successivo.
    avvisi: lista opzionale in cui vengono scritti i righi saltati perché senza note.
    Se la lista è vuota (o nessun rigo ha note) solleva ErroreAudio con un messaggio chiaro.
    """
    import pretty_midi
    if avvisi is None:
        avvisi = []
    if righi is None or len(righi) == 0:
        raise ErroreAudio("Nessun rigo da unire: non è stato riconosciuto nessun pentagramma nella foto.")

    tutte, spostamento, tempo = [], 0.0, None
    for numero, pm in enumerate(righi, 1):
        note = _note_da_pm(pm) if pm is not None else []
        if not note:
            avvisi.append(f"Il rigo {numero} non ha note: saltato.")
            continue
        if tempo is None:
            try:
                tempo = float(pm.get_tempo_changes()[1][0])       # il tempo del primo rigo valido
            except Exception:
                tempo = 100.0
        for n in note:
            tutte.append(pretty_midi.Note(velocity=getattr(n, 'velocity', 90), pitch=int(n.pitch),
                                          start=n.start + spostamento, end=n.end + spostamento))
        spostamento += max(n.end for n in note) + pausa            # il prossimo rigo parte qui

    if not tutte:
        raise ErroreAudio("I righi riconosciuti non contengono nessuna nota: non c'è niente da unire.")
    brano = pretty_midi.PrettyMIDI(initial_tempo=tempo or 100.0)
    strumento = pretty_midi.Instrument(program=0)
    strumento.notes = sorted(tutte, key=lambda n: (n.start, n.pitch))
    brano.instruments.append(strumento)
    return brano
