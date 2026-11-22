"""Strumenti musicali per l'app: suono sintetizzato e programma MIDI.

Il suono dell'app è generato con una sintesi additiva (somma di armoniche con un inviluppo), quindi
funziona ovunque senza installare altro. Il file MIDI scaricabile contiene invece il numero di strumento
General MIDI giusto: aprendolo in un lettore vero (MuseScore, GarageBand...) suona con lo strumento reale.
"""
import numpy as np

from audio import _note_da_pm, ErroreAudio

# armoniche: ampiezza relativa di 1ª, 2ª, 3ª... armonica | attacco e rilascio in secondi
# decadimento: quanto in fretta si spegne la nota (0 = tenuta) | sostegno: livello a cui si stabilizza (0-1)
# vibrato: (frequenza in Hz, profondità relativa)
PRESET = {
    'Pianoforte': dict(programma=0, armoniche=(1.0, 0.55, 0.3, 0.15, 0.08), attacco=0.004, decadimento=2.2,
                       sostegno=0.05, rilascio=0.15, vibrato=(0.0, 0.0)),
    'Chitarra': dict(programma=24, armoniche=(1.0, 0.6, 0.4, 0.25, 0.12, 0.06), attacco=0.003, decadimento=3.5,
                     sostegno=0.0, rilascio=0.10, vibrato=(0.0, 0.0)),
    'Violino': dict(programma=40, armoniche=(1.0, 0.75, 0.55, 0.4, 0.3, 0.2, 0.12, 0.08), attacco=0.09,
                    decadimento=0.0, sostegno=1.0, rilascio=0.12, vibrato=(5.5, 0.006)),
    'Flauto': dict(programma=73, armoniche=(1.0, 0.12, 0.04), attacco=0.06, decadimento=0.0, sostegno=1.0,
                   rilascio=0.10, vibrato=(5.0, 0.003)),
    'Clarinetto': dict(programma=71, armoniche=(1.0, 0.0, 0.5, 0.0, 0.25, 0.0, 0.12), attacco=0.04,
                       decadimento=0.0, sostegno=1.0, rilascio=0.08, vibrato=(0.0, 0.0)),
    'Organo': dict(programma=19, armoniche=(1.0, 0.8, 0.6, 0.45, 0.3), attacco=0.015, decadimento=0.0,
                   sostegno=1.0, rilascio=0.06, vibrato=(0.0, 0.0)),
    'Xilofono': dict(programma=13, armoniche=(1.0, 0.0, 0.35, 0.0, 0.1), attacco=0.001, decadimento=7.0,
                     sostegno=0.0, rilascio=0.08, vibrato=(0.0, 0.0)),
    'Sintetizzatore': dict(programma=80, armoniche=tuple((1.0 / n if n % 2 else 0.0) for n in range(1, 10)),
                           attacco=0.01, decadimento=0.5, sostegno=0.6, rilascio=0.08, vibrato=(0.0, 0.0)),
}
NOMI_STRUMENTI = list(PRESET)
STRUMENTO_PREDEFINITO = 'Pianoforte'


def _preset(strumento):
    if strumento not in PRESET:
        raise ErroreAudio(f"Strumento sconosciuto: {strumento}. Scegli tra: {', '.join(NOMI_STRUMENTI)}.")
    return PRESET[strumento]


def sintetizza(pm, strumento=STRUMENTO_PREDEFINITO, fs=22050, volume=0.8, trasposizione=0, coda=0.4):
    """PrettyMIDI -> (frequenza, array int16) con il timbro dello strumento scelto.

    trasposizione: semitoni da aggiungere (positivo = più acuto).
    Niente distorsione: inviluppo su ogni nota, armoniche sopra il limite di campionamento saltate,
    volume finale riportato a 'volume' (0-1) del massimo.
    """
    p = _preset(strumento)
    if not 0 < volume <= 1:
        raise ErroreAudio("Il volume deve essere un numero tra 0 (escluso) e 1.")
    note = [n for n in _note_da_pm(pm) if n.end > n.start]
    if not note:
        raise ErroreAudio("Il MIDI non contiene note: non c'è niente da riprodurre.")

    n_rel = max(1, int(p['rilascio'] * fs))
    lunghezza = int(np.ceil((max(n.end for n in note) + coda) * fs)) + n_rel + 2
    out = np.zeros(lunghezza, dtype=np.float64)
    freq_vib, prof_vib = p['vibrato']

    for n in note:
        altezza = int(min(108, max(21, n.pitch + int(trasposizione))))     # intervallo del pianoforte: oltre non si sente
        freq = 440.0 * 2.0 ** ((altezza - 69) / 12.0)
        i0 = int(round(n.start * fs))
        n_nota = max(int(round((n.end - n.start) * fs)), 2)
        t = np.arange(n_nota + n_rel) / fs
        fase = 2 * np.pi * freq * t
        if freq_vib > 0 and prof_vib > 0:                       # vibrato: piccola oscillazione dell'intonazione
            fase = fase + (prof_vib * freq / freq_vib) * np.sin(2 * np.pi * freq_vib * t) * \
                np.clip(t / 0.25, 0.0, 1.0)                     # compare gradualmente
        onda = np.zeros_like(t)
        for k, ampiezza in enumerate(p['armoniche'], 1):
            if ampiezza and freq * k < 0.45 * fs:                # niente armoniche oltre il limite di Nyquist
                onda += ampiezza * np.sin(k * fase)
        # inviluppo: attacco, calo verso il livello di sostegno, rilascio dopo la fine della nota
        env = p['sostegno'] + (1.0 - p['sostegno']) * np.exp(-p['decadimento'] * t)
        n_att = max(1, min(int(p['attacco'] * fs), n_nota // 2))
        env[:n_att] *= np.linspace(0.0, 1.0, n_att, endpoint=False)
        env[n_nota:] *= np.linspace(1.0, 0.0, n_rel)
        velocita = getattr(n, 'velocity', 90) / 127.0
        fine_i = min(i0 + n_nota + n_rel, len(out))
        out[i0:fine_i] += (onda * env * velocita)[:fine_i - i0]

    picco = np.max(np.abs(out))
    if picco <= 0 or not np.isfinite(picco):
        raise ErroreAudio("Non è stato possibile generare il suono.")
    out *= volume / picco
    dissolvenza = min(int(0.005 * fs), len(out) // 2)            # 5 ms agli estremi: niente click
    if dissolvenza > 1:
        rampa = np.linspace(0.0, 1.0, dissolvenza)
        out[:dissolvenza] *= rampa
        out[-dissolvenza:] *= rampa[::-1]
    return fs, np.round(out * 32767).astype(np.int16)


def midi_con_strumento(pm, strumento=STRUMENTO_PREDEFINITO, trasposizione=0):
    """Copia del brano con il programma MIDI dello strumento e la trasposizione applicata (per il download)."""
    import pretty_midi
    p = _preset(strumento)
    try:
        tempo = float(pm.get_tempo_changes()[1][0])
    except Exception:
        tempo = 100.0
    nuovo = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    strumento_midi = pretty_midi.Instrument(program=p['programma'])
    for n in _note_da_pm(pm):
        altezza = int(min(127, max(0, n.pitch + int(trasposizione))))
        strumento_midi.notes.append(pretty_midi.Note(velocity=getattr(n, 'velocity', 90), pitch=altezza,
                                                     start=n.start, end=n.end))
    nuovo.instruments.append(strumento_midi)
    return nuovo
