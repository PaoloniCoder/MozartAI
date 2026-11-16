"""Conversione di una sequenza di simboli "semantic" (PrIMuS) in MIDI.

Uso:
    pm = tokens_to_midi(simboli, tempo=100)      # simboli: lista di stringhe, per es. ['clef-G2', 'note-D5_quarter', ...]
    pm.write('melodia.mid')

La funzione NON va mai in errore a causa dei simboli: quelli sconosciuti o malformati vengono saltati
e segnalati (l'output del modello su foto vere può essere sbagliato).
"""
import re
import warnings

# ---------------------------------------------------------------- tabelle
SEMITONI = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}      # semitoni sopra il do
ALTERAZIONI = {'': 0, '#': 1, 'b': -1, '##': 2, 'bb': -2, 'N': 0, 'n': 0}   # N = bequadro

# durata in battiti (il quarto vale 1). L'ordine conta: i nomi più lunghi prima.
DURATE = {
    'quadruple_whole': 16.0, 'double_whole': 8.0, 'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
    'eighth': 0.5, 'sixteenth': 0.25, 'thirty_second': 0.125, 'sixty_fourth': 0.0625,
    'hundred_twenty_eighth': 0.03125,
}
_NOMI_DURATA = sorted(DURATE, key=len, reverse=True)

# armature: numero di diesis (+) o bemolle (-)
_MAGGIORI = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
             'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7}
_MINORI = {'A': 0, 'E': 1, 'B': 2, 'F#': 3, 'C#': 4, 'G#': 5, 'D#': 6, 'A#': 7,
           'D': -1, 'G': -2, 'C': -3, 'F': -4, 'Bb': -5, 'Eb': -6, 'Ab': -7}
_ORDINE_DIESIS = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
_ORDINE_BEMOLLI = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

_RE_NOTA = re.compile(r'^([A-G])(##|bb|#|b|N|n)?(-?\d+)_(.+)$')
_RE_TEMPO = re.compile(r'^(\d+)/(\d+)$')
_RE_ARMATURA = re.compile(r'^([A-G])(#|b)?([Mm])$')


# ---------------------------------------------------------------- funzioni di base
def nome_nota_a_midi(lettera, alterazione, ottava):
    """('D', '', 5) -> 74.   Formula: 12 * (ottava + 1) + semitoni del nome + alterazione."""
    return 12 * (ottava + 1) + SEMITONI[lettera] + ALTERAZIONI[alterazione or '']


def durata_in_battiti(durata, punti=0):
    """Durata base (in battiti) con i punti di valore: ogni punto aggiunge metà del valore precedente."""
    return durata * (2.0 - 0.5 ** punti)


def _leggi_durata(testo):
    """'quarter' -> (1.0, 0) | 'eighth..' -> (0.5, 2) | 'half._fermata' -> (2.0, 1). None se non la riconosce."""
    for nome in _NOMI_DURATA:
        if testo.startswith(nome):
            dopo = testo[len(nome):]
            punti = len(dopo) - len(dopo.lstrip('.'))
            return DURATE[nome], punti          # quello che segue (es. "_fermata") non cambia la durata
    return None


def armatura_in_alterazioni(nome):
    """'DM' -> {'F': 1, 'C': 1}; 'AbM' -> {'B': -1, 'E': -1, 'A': -1, 'D': -1}. None se non valida."""
    m = _RE_ARMATURA.match(nome or '')
    if not m:
        return None
    tabella = _MINORI if m.group(3) == 'm' else _MAGGIORI
    n = tabella.get(m.group(1) + (m.group(2) or ''))
    if n is None:
        return None
    if n >= 0:
        return {lettera: 1 for lettera in _ORDINE_DIESIS[:n]}
    return {lettera: -1 for lettera in _ORDINE_BEMOLLI[:-n]}


# ---------------------------------------------------------------- analisi di un simbolo
def analizza_token(token):
    """Riconosce un simbolo. Restituisce SEMPRE un dizionario con 'tipo':
    note | gracenote | rest | multirest | clef | keySignature | timeSignature | barline | tie
    oppure 'malformato' (famiglia nota ma contenuto illeggibile) o 'sconosciuto'.
    """
    try:
        if not isinstance(token, str) or not token.strip():
            return {'tipo': 'sconosciuto', 'token': token}
        token = token.strip()
        famiglia, _, resto = token.partition('-')

        if famiglia in ('note', 'gracenote'):
            m = _RE_NOTA.match(resto)
            d = _leggi_durata(m.group(4)) if m else None
            if not m or d is None:
                return {'tipo': 'malformato', 'token': token}
            return {'tipo': famiglia, 'lettera': m.group(1), 'alterazione': m.group(2),
                    'ottava': int(m.group(3)), 'durata': d[0], 'punti': d[1]}

        if famiglia == 'rest':
            d = _leggi_durata(resto)
            if d is None:
                return {'tipo': 'malformato', 'token': token}
            return {'tipo': 'rest', 'durata': d[0], 'punti': d[1]}

        if famiglia == 'multirest':
            return {'tipo': 'multirest', 'misure': int(resto)} if resto.isdigit() else {'tipo': 'malformato', 'token': token}

        if famiglia == 'timeSignature':
            if resto == 'C':
                return {'tipo': 'timeSignature', 'num': 4, 'den': 4}
            if resto == 'C/':
                return {'tipo': 'timeSignature', 'num': 2, 'den': 2}
            m = _RE_TEMPO.match(resto)
            if m and int(m.group(2)) > 0:
                return {'tipo': 'timeSignature', 'num': int(m.group(1)), 'den': int(m.group(2))}
            return {'tipo': 'malformato', 'token': token}

        if famiglia == 'keySignature':
            alt = armatura_in_alterazioni(resto)
            return {'tipo': 'keySignature', 'alterazioni': alt} if alt is not None else {'tipo': 'malformato', 'token': token}

        if famiglia == 'clef':
            return {'tipo': 'clef'} if re.match(r'^[GFC]\d$', resto) else {'tipo': 'malformato', 'token': token}

        if token == 'barline':
            return {'tipo': 'barline'}
        if token == 'tie':
            return {'tipo': 'tie'}
    except Exception:
        return {'tipo': 'malformato', 'token': token}
    return {'tipo': 'sconosciuto', 'token': token}


# ---------------------------------------------------------------- dalla sequenza alla lista di note
def tokens_to_notes(tokens, applica_armatura=False, gracenote='salta', avvisi=None):
    """Lista di simboli -> lista di note [altezza MIDI, inizio in battiti, durata in battiti].

    applica_armatura: se True, le note senza alterazione esplicita prendono quelle dell'armatura.
                      Di default NO: nel formato semantic il nome della nota sembra già includere
                      l'alterazione (es. note-Bb3 con armatura AbM). Da verificare con i .mid (Issue 2).
    gracenote: 'salta' (ignora le appoggiature) oppure 'breve' (le suona molto corte, senza far avanzare il tempo).
    avvisi: lista a cui aggiungere i messaggi sui simboli saltati.
    """
    if avvisi is None:
        avvisi = []
    if isinstance(tokens, str):
        tokens = tokens.split()
    note = []
    tempo_corr = 0.0              # in battiti
    battiti_misura = 4.0
    armatura = {}
    tie_attivo = False
    ultima = None                 # ultima nota suonata (serve alle legature)
    saltati_grace = 0

    for token in (tokens or []):
        info = analizza_token(token)
        tipo = info['tipo']
        try:
            if tipo == 'note':
                alt = info['alterazione']
                if alt is None and applica_armatura:
                    alt = {1: '#', -1: 'b'}.get(armatura.get(info['lettera'], 0), '')
                midi = nome_nota_a_midi(info['lettera'], alt, info['ottava'])
                dur = durata_in_battiti(info['durata'], info['punti'])
                if not 0 <= midi <= 127:
                    avvisi.append(f"Nota fuori dall'intervallo MIDI saltata: {token}")
                    continue
                if tie_attivo and ultima is not None and ultima[0] == midi \
                        and abs(ultima[1] + ultima[2] - tempo_corr) < 1e-6:
                    ultima[2] += dur                     # legatura: allungo la nota precedente
                else:
                    if tie_attivo:
                        avvisi.append(f"Legatura tra note diverse ignorata (prima di {token})")
                    ultima = [midi, tempo_corr, dur]
                    note.append(ultima)
                tempo_corr += dur
                tie_attivo = False
            elif tipo == 'gracenote':
                if gracenote == 'breve':
                    midi = nome_nota_a_midi(info['lettera'], info['alterazione'], info['ottava'])
                    if 0 <= midi <= 127:
                        note.append([midi, tempo_corr, 0.125])
                else:
                    saltati_grace += 1
            elif tipo == 'rest':
                tempo_corr += durata_in_battiti(info['durata'], info['punti'])
                tie_attivo = False
            elif tipo == 'multirest':
                tempo_corr += min(info['misure'], 1000) * battiti_misura
                tie_attivo = False
            elif tipo == 'timeSignature':
                battiti_misura = info['num'] * 4.0 / info['den']
            elif tipo == 'keySignature':
                armatura = info['alterazioni']
            elif tipo == 'tie':
                tie_attivo = True
            elif tipo in ('clef', 'barline'):
                pass                                    # non cambiano il suono (i nomi delle note sono assoluti)
            else:
                avvisi.append(f"Simbolo {tipo} saltato: {info.get('token', token)}")
        except Exception as e:                          # qualunque imprevisto: salto il simbolo
            avvisi.append(f"Simbolo saltato per errore ({e}): {token}")

    if saltati_grace:
        avvisi.append(f"{saltati_grace} appoggiature (gracenote) ignorate")
    note.sort(key=lambda n: (n[1], n[0]))
    return note


# ---------------------------------------------------------------- dalla lista di note al MIDI
def tokens_to_midi(tokens, tempo=100, applica_armatura=False, gracenote='salta', avvisi=None, programma=0):
    """Lista di simboli -> oggetto pretty_midi.PrettyMIDI (si salva con .write('file.mid')).

    tempo: battiti (quarti) al minuto. avvisi: lista opzionale; se non la passi, i problemi
    vengono segnalati con un unico warnings.warn.
    """
    import pretty_midi
    propri = avvisi if avvisi is not None else []
    try:
        tempo = float(tempo)
        if not tempo > 0:
            raise ValueError
    except Exception:
        propri.append("Tempo non valido: uso 100")
        tempo = 100.0
    try:
        note = tokens_to_notes(tokens, applica_armatura, gracenote, propri)
    except Exception as e:                              # rete di sicurezza: mai errori per colpa dei simboli
        propri.append(f"Errore inatteso, MIDI vuoto: {e}")
        note = []
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    strumento = pretty_midi.Instrument(program=programma)
    secondi = 60.0 / tempo
    for midi, inizio, durata in note:
        strumento.notes.append(pretty_midi.Note(velocity=90, pitch=int(midi),
                                                start=inizio * secondi, end=(inizio + durata) * secondi))
    pm.instruments.append(strumento)
    if avvisi is None and propri:
        warnings.warn("tokens_to_midi: " + "; ".join(propri[:5]) + (" ..." if len(propri) > 5 else ""))
    return pm
