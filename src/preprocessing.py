"""Pre-elaborazione delle foto di spartiti.

foto -> lista di righi: ogni rigo è un'immagine binaria (uint8, inchiostro 0 su sfondo 255)
con un solo pentagramma, la chiave a sinistra e nient'altro (niente titoli, sillabe, numeri).
"""
import cv2
import numpy as np

LARGHEZZA = 2000   # larghezza di lavoro: tutti i parametri in pixel sono calibrati su questa


# ---------------------------------------------------------------- caricamento e pulizia
def carica(sorgente, larghezza=LARGHEZZA):
    """Percorso (o immagine BGR) -> (colore, grigio) alla larghezza di lavoro."""
    img = cv2.imread(sorgente) if isinstance(sorgente, str) else sorgente
    if img is None:
        raise ValueError(f"Non riesco a leggere l'immagine: {sorgente}")
    h, w = img.shape[:2]
    if w != larghezza:
        interp = cv2.INTER_AREA if w > larghezza else cv2.INTER_CUBIC
        img = cv2.resize(img, (larghezza, round(h * larghezza / w)), interpolation=interp)
    grigio = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, grigio


def binarizza(grigio, blur=5, block=51, C=20, min_area=20, max_sottile=150):
    """grigio -> binaria: inchiostro nero (0) su sfondo bianco (255)."""
    g = cv2.GaussianBlur(grigio, (blur, blur), 0)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)
    inchiostro = (b == 0).astype(np.uint8)
    n, etichette, st, _ = cv2.connectedComponentsWithStats(inchiostro, connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    larg = st[:, cv2.CC_STAT_WIDTH]
    alt = st[:, cv2.CC_STAT_HEIGHT]
    quasi_quadrato = (larg >= 5) & (larg <= 18) & (alt >= 5) & (alt <= 18)   # tipo punto di valore
    tieni = (area >= max_sottile) | ((area >= min_area) & quasi_quadrato)
    tieni[0] = False
    return np.where(tieni[etichette], 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- raddrizzamento
def ruota(img, angolo, bordo=255, interp=cv2.INTER_LINEAR):
    """Ruota di 'angolo' gradi (positivo = antiorario) senza tagliare gli angoli."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angolo, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=bordo)


def _punteggio(ink, angolo):
    r = ruota(ink, angolo, bordo=0)
    return np.var(r.sum(axis=1).astype(np.float64))       # righe nette = varianza alta


def stima_angolo(binaria, max_angolo=10, scala=0.25):
    ink = (binaria == 0).astype(np.float32)
    h, w = ink.shape
    ink = ink[:, int(w * 0.08):int(w * 0.92)]
    piccola = cv2.resize(ink, None, fx=scala, fy=scala, interpolation=cv2.INTER_AREA)
    grossi = np.arange(-max_angolo, max_angolo + 0.01, 1.0)
    a0 = grossi[np.argmax([_punteggio(piccola, a) for a in grossi])]
    fini = np.arange(a0 - 1, a0 + 1.01, 0.1)
    return float(fini[np.argmax([_punteggio(piccola, a) for a in fini])])


def raddrizza(binaria):
    """binaria -> (binaria dritta, angolo usato)."""
    angolo = stima_angolo(binaria)
    r = ruota(binaria, angolo, bordo=255)
    return np.where(r < 128, 0, 255).astype(np.uint8), angolo


# ---------------------------------------------------------------- pentagrammi
def trova_pentagrammi(binaria, soglia_picco=0.3, minimo_assoluto=0.15, salto=3.0):
    """Trova i pentagrammi con la proiezione orizzontale.

    Le righe di un pentagramma sono linee lunghe: sommando i pixel neri di ogni riga
    dell'immagine, le righe del pentagramma danno picchi molto più alti di note, testi e sillabe.

    Restituisce una lista di dizionari: y0, y1 (prima e ultima riga), s (distanza tra le righe),
    x0, x1 (estremi orizzontali).
    """
    h, w = binaria.shape
    ink = (binaria == 0).astype(np.uint8)
    # 1) proiezione orizzontale, leggermente smussata (le righe vere sono un po' ondulate)
    proiezione = np.convolve(ink.sum(axis=1).astype(np.float32), np.ones(3, np.float32) / 3, mode='same')
    soglia = max(soglia_picco * proiezione.max(), minimo_assoluto * w)
    righe = np.where(proiezione > soglia)[0]
    if len(righe) == 0:
        return []
    # 2) righe consecutive -> una sola linea (spessa 3-5 px): ne prendo il centro
    centri, inizio, prec = [], righe[0], righe[0]
    for y in righe[1:]:
        if y - prec > 2:
            centri.append((inizio + prec) / 2)
            inizio = y
        prec = y
    centri.append((inizio + prec) / 2)
    if len(centri) < 4:
        return []
    # 3) linee vicine = stesso pentagramma; uno stacco grande = pentagramma nuovo
    salti = np.diff(centri)
    s_tipico = np.median(salti)
    gruppi, corrente = [], [centri[0]]
    for c, d in zip(centri[1:], salti):
        if d > salto * s_tipico:
            gruppi.append(corrente)
            corrente = []
        corrente.append(c)
    gruppi.append(corrente)
    # 4) tengo i gruppi di 4-6 linee (5 ideali; una può mancare o raddoppiare)
    pentagrammi = []
    for g in gruppi:
        if not 4 <= len(g) <= 6:
            continue
        y0, y1 = g[0], g[-1]
        fascia = ink[max(0, int(y0) - 2):int(y1) + 3]
        colonne = np.where(fascia.sum(axis=0) >= 3)[0]      # colonne con le righe del pentagramma
        pentagrammi.append(dict(y0=float(y0), y1=float(y1), s=float(np.median(np.diff(g))),
                                x0=int(colonne[0]), x1=int(colonne[-1])))
    return pentagrammi


# ---------------------------------------------------------------- ritaglio
def ritaglia_righi(binaria, pentagrammi, margine=3.5, tolleranza=0.3, margine_x=1.0):
    """Ritaglia ogni pentagramma.

    margine: spazio sopra e sotto, in "distanze tra le righe" (s)
    tolleranza: un segno resta solo se tocca la fascia del pentagramma (± tolleranza * s):
                titoli, sillabe e numeri restano fuori
    margine_x: margine bianco a sinistra e a destra, in distanze tra le righe
    """
    if not pentagrammi:
        return []
    h, w = binaria.shape
    pad = int(max(p['s'] for p in pentagrammi) * (margine + 1)) + 5
    P = cv2.copyMakeBorder(binaria, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)
    righi = []
    for p in pentagrammi:
        s = p['s']
        alto = int(p['y0'] - margine * s) + pad
        basso = int(p['y1'] + margine * s) + pad
        crop = P[alto:basso, :]
        ink = (crop == 0).astype(np.uint8)
        n, etichette, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        fascia_a = p['y0'] - tolleranza * s + pad - alto
        fascia_b = p['y1'] + tolleranza * s + pad - alto
        y = st[:, cv2.CC_STAT_TOP]
        hh = st[:, cv2.CC_STAT_HEIGHT]
        tocca_fascia = (y <= fascia_b) & (y + hh >= fascia_a)
        attraversa_tutto = (y <= 0) & (y + hh >= crop.shape[0])    # tipo bordo del foglio
        tieni = tocca_fascia & ~attraversa_tutto
        tieni[0] = False
        pulito = np.where(tieni[etichette], 0, 255).astype(np.uint8)
        colonne = np.where((pulito == 0).any(axis=0))[0]
        if len(colonne) == 0:
            continue
        m = int(margine_x * s)
        righi.append(pulito[:, max(0, colonne[0] - m):min(w, colonne[-1] + m + 1)])
    return righi


# ---------------------------------------------------------------- funzione unica
def foto_a_righi(foto, margine=3.5, con_info=False):
    """Foto (percorso o immagine) -> lista di righi, dall'alto verso il basso.

    Se non trova nessun pentagramma restituisce una lista vuota.
    """
    _, grigio = carica(foto)
    dritta, angolo = raddrizza(binarizza(grigio))
    pentagrammi = trova_pentagrammi(dritta)
    righi = ritaglia_righi(dritta, pentagrammi, margine=margine)
    if con_info:
        return righi, dict(angolo=angolo, pentagrammi=pentagrammi)
    return righi
