"""MozartAI · app web (Gradio): fotografa uno spartito, ascolta la melodia.

Avvio:
    MOZART_MODELLO=/percorso/best.pt python app/app.py            # in locale
    MOZART_MODELLO=/percorso/best.pt MOZART_SHARE=1 python app/app.py   # con link pubblico temporaneo

La logica (elabora_foto, riascolta, costruisci_audio) non dipende da Gradio: si può provare anche senza interfaccia.
"""
import base64
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE / 'src'))

from audio import ErroreAudio, staves_to_midi                                    # noqa: E402
from midi_convert import tokens_to_midi                                          # noqa: E402
from pipeline import carica_modello_una_volta, foto_a_audio                      # noqa: E402
from strumenti import NOMI_STRUMENTI, STRUMENTO_PREDEFINITO, midi_con_strumento, sintetizza   # noqa: E402

# ---------------------------------------------------------------- modello
CANDIDATI_MODELLO = [
    os.environ.get('MOZART_MODELLO'),
    str(RADICE / 'app' / 'modello' / 'best.pt'),
    str(RADICE / 'best.pt'),
    '/content/drive/MyDrive/MozartAI/checkpoints_v2/best.pt',
]


def trova_modello():
    """Percorso di best.pt (o None se non si trova)."""
    for percorso in CANDIDATI_MODELLO:
        if percorso and os.path.exists(percorso):
            return percorso
    return None


def precarica():
    """Carica il modello subito (all'avvio dell'app), così la prima foto non aspetta."""
    percorso = trova_modello()
    if percorso:
        carica_modello_una_volta(percorso)
    return percorso


# ---------------------------------------------------------------- logica (senza Gradio)
def _a_rgb(rigo):
    """Il rigo è in scala di grigi: lo porto a 3 canali per mostrarlo nella galleria."""
    import cv2
    return cv2.cvtColor(rigo, cv2.COLOR_GRAY2RGB) if rigo.ndim == 2 else rigo


def _salva_midi(pm):
    cartella = tempfile.mkdtemp(prefix='mozartai_')
    percorso = os.path.join(cartella, 'melodia.mid')
    pm.write(percorso)
    return percorso


def costruisci_audio(simboli, tempo=100, strumento=STRUMENTO_PREDEFINITO, trasposizione=0, appoggiature=False):
    """Simboli -> ((frequenza, array), percorso del .mid, numero di note, avvisi).

    Veloce (non usa il modello): serve anche per riascoltare con un altro strumento o un'altra velocità.
    Solleva ErroreAudio (con un messaggio comprensibile) se non ci sono note.
    """
    avvisi = []
    midi_righi = [tokens_to_midi(s, tempo=tempo, gracenote='breve' if appoggiature else 'salta', avvisi=avvisi)
                  for s in simboli]
    brano = staves_to_midi(midi_righi, avvisi=avvisi)
    brano = midi_con_strumento(brano, strumento, trasposizione)          # programma MIDI + trasposizione
    audio = sintetizza(brano, strumento)
    n_note = sum(len(s.notes) for s in brano.instruments)
    return audio, _salva_midi(brano), n_note, avvisi


def elabora_foto(foto, strumento=STRUMENTO_PREDEFINITO, tempo=100, trasposizione=0, appoggiature=False):
    """Foto -> dizionario con ok, messaggio, righi, testo, audio, midi, simboli. Non solleva mai errori."""
    inizio = time.perf_counter()
    esito = dict(ok=False, messaggio='', righi=[], testo='', audio=None, midi=None, simboli=[])
    try:
        if foto is None:
            esito['messaggio'] = "📷 Scatta una foto o caricane una, poi premi **Riconosci e suona**."
            return esito
        modello = trova_modello()
        if modello is None:
            esito['messaggio'] = ("⚠️ Il modello non è stato trovato sul server. Non è colpa della tua foto: "
                                  "avvisa chi gestisce l'app.")
            return esito

        ris = foto_a_audio(foto, modello, tempo=tempo, gracenote='breve' if appoggiature else 'salta')
        # foto girata di lato? Se non trova pentagrammi, riprovo ruotando di 90° nei due versi
        if not ris.righi and isinstance(foto, np.ndarray) and foto.ndim == 3:
            for k in (1, 3):
                prova = foto_a_audio(np.ascontiguousarray(np.rot90(foto, k)), modello, tempo=tempo)
                if prova.righi:
                    ris = prova
                    break

        esito['righi'] = [_a_rgb(r) for r in ris.righi]
        esito['simboli'] = ris.simboli
        esito['testo'] = ris.testo_simboli() if ris.simboli else ''
        if not ris.ok:
            esito['messaggio'] = "⚠️ " + ris.messaggio
            return esito

        audio, midi, n_note, avvisi = costruisci_audio(ris.simboli, tempo, strumento, trasposizione, appoggiature)
        esito.update(ok=True, audio=audio, midi=midi)
        secondi = time.perf_counter() - inizio
        esito['messaggio'] = (f"✅ **Ho riconosciuto {len(ris.righi)} rigo/righi e {n_note} note.** "
                              f"(elaborato in {secondi:.1f} s)\n\nSe la melodia non ti convince, guarda "
                              "*Cosa ha capito il modello* qui sotto: il modello può sbagliare.")
        if avvisi:
            esito['messaggio'] += f"\n\n_Alcuni simboli sono stati saltati ({len(avvisi)} avvisi)._"
        return esito
    except ErroreAudio as e:
        esito['messaggio'] = f"⚠️ {e}"
        return esito
    except Exception:
        esito['ok'] = False
        esito['messaggio'] = "⚠️ Qualcosa è andato storto. Riprova con un'altra foto."
        return esito


def riascolta(simboli, strumento, tempo, trasposizione, appoggiature):
    """Rifà solo il suono (senza rieseguire il modello). Restituisce (audio, midi, messaggio_di_errore_o_None)."""
    if not simboli:
        return None
    try:
        audio, midi, _, _ = costruisci_audio(simboli, tempo, strumento, trasposizione, appoggiature)
        return audio, midi, None
    except ErroreAudio as e:
        return None, None, f"⚠️ {e}"
    except Exception:
        return None, None, "⚠️ Non riesco a rifare il suono. Riprova."


# ---------------------------------------------------------------- aspetto
CSS = """
.gradio-container {max-width: 820px !important; margin: auto !important;}
.hero {background: linear-gradient(135deg, #4338ca 0%, #7c3aed 55%, #db2777 100%); color: white;
       border-radius: 22px; padding: 26px 20px; text-align: center; box-shadow: 0 12px 30px rgba(67,56,202,.35);}
.hero .nota {font-size: 2.6rem; line-height: 1; margin-bottom: 4px;}
.hero h1 {font-size: 2.3rem; margin: 0 0 6px; color: white !important; letter-spacing: .5px;}
.hero p {margin: 0; font-size: 1.05rem; opacity: .95; color: white !important;}
.hero img {height: 70px; border-radius: 14px; margin-bottom: 8px; background: white; padding: 4px;}
.scheda {border-radius: 18px !important; padding: 14px !important; margin-top: 14px !important;
         box-shadow: 0 2px 14px rgba(0,0,0,.08);}
.scheda h3 {margin-top: 0;}
#bottone-principale {font-size: 1.15rem; padding: 14px; border-radius: 14px; border: none; color: white;
                     background: linear-gradient(135deg, #4f46e5, #7c3aed);}
#bottone-principale:hover {filter: brightness(1.08);}
.piede {text-align: center; opacity: .7; font-size: .85rem; margin: 18px 0 8px;}
@media (max-width: 600px) {
  .hero h1 {font-size: 1.7rem;}
  .gradio-container {padding: 6px !important;}
}
"""

CONSIGLI = """
**Per una foto che funziona:**
- 💡 tanta luce, senza ombre sullo spartito
- 📐 telefono **parallelo** al foglio, non di taglio
- 🎼 una **melodia su un solo pentagramma** (non pianoforte a due righi)
- 🖨️ meglio spartiti **stampati** e semplici
- ✂️ inquadra solo lo spartito, senza troppo sfondo
"""

LIMITI = """
- Il modello è stato addestrato su spartiti **sintetici**: con foto sfocate o spartiti dallo stile molto diverso può
  sbagliare note, armatura o tempo.
- Legge **un solo pentagramma per volta**; niente accordi, dinamiche o più voci.
- Se qualcosa suona strano, guarda i **simboli letti**: lì vedi cosa ha capito il modello.
- Il file **MIDI** contiene lo strumento scelto: aprendolo in MuseScore o GarageBand suona con il suono vero.
"""


def _logo_html():
    """Logo del progetto dentro l'intestazione (se esiste logo.png nella radice e non è enorme)."""
    percorso = RADICE / 'logo.png'
    try:
        if percorso.exists() and percorso.stat().st_size < 300_000:
            dati = base64.b64encode(percorso.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{dati}" alt="logo">'
    except Exception:
        pass
    return '<div class="nota">♪</div>'


def _intestazione():
    return (f'<div class="hero">{_logo_html()}<h1>MozartAI</h1>'
            '<p>Fotografa uno spartito, ascolta la melodia.</p></div>')


PIEDE = '<div class="piede">Progetto scolastico · Daniele, Lorenzo e Martino · riconoscimento musicale con CRNN + CTC</div>'


def crea_app():
    """Costruisce l'interfaccia Gradio (richiede 'pip install gradio')."""
    import gradio as gr

    tema = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", radius_size="lg")

    def _elabora(foto, strumento, tempo, trasposizione, appoggiature, progress=gr.Progress()):
        progress(0.2, desc="Cerco i pentagrammi e leggo le note…")
        e = elabora_foto(foto, strumento, tempo, trasposizione, appoggiature)
        return e['messaggio'], (e['righi'] or None), e['testo'], e['audio'], e['midi'], e['simboli']

    def _riascolta(simboli, strumento, tempo, trasposizione, appoggiature):
        r = riascolta(simboli, strumento, tempo, trasposizione, appoggiature)
        if r is None:                                     # ancora nessuna foto elaborata
            return gr.update(), gr.update(), gr.update()
        audio, midi, errore = r
        if errore:
            return errore, None, None
        return gr.update(), audio, midi

    with gr.Blocks(theme=tema, css=CSS, title="MozartAI") as demo:
        gr.HTML(_intestazione())
        stato = gr.State([])

        with gr.Group(elem_classes="scheda"):
            gr.Markdown("### 1 · Fotografa lo spartito")
            foto = gr.Image(label="Foto dello spartito", sources=["upload", "webcam"], type="numpy", height=320)
            with gr.Accordion("💡 Come fare una buona foto", open=False):
                gr.Markdown(CONSIGLI)

        with gr.Group(elem_classes="scheda"):
            gr.Markdown("### 2 · Scegli come suonarlo")
            strumento = gr.Dropdown(choices=NOMI_STRUMENTI, value=STRUMENTO_PREDEFINITO, label="🎻 Strumento")
            tempo = gr.Slider(40, 200, value=100, step=5, label="⏱️ Velocità (battiti al minuto)")
            with gr.Accordion("Opzioni avanzate", open=False):
                trasposizione = gr.Slider(-12, 12, value=0, step=1, label="Trasposizione (semitoni, + = più acuto)")
                appoggiature = gr.Checkbox(value=False, label="Suona anche le appoggiature (le note piccole)")
            bottone = gr.Button("🎵 Riconosci e suona", variant="primary", elem_id="bottone-principale")

        with gr.Group(elem_classes="scheda"):
            gr.Markdown("### 3 · Il risultato")
            messaggio = gr.Markdown("📷 Scatta una foto o caricane una, poi premi **Riconosci e suona**.")
            audio = gr.Audio(label="🔊 Ascolta", type="numpy", interactive=False)
            midi = gr.File(label="⬇️ Scarica il file MIDI", interactive=False)
            with gr.Accordion("🔍 Cosa ha capito il modello", open=False):
                galleria = gr.Gallery(label="Righi riconosciuti", columns=1, height=300, object_fit="contain")
                testo = gr.Textbox(label="Simboli letti", lines=6, interactive=False)

        with gr.Accordion("ℹ️ Limiti", open=False):
            gr.Markdown(LIMITI)
        gr.HTML(PIEDE)

        ingressi = [foto, strumento, tempo, trasposizione, appoggiature]
        bottone.click(_elabora, inputs=ingressi, outputs=[messaggio, galleria, testo, audio, midi, stato])
        # cambiando strumento, velocità o trasposizione si rifà solo il suono (senza rieseguire il modello)
        rifai = [stato, strumento, tempo, trasposizione, appoggiature]
        uscite = [messaggio, audio, midi]
        strumento.change(_riascolta, inputs=rifai, outputs=uscite)
        tempo.release(_riascolta, inputs=rifai, outputs=uscite)
        trasposizione.release(_riascolta, inputs=rifai, outputs=uscite)
        appoggiature.change(_riascolta, inputs=rifai, outputs=uscite)
    return demo


def main():
    percorso = precarica()
    print("Modello:", percorso or "NON TROVATO (imposta la variabile MOZART_MODELLO)")
    demo = crea_app()
    demo.queue(max_size=8).launch(server_name="0.0.0.0", share=os.environ.get('MOZART_SHARE') == '1')


if __name__ == "__main__":
    main()
