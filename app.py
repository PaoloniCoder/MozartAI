"""MozartAI · app web (Gradio): fotografa uno spartito, ascolta la melodia.

Avvio:
    MOZART_MODELLO=/percorso/best.pt python app.py            # in locale
    MOZART_MODELLO=/percorso/best.pt MOZART_SHARE=1 python app.py   # con link pubblico temporaneo

La logica (elabora_foto, riascolta, costruisci_audio) non dipende da Gradio: si può provare anche senza interfaccia.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Funziona con qualsiasi disposizione dei file: app/app.py + src/, oppure app.py nella radice, oppure tutto in una cartella
AQUI = Path(__file__).resolve().parent
RADICE = AQUI if ((AQUI / 'src').exists() or (AQUI / 'pipeline.py').exists()) else AQUI.parent
for _cartella in (AQUI, RADICE / 'src'):
    if _cartella.exists() and str(_cartella) not in sys.path:
        sys.path.insert(0, str(_cartella))

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
ASSET = RADICE / 'app'
CSS = (ASSET / 'studio.css').read_text(encoding='utf-8')
JS = (ASSET / 'studio.js').read_text(encoding='utf-8')
HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11.2.10/swiper-bundle.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/aos@2.3.4/dist/aos.css">
<script defer src="https://cdn.jsdelivr.net/npm/swiper@11.2.10/swiper-bundle.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/ScrollTrigger.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/aos@2.3.4/dist/aos.js"></script>
"""

MARCA = """
<header class="studio-bar">
  <div class="studio-brandmark"><span aria-hidden="true">♪</span>MozartAI</div>
  <a href="#studio-controls">Analizza uno spartito</a>
</header>
"""
HERO = """
<section class="studio-hero" aria-labelledby="studio-title">
  <div class="studio-score" aria-hidden="true"><i></i><i></i><i></i></div>
  <h1 id="studio-title">Dallo spartito <span>al suono.</span></h1>
  <p>Carica o fotografa una partitura, scegli uno strumento e ascolta la melodia.</p>
</section>
"""
CONSIGLI = """
<div class="swiper studio-tips">
  <ul class="swiper-wrapper">
    <li class="swiper-slide"><strong>Illumina il foglio</strong>Usa luce uniforme ed evita le ombre.</li>
    <li class="swiper-slide"><strong>Tieni il telefono parallelo</strong>Inquadra la pagina dall’alto, senza inclinarla.</li>
    <li class="swiper-slide"><strong>Scegli una melodia semplice</strong>Usa uno spartito stampato su un solo pentagramma.</li>
    <li class="swiper-slide"><strong>Riempi l’inquadratura</strong>Taglia lo sfondo e lascia visibile tutto lo spartito.</li>
  </ul>
  <div class="studio-tip-nav">
    <button type="button" class="studio-tip-prev" aria-label="Consiglio precedente">←</button>
    <button type="button" class="studio-tip-next" aria-label="Consiglio successivo">→</button>
  </div>
</div>
"""

LIMITI = """
- Il modello è stato addestrato su spartiti **sintetici**: con foto sfocate o spartiti dallo stile molto diverso può
  sbagliare note, armatura o tempo.
- Legge **un solo pentagramma per volta**; niente accordi, dinamiche o più voci.
- Se qualcosa suona strano, guarda i **simboli letti**: lì vedi cosa ha capito il modello.
- Il file **MIDI** contiene lo strumento scelto: aprendolo in MuseScore o GarageBand suona con il suono vero.
"""


PIEDE = '<div class="studio-footer">MozartAI · Progetto scolastico di Daniele, Lorenzo e Martino</div>'


def _tema(gr):
    return gr.themes.Soft(primary_hue="lime", neutral_hue="slate", radius_size="lg").set(
        body_background_fill="#101416", body_background_fill_dark="#101416",
        body_text_color="#f2f4ed", body_text_color_dark="#f2f4ed",
        block_background_fill="#191e21", block_background_fill_dark="#191e21",
        block_border_color="#353e40", block_border_color_dark="#353e40",
        block_label_background_fill="#22292b", block_label_background_fill_dark="#22292b",
        block_label_text_color="#e3ebe2", block_label_text_color_dark="#e3ebe2",
        block_title_text_color="#f2f4ed", block_title_text_color_dark="#f2f4ed",
        input_background_fill="#22292b", input_background_fill_dark="#22292b",
        input_border_color="#414d49", input_border_color_dark="#414d49",
        background_fill_primary="#191e21", background_fill_primary_dark="#191e21",
        background_fill_secondary="#22292b", background_fill_secondary_dark="#22292b",
        button_secondary_background_fill="#2a3334", button_secondary_background_fill_dark="#2a3334",
        button_secondary_text_color="#f2f4ed", button_secondary_text_color_dark="#f2f4ed",
        slider_color="#b5d96a", slider_color_dark="#b5d96a",
    )


def crea_app():
    """Costruisce l'interfaccia Gradio."""
    import gradio as gr

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

    with gr.Blocks(title="MozartAI", fill_width=True) as demo:
        gr.HTML(MARCA, elem_id="studio-brand")
        gr.HTML(HERO, elem_id="studio-hero")
        stato = gr.State([])

        with gr.Row(elem_id="studio-controls"):
            with gr.Column(scale=3, min_width=300, elem_classes="studio-card"):
                gr.Markdown("## Foto dello spartito")
                foto = gr.Image(label="Foto dello spartito", show_label=False, sources=["upload", "webcam"],
                                type="numpy", height=320)
                with gr.Accordion("Come fare una buona foto", open=False):
                    gr.HTML(CONSIGLI)

            with gr.Column(scale=2, min_width=270, elem_classes="studio-card", elem_id="studio-settings"):
                gr.Markdown("## Come vuoi suonarlo?")
                strumento = gr.Dropdown(choices=NOMI_STRUMENTI, value=STRUMENTO_PREDEFINITO, label="Strumento")
                tempo = gr.Slider(40, 200, value=100, step=5, label="Velocità (battiti al minuto)")
                with gr.Accordion("Opzioni avanzate", open=False):
                    trasposizione = gr.Slider(-12, 12, value=0, step=1, label="Trasposizione (semitoni, + = più acuto)")
                    appoggiature = gr.Checkbox(value=False, label="Suona anche le appoggiature (le note piccole)")
                bottone = gr.Button("Riconosci e suona", variant="primary", elem_id="bottone-principale")

        with gr.Group(elem_classes="studio-card"):
            gr.Markdown("## Il risultato")
            messaggio = gr.Markdown("Carica o scatta una foto e premi **Riconosci e suona**.", elem_id="studio-status")
            audio = gr.Audio(label="🔊 Ascolta", type="numpy", interactive=False)
            midi = gr.File(label="⬇️ Scarica il file MIDI", interactive=False)
            with gr.Accordion("🔍 Cosa ha capito il modello", open=False):
                galleria = gr.Gallery(label="Righi riconosciuti", columns=1, height=300, object_fit="contain")
                testo = gr.Textbox(label="Simboli letti", lines=6, interactive=False)

        with gr.Accordion("ℹ️ Limiti", open=False):
            gr.Markdown(LIMITI)
        gr.HTML(PIEDE, elem_id="studio-footer")

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


def avvia_app(**opzioni):
    """Avvia l'interfaccia con gli asset grafici, in locale o in Colab."""
    import gradio as gr

    return crea_app().queue(max_size=8).launch(theme=_tema(gr), css=CSS, js=JS, head=HEAD, **opzioni)


def main():
    percorso = precarica()
    print("Modello:", percorso or "NON TROVATO (imposta la variabile MOZART_MODELLO)")
    avvia_app(server_name="0.0.0.0", share=os.environ.get('MOZART_SHARE') == '1')


if __name__ == "__main__":
    main()
