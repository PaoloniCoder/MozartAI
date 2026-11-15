"""Modello CRNN + CTC addestrato in M1 e funzioni per usarlo fuori dal notebook.
Uso tipico:
    model, id2tok = carica_modello('/percorso/best.pt')
    simboli = foto_a_simboli('/percorso/foto.jpg', model, id2tok) 
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_H = 128   # altezza a cui il modello è stato addestrato (viene comunque letta dal checkpoint)

class CRNN(nn.Module):
    """Deve essere IDENTICA a quella usata nell'addestramento, altrimenti i pesi non si caricano."""

    def __init__(self, n_classes, img_h=IMG_H):
        super().__init__()

        def blk(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(True), nn.MaxPool2d(pool))

        # altezza /16, larghezza /4: servono molte "fettine" per scrivere tutti i simboli (CTC)
        self.cnn = nn.Sequential(blk(1, 32, 2), blk(32, 64, 2), blk(64, 128, (2, 1)), blk(128, 256, (2, 1)))
        self.proj = nn.Linear(256 * (img_h // 16), 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.cnn(x)                                    # (B, C, H', T)
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)     # (B, T, C*H')
        f, _ = self.rnn(torch.relu(self.proj(f)))
        return self.fc(f)                                  # (B, T, classi)

def carica_modello(percorso, device='cpu'):
    """best.pt -> (modello in modalità valutazione, dizionario numero -> simbolo)."""
    ck = torch.load(percorso, map_location=device, weights_only=False)
    if not isinstance(ck, dict) or 'model' not in ck or 'tok2id' not in ck:
        raise ValueError("Il checkpoint non ha le chiavi attese ('model', 'tok2id'): "
                         "è stato salvato dal notebook di addestramento giusto?")
    tok2id = ck['tok2id']
    model = CRNN(len(tok2id) + 1, ck.get('img_h', IMG_H)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    model.img_h = ck.get('img_h', IMG_H)
    id2tok = {i: t for t, i in tok2id.items()}
    return model, id2tok

def prepara_rigo(rigo, img_h=IMG_H):
    """Immagine di un rigo (uint8, scala di grigi, inchiostro scuro su sfondo chiaro) -> tensore (1, 1, H, W).

    Stessa preparazione dell'addestramento: altezza fissa, valori 0-1, sfondo = 1.
    """
    h, w = rigo.shape[:2]
    nuova_w = max(8, round(w * img_h / h))
    im = Image.fromarray(rigo).convert('L').resize((nuova_w, img_h), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(im, dtype=np.float32) / 255.0)
    return x.unsqueeze(0).unsqueeze(0)

def decodifica_greedy(logits, lunghezza):
    """Per ogni fettina prende il simbolo più probabile, unisce i ripetuti, toglie i blank (0)."""
    migliori = logits[:lunghezza].argmax(-1).tolist()
    out, prec = [], 0
    for p in migliori:
        if p != prec and p != 0:
            out.append(p)
        prec = p
    return out

@torch.no_grad()
def riconosci_righi(righi, model, id2tok, device='cpu'):
    """Lista di righi (immagini) -> lista di liste di simboli, nello stesso ordine."""
    img_h = getattr(model, 'img_h', IMG_H)
    risultati = []
    for rigo in righi:
        x = prepara_rigo(rigo, img_h).to(device)
        logits = model(x).float().cpu()[0]                 # (T, classi)
        ids = decodifica_greedy(logits, x.shape[3] // 4)   # T = larghezza // 4
        risultati.append([id2tok[i] for i in ids])
    return risultati

def foto_a_simboli(foto, model, id2tok, device='cpu'):
    """Foto (percorso o immagine) -> lista di righi, ognuno una lista di simboli. Vuota se non trova pentagrammi."""
    from preprocessing import foto_a_righi          # import qui: model.py si può usare anche senza OpenCV
    righi = foto_a_righi(foto)
    return riconosci_righi(righi, model, id2tok, device)
