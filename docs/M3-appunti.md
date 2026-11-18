## Validazione con i .mid originali (Issue 2)

Metodo: per N esempi casuali di PrIMuS (seed 42) si converte il `.semantic` con
`tokens_to_midi` e si confronta il risultato con il `.mid` originale, a tre livelli:
altezze, inizi in proporzione, durate in proporzione (tolleranze 1% e 20%).

### Risultati
- Esempi confrontati: ____
- Altezze identiche: ____%
- Inizi identici (in proporzione): ____%
- Durate identiche (in proporzione): ____%
- **Esempi identici (tutti e tre i livelli): ____%**

### Cause delle differenze
(incolla la tabella della cella 5)

### Correzioni fatte
- ____ (es. armatura applicata di default: dal __% al __% di esempi identici)
- ____

### Casi che non gestiamo
- Simboli non riconosciuti (cella 7): ____
- Appoggiature (`gracenote`): ignorate nell'app (il modello ne inventa molte).
- Terzine, dinamiche e più voci.
- Tempo fisso a 100 battiti al minuto.
- Differenze residue non spiegate: ____
