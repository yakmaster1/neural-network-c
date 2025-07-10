# Neuronales Netzwerk - Programmiert in C
#
Ein vollständig entwickeltes neuronales Netzwerk in C, mit Konsoleninterface, Zeichenfenster, Trainingsfunktionen, MNIST-Unterstützung und einem modellbasierten Dateisystem zur Speicherung und Wiederverwendung von Gewichtskonfigurationen.

Link zur Dokumentation & Präsentation: [coming soon]
#
#
#
**Features**
- Dynamisch konfigurierbare Netzstruktur (z. B. 784–64–32–10)
- He-Initialisierung der Gewichte (für ReLU)
- Softmax-Ausgabe + Cross-Entropy-Kostenfunktion
- Trainingsdaten im MNIST-Format (IDX)
- CLI mit Speicher-/Ladefunktion (`data/a.txt`)
- Windows-Zeichenfenster zum digitalen Testen von Ziffern
#
#
#
**Voraussetzungen**
- GCC oder MinGW (für Windows)
- `windows.h` (nur unter Windows)
- MNIST-Datensätze (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`) im `./data/`-Ordner
#
#
#
### Kompilieren unter Windows:

```bash
gcc main.c neural_network.c draw_window.c image_extr.c alglib.c -o nn.exe -lgdi32
```
#
#
#
Dieses Projekt wurde im Rahmen eines Praxistransferprojekts an der Rheinischen Hochschule Köln entwickelt.
Alle Inhalte, einschließlich Quellcode, Dokumentation, Grafiken und begleitende Materialien, wurden vom Autor eigenständig erstellt und unterliegen dem Urheberrecht des Autors.
Die Veröffentlichung dieses Projekts dient ausschließlich wissenschaftlichen, akademischen und nicht-kommerziellen Zwecken. Eine kommerzielle Nutzung, Vervielfältigung oder Verwertung über die reine Lehre hinaus ist ohne ausdrückliche schriftliche Zustimmung des Autors nicht gestattet.

Bei Fragen zur Nutzung oder Weiterverarbeitung wenden Sie sich bitte an den Autor.