# Afvalherkenning_Jetson_Nano


Dit project is een **smart afvalsorteringssysteem** op basis van een
**NVIDIA Jetson Nano**.\
Met behulp van een camera herkent het systeem verschillende soorten
afval en geeft het via **LED's** aan in welke afvalbak het object hoort.

Het systeem gebruikt een **getraind ResNet18‑model (PyTorch)** om live
camerabeelden te classificeren.\
Doel: afvalsortering vereenvoudigen en **bewustwording rond recyclage**
vergroten.

------------------------------------------------------------------------

## Overzicht

-   Jetson Nano draait een **vooraf klaargemaakte SD‑kaart image** met
    alle dependencies.
-   Een **USB‑ of CSI‑camera** filmt het object.
-   Het script `live_bins.py`:
    -   classificeert elk cameraframe
    -   toont label + confidence op het scherm
    -   stuurt via GPIO de juiste LED aan

Modelbestand:\
`bins_resnet18.pth`

------------------------------------------------------------------------

## Typische use cases

-   **Les/demo**: tonen welke afvalsoorten het systeem herkent (beker,
    papier, PMD, restafval).
-   **Foutvoorbeelden** bespreken: lage confidence of verkeerde
    classificatie.

------------------------------------------------------------------------

## Hardware en software

### Hardware

-   NVIDIA Jetson Nano (2GB of 4GB) + voeding\
-   Micro‑SD kaart (min. 32 GB) met voorgebakken image\
-   USB‑camera of Raspberry Pi CSI‑camera\
-   4 LED's\
-   4 weerstanden (±220 Ω)\
-   Breadboard + jumper wires

### Software

Vooraf geïnstalleerd op de image:

-   Ubuntu voor Jetson Nano\
-   Python 3\
-   PyTorch + torchvision (ResNet18)\
-   OpenCV (`cv2`)\
-   Jetson.GPIO

Repository‑inhoud:

-   `live_bins.py`
-   `bins_resnet18.pth`

------------------------------------------------------------------------

## I/O overzicht

**Inputs** - USB‑ of CSI‑camera (live videobeeld)

**Outputs** - Monitor: camerafeed met label + confidence - LED's: -
cups - paper - PMD - residual

**Processing** - Hardware: NVIDIA Jetson Nano\
- Software: Python, PyTorch, OpenCV, Jetson.GPIO

------------------------------------------------------------------------

## Installatie (out‑of‑the‑box)

### 1. SD‑kaart flashen

1.  Download de image `jetson-smart-bins.img` (link in deze repo).
2.  Flash de image met **balenaEtcher** of **Raspberry Pi Imager**.
3.  Plaats de SD‑kaart in de Jetson Nano en start op.

Na het booten: - Alle dependencies zijn aanwezig - Project staat bv. in
`~/smart-bins/`

------------------------------------------------------------------------

### 2. Camera aansluiten

Sluit een USB‑camera aan.

Controleer of de camera herkend wordt:

``` bash
ls /dev/video*
```

`/dev/video0` moet bestaan.\
`live_bins.py` gebruikt standaard `VideoCapture(0)`.

------------------------------------------------------------------------

### 3. LED's en GPIO‑bekabeling

De code gebruikt **BOARD‑nummering**.

GPIO‑pinnen:

-   CUPS: `PIN 29`
-   PAPER: `PIN 11`
-   PMD: `PIN 13`
-   RESIDUAL: `PIN 15`

Stap‑voor‑stap:

1.  Plaats elke LED in serie met een weerstand (±220 Ω).
2.  Verbind de **korte poot** van elke LED met **GND**.
3.  Verbind de **lange poot** via de weerstand met de juiste GPIO‑pin.
4.  Controleer of de pinnen overeenkomen met `live_bins.py`.

Pas indien nodig de pin‑constanten aan in de code.

------------------------------------------------------------------------

## Code en werking

De kern van het project is `live_bins.py`.

### Model

-   ResNet18 (ImageNet‑pretrained)
-   Laatste fully connected laag aangepast
-   Getraind op afvalklassen

**Klassen** - `cups` - `paper` - `pmd` - `residual` - `other` (behandeld
als residual)

------------------------------------------------------------------------

### Pipeline per frame

1.  Cameraframe uitlezen met OpenCV
2.  BGR → RGB
3.  Omzetten naar PIL‑image
4.  Resizen & normaliseren
5.  Inference met ResNet18
6.  Softmax → probabilities
7.  Klasse + confidence bepalen
8.  Juiste LED aansturen via GPIO

------------------------------------------------------------------------

## Beslissingslogica (LED's & thresholds)

``` python
idx = int(np.argmax(probs))
label = class_names[idx]
conf = float(probs[idx])

all_leds_off()

if label == "cups":
    GPIO.output(PIN_CUPS, GPIO.HIGH)
elif label == "paper":
    GPIO.output(PIN_PAPER, GPIO.HIGH)
elif label == "pmd":
    if conf >= 0.80:
        GPIO.output(PIN_PMD, GPIO.HIGH)
elif label == "residual" or label == "other":
    if conf >= 0.60:
        GPIO.output(PIN_RESIDUAL, GPIO.HIGH)
```

**Uitleg** - Cups & paper: LED altijd aan bij voorspelling - PMD: alleen
bij ≥ **80% confidence** - Residual/other: alleen bij ≥ **60%
confidence**

Op het camerabeeld verschijnt ook:

    label (xx.x%)

Handig om tijdens demo's confidence en fouten te bespreken.

------------------------------------------------------------------------

## Demo starten

``` bash
cd ~/smart-bins
python3 live_bins.py
```

-   OpenCV‑venster met live camerabeeld verschijnt
-   Houd afvalobjecten voor de camera
-   Observeer:
    -   Label + confidence op het scherm
    -   Juiste LED die oplicht

**Stoppen** - Druk op `q` in het cameravenster

------------------------------------------------------------------------

## Opmerkingen

-   ML‑voorspellingen zijn **niet perfect**
-   Licht, achtergrond en objectpositie beïnvloeden de accuracy
-   Ideaal project om **AI‑beperkingen** uit te leggen in een educatieve
    context
