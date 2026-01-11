import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import Jetson.GPIO as GPIO
import time

# ---------------- GPIO SETUP ----------------
# BOARD-pin nummers
PIN_CUPS     = 29   # cups-led
PIN_PAPER    = 11   # paper-led
PIN_PMD      = 13   # pmd-led
PIN_RESIDUAL = 15   # residual-led

GPIO.setmode(GPIO.BOARD)
for pin in [PIN_CUPS, PIN_PAPER, PIN_PMD, PIN_RESIDUAL]:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

def all_leds_off():
    GPIO.output(PIN_CUPS, GPIO.LOW)
    GPIO.output(PIN_PAPER, GPIO.LOW)
    GPIO.output(PIN_PMD, GPIO.LOW)
    GPIO.output(PIN_RESIDUAL, GPIO.LOW)

# --------------- MODEL LADEN ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("bins_resnet18.pth", map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet18(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Classes in model:", class_names)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------- CAMERA LOOP ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera niet gevonden")
    GPIO.cleanup()
    raise SystemExit

print("Druk op 'q' om te stoppen.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB -> PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        x = tf(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        label = class_names[idx]
        conf = float(probs[idx])

        # LEDs zetten
        all_leds_off()

        if label == "cups":
            GPIO.output(PIN_CUPS, GPIO.HIGH)
        elif label == "paper":
            GPIO.output(PIN_PAPER, GPIO.HIGH)
        elif label == "pmd":
            if conf >= 0.80:   # PMD-led alleen bij ≥ 80% confidence
                GPIO.output(PIN_PMD, GPIO.HIGH)
        elif label == "residual" or label == "other":
            if conf >= 0.60:   # residual-led alleen bij ≥ 60% confidence
                GPIO.output(PIN_RESIDUAL, GPIO.HIGH)

        # Tekst overlay + print
        text = "{} ({:.1f}%)".format(label, conf * 100.0)
        print(text)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("bins_live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    all_leds_off()
    GPIO.cleanup()
