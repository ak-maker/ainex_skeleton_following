import sys
import time
import torch
import cv2
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
ROUNDS = 4

infer_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def draw_results(frame, results: list[tuple[str, float]], round_num: int):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 40 + len(results) * 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Round {round_num}/{ROUNDS}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i, (label, prob) in enumerate(results):
        y = 64 + i * 36
        bar_w = int(prob * 300)
        cv2.rectangle(frame, (10, y - 20), (10 + bar_w, y), (0, 200, 80), -1)
        cv2.putText(frame, f"{label}: {prob:.1%}", (14, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def capture_and_classify(cap, model, classes, device, round_num: int):
    win = "Pose Classifier — SPACE to capture, Q to quit"

    # wait for space
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Round {round_num}/{ROUNDS} — SPACE to capture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord(' '):
            break

    # countdown
    for i in range(3, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, str(i), (frame.shape[1] // 2 - 40, frame.shape[0] // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
            cv2.imshow(win, frame)
            cv2.waitKey(1)

    ret, frame = cap.read()

    # classify
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = infer_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_prob, top_idx = probs.topk(min(3, len(classes)))
    results = [(classes[idx], prob.item()) for prob, idx in zip(top_prob, top_idx)]

    print(f"\nRound {round_num} results:")
    for label, prob in results:
        print(f"  {label}: {prob:.1%}")

    # show results for 3 seconds
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ret, display = cap.read()
        if not ret:
            break
        draw_results(display, results, round_num)
        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True


def load_model(model_path: str, num_classes: int, device: torch.device):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model


def classify(model_path: str = 'model.pt') -> None:
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    model = load_model(model_path, len(classes), device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        sys.exit(1)

    for round_num in range(1, ROUNDS + 1):
        if not capture_and_classify(cap, model, classes, device, round_num):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    classify()
