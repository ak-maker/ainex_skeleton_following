#!/usr/bin/env python3
"""
Two-phase MobileNetV3-Large fine-tuner.

Phase 1: freeze backbone, train classifier head only  — fast convergence.
Phase 2: unfreeze all layers, full fine-tune          — conservative LR.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from datetime import datetime
import pillow_heif
pillow_heif.register_heif_opener()
from sklearn.metrics import confusion_matrix

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

# ── Hyperparameters ───────────────────────────────────────────────────────────
DATA_DIR    = 'data'
BATCH_SIZE  = 32
NUM_WORKERS = 4

P1_LR     = 1e-3   # head-only phase  — fresh linear layer, can be aggressive
P1_EPOCHS = 5
P1_WD     = 1e-4

P2_LR     = 1e-4   # full fine-tune   — nudge pretrained features, don't overwrite
P2_EPOCHS = 15
P2_WD     = 1e-4
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# these transform the images into the right size, brightness, etc. for standardized results
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# does the same for the validation set
# CORRECTION: not quite the same — val_tf skips all augmentation (no random crop, flip, or color jitter). Augmentation is only for training to prevent overfitting; validation uses a clean, deterministic pipeline so results are comparable across epochs.
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        return img.convert('RGB')


class NestedImageDataset(Dataset):
    """Loads images from data/<split>/<class>/<subfolder>/<image>."""
    def __init__(self, root: str, transform=None):
        self.transform = transform
        root = Path(root)
        self.classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (p, self.class_to_idx[p.parts[len(root.parts)]])
            for class_dir in root.iterdir() if class_dir.is_dir()
            for p in class_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = pil_loader(str(path))
        if self.transform:
            img = self.transform(img)
        return img, label


# builds the "loaders":  pull the data from the libary into a pytorch object
def make_loaders():
    train_ds = NestedImageDataset(f'{DATA_DIR}/train', transform=train_tf)
    val_ds   = NestedImageDataset(f'{DATA_DIR}/val',   transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    return train_dl, val_dl, train_ds.classes


# reshapes the FC layer to match our number of classes
def build_model(num_classes, device):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features

    # accesses the last block of layers, adjust the fully connected layers to our 
    # number of classes
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)



# many torchvision models split the model into the features (backbone) and classifier (head)
# The classifier is always able to be updated, but the backbone we want frozen at first
def set_backbone_frozen(model, frozen: bool):
    for param in model.features.parameters():
        param.requires_grad = not frozen
    for param in model.classifier.parameters():
        param.requires_grad = True



# the optimizer is how the model's weights are updated (Adam is good, but stochatistic gradient descent is also used)
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    # when the model is in training mode, it can update its weights
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    # not understanding here
    # CORRECTION: torch.set_grad_enabled(train) is a context manager. 
    # When train=True (training), PyTorch tracks every operation on tensors to build a 
    # computation graph used by loss.backward(). When train=False (validation), that tracking is 
    # disabled — no graph is built, saving memory and speeding up the forward pass.
    with torch.set_grad_enabled(train):

        # loader is what holds our images (that pytorch object)
        for imgs, labels in loader:

            # must be converted to the device to be processed correctly
            imgs, labels = imgs.to(device), labels.to(device)

            # out is the vector holding the probabilities "guesses" made by the model on the images in the loader
            # CORRECTION: out holds raw logits (one score per class), not probabilities. To get probabilities you'd apply softmax, but CrossEntropyLoss does that internally — so we pass logits directly.
            out  = model(imgs)

            # what is the criteron? its our loss function (cross-entropy) somehow it takes our guesses and makes them into loss
            # CORRECTION: CrossEntropyLoss applies softmax to the logits internally, then computes -log(probability assigned to the correct class). If the model was very confident about the wrong class, that probability is near 0, so -log(~0) = very high loss. Perfect prediction → loss near 0.
            loss = criterion(out, labels)
            if train:

                # we must zero the gradients before hand
                optimizer.zero_grad()

                # unsure
                # CORRECTION: loss.backward() runs backpropagation — it walks backward through the computation graph 
                # PyTorch built during the forward pass and uses the chain rule to compute the gradient (∂loss/∂weight)
                #  for every trainable parameter. Those gradients are then used by optimizer.step() to update the weights.
                loss.backward()

                # steps according to how our optimizer chooses to step
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def train_phase(label, model, train_dl, val_dl, epochs, lr, wd, device):
    print(f'\n── {label} ──')

    # builds our optimizer (Adam)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd,
    )

    # unsure
    # CORRECTION: CosineAnnealingLR is a learning rate scheduler. It smoothly decreases 
    # the LR from the initial value down to ~0 following a cosine curve over T_max epochs. 
    # Large steps early (fast learning) → tiny steps late (fine-tuning convergence). Much smoother than abrupt step-decay.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # our loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # training loss and accuracy
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, device, train=True)

        # validaiton set loss and accuracu
        vl_loss, vl_acc = run_epoch(model, val_dl,   criterion, None,      device, train=False)
        
        #????
        # CORRECTION: scheduler.step() tells the scheduler that one epoch has finished, so it 
        # updates the learning rate for the next epoch according to the cosine curve. Without this 
        # call the LR never changes.
        scheduler.step()
        print(f'  epoch {epoch:2d}/{epochs}  '
              f'train  loss={tr_loss:.4f}  acc={tr_acc:.3f}  '
              f'val  loss={vl_loss:.4f}  acc={vl_acc:.3f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')


def print_confusion_matrix(model, loader, classes, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    width = max(len(c) for c in classes)
    header = ' ' * (width + 2) + '  '.join(f'{c:>{width}}' for c in classes)
    print(f'\n── Confusion Matrix (rows=actual, cols=predicted) ──\n{header}')
    for i, row in enumerate(cm):
        cells = '  '.join(f'{v:{width}d}' for v in row)
        print(f'  {classes[i]:>{width}}  {cells}')


def main():

    # select the device to do our training (cude requires internet?)
    # CORRECTION: No internet needed — cuda/mps/cpu all refer to local hardware. 
    # cuda = NVIDIA GPU (uses CUDA drivers), mps = Apple Silicon GPU (Metal Performance Shaders), 
    # cpu = your processor. The code picks the fastest one available on your machine.
    device = (
        torch.device('cuda') if torch.cuda.is_available()  else
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f'device: {device}')

    # loads the data in the correct batch sizes
    train_dl, val_dl, classes = make_loaders()
    print(f'classes ({len(classes)}): {classes}')


    # use our build model function
    model = build_model(len(classes), device)

    # freeze the feature extraction layers

    set_backbone_frozen(model, frozen=True)

    # train just the head (FC layers)
    train_phase('Phase 1 — head only', model, train_dl, val_dl,
                P1_EPOCHS, P1_LR, P1_WD, device)


    # unfreeze the backbone
    set_backbone_frozen(model, frozen=False)

    # train all layers
    train_phase('Phase 2 — full fine-tune', model, train_dl, val_dl,
                P2_EPOCHS, P2_LR, P2_WD, device)

    # save our model under a full-ISO-timestamped filename, never overwriting an existing one
    # (colons are replaced with dashes so the name is filesystem-safe)
    stamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    out_path = Path(f'{stamp}-pose.pt')
    counter = 2
    while out_path.exists():
        out_path = Path(f'{stamp}-pose_{counter}.pt')
        counter += 1
    torch.save({'classes': classes, 'state_dict': model.state_dict()}, out_path)
    print(f'\nsaved {out_path}')

    print_confusion_matrix(model, val_dl, classes, device)


if __name__ == '__main__':
    main()
