from efficientnet_pytorch import EfficientNet
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from img_utils import add_padding

clsNexdate = [
    ('apple', 30),
    ('banana', 7),
    ('beetroot', 14),
    ('bell pepper', 10),
    ('cabbage', 14),
    ('capsicum', 10),
    ('carrot', 21),
    ('cauliflower', 10),
    ('chilli pepper', 10),
    ('corn', 7),
    ('cucumber', 7),
    ('eggplant', 7),
    ('garlic', 60),
    ('ginger', 30),
    ('grapes', 7),
    ('jalepeno', 7),
    ('kiwi', 14),
    ('lemon', 30),
    ('lettuce', 5),
    ('mango', 7),
    ('onion', 60),
    ('orange', 30),
    ('paprika', 10),
    ('pear', 14),
    ('peas', 7),
    ('pineapple', 7),
    ('pomegranate', 30),
    ('potato', 30),
    ('raddish', 10),
    ('soy beans', 7),
    ('spinach', 5),
    ('sweetcorn', 7),
    ('sweetpotato', 30),
    ('tomato', 7),
    ('turnip', 14),
    ('watermelon', 7)
]

model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(clsNexdate))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(torch.device(device))

model.load_state_dict(torch.load("models/classify.pth", map_location=device))
model.eval()

def classify(img):
    img = add_padding(img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return clsNexdate[predicted.item()]
