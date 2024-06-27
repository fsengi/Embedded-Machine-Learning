'''
this implementation is a first try to run a basic custom trained example from the lecture on a raspberry pi
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2


class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _make_layers(self, dropout_p):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(end_dim=-1),

            nn.Linear(in_features=512, out_features=4096),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10)          
        ]
        return nn.ModuleList(layers)

    def forward(self, x):       
        for mod in self.layers:
            x = mod(x)
        output = F.log_softmax(x, dim=1)
        return output

filepath = "Embedded-Machine-Learning/ex04/4_1/trained_VGG11_L2-1e-05.pt"
model = VGG11()
model.state_dict(torch.load(filepath))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class (adjust as needed based on your model)
    _, predicted_class = output.max(1)
    print(f"Predicted class: {predicted_class.item()}")

    # Display the frame (optional)
    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
