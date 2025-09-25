# ===============================
# 1️⃣ Imports
# ===============================
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import pickle

# ===============================
# 2️⃣ Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 3️⃣ FastAPI App & CORS
# ===============================
app = FastAPI(title="Crop Classification API")

# Allow all origins for development. In production, replace "*" with your Flutter app domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 4️⃣ Image Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ===============================
# 5️⃣ Define CNN+Transformer Model (same as training)
# ===============================
class CNN_Transformer(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Transformer, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        self.transformer_dim = 512
        self.num_heads = 8
        self.num_layers = 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, 
            nhead=self.num_heads, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.transformer_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # B,C,H,W
        B, C, H, W = features.size()
        features = features.view(B, C, H*W).permute(0,2,1)  # B,seq,C
        features = self.transformer(features)
        features = features.mean(dim=1)
        out = self.fc(features)
        return out

# ===============================
# 6️⃣ Load Model & Classes
# ===============================
CLASSES_PATH = "app/classes.pkl"
MODEL_PATH = "app/cnn_transformer_resnet18.pth"

with open(CLASSES_PATH, "rb") as f:
    class_names = pickle.load(f)

num_classes = len(class_names)

model = CNN_Transformer(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ===============================
# 7️⃣ API Endpoint
# ===============================
@app.post("/predict")
async def predict_crop(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        predicted_class = class_names[pred.item()]

    return {"predicted_crop": predicted_class}

# ===============================
# 8️⃣ Root Endpoint
# ===============================
@app.get("/")
def root():
    return {"message": "🌱 Crop Classification API is running"}

# ===============================
# 9️⃣ Run server
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
