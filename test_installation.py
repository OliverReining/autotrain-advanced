import torch
import torchvision
import torchaudio
import numpy as np

# Überprüfen der installierten Versionen
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"NumPy Version: {np.__version__}")

# Teste CUDA-Verfügbarkeit und einfache CUDA-Berechnung
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print("CUDA is available. Tensor on CUDA:", x)
else:
    print("CUDA is not available.")

# Teste ein einfaches Modell mit torchvision
model = torchvision.models.resnet18(pretrained=False)
print("Torchvision model created:", model)
