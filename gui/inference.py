import torch
import torch.nn.functional as F
from models.mlp import DigitClassifier

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = DigitClassifier().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/adam.pth",
        map_location = DEVICE 
    )
)
model.eval()


def predict(tensor):
    with torch.no_grad():
        logits = model(tensor)

        probabilities = F.softmax(logits, dim = 1)
        probabilities = probabilities.squeeze().cpu().numpy()
        predicted_digit = probabilities.argmax()
    return predicted_digit, probabilities
