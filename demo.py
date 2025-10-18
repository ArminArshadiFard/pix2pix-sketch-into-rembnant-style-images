import gradio as gr
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from model import Generator

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading trained generator...")
model = Generator(in_channels=3, features=64)
model.load_state_dict(
    torch.load("./checkpoints/best_gen.pth", map_location=device, weights_only=True)
)
model = model.to(device).eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def sharpen_image(img_tensor):
    # Convert tensor → PIL
    img = transforms.ToPILImage()(img_tensor * 0.5 + 0.5)
    # Apply sharpening
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    # Back to tensor
    return transforms.ToTensor()(sharpened) * 2 - 1

# Custom dataset for paired A/B images

def translate_dermoscopic(input_img):
    """Convert dermoscopic image to clinical-style"""
    # Preprocess
    img_tensor = transform(input_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        fake_clinical = model(img_tensor)

    # Postprocess: [-1, 1] → [0, 1]
    output_img = (fake_clinical.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
    output_img = transforms.ToPILImage()(output_img)

    return output_img


# Build interface
demo = gr.Interface(
    fn=translate_dermoscopic,
    inputs=gr.Image(type="pil", label="Dermoscopic Image"),
    outputs=gr.Image(type="pil", label="Synthetic Clinical Image"),
    title="🩺 Derm2Real: Dermoscopic → Clinical Skin Lesion Translator",
    description=(
        "Upload a dermoscopic skin lesion image to generate a realistic clinical-style photo. "
        "Helps bridge the gap between specialist imaging and patient-friendly visuals."
    ),
    examples=[
        ["examples/derm_1.jpg"],
        ["examples/derm_2.jpg"]
    ]
)

if __name__ == "__main__":
    demo.launch()