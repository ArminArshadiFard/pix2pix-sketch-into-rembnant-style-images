import kagglehub
import os
from PIL import Image
from pathlib import Path


def prepare_rembrandt_from_side_by_side(output_dir="datasets/rembrandt"):
    print("Downloading Rembrandt dataset...")
    data_dir = kagglehub.dataset_download("grafstor/rembrandt-pix2pix-dataset")

    # Find the folder with the 301 images (likely 'gd')
    image_folder = None
    for root, dirs, files in os.walk(data_dir):
        jpg_files = [f for f in files if f.endswith(".jpg")]
        if len(jpg_files) > 300:  # Found the main image folder
            image_folder = Path(root)
            break

    if image_folder is None:
        raise FileNotFoundError("Could not locate the folder with 301 images")

    print(f"Found image folder: {image_folder}")
    jpg_files = sorted([f for f in image_folder.glob("*.jpg")])
    print(f"Processing {len(jpg_files)} side-by-side images...")

    # Create output directories
    os.makedirs(f"{output_dir}/train/A", exist_ok=True)  # sketches (left)
    os.makedirs(f"{output_dir}/train/B", exist_ok=True)  # rembrandt (right)

    for i, img_path in enumerate(jpg_files):
        try:
            full_img = Image.open(img_path).convert("RGB")
            w, h = full_img.size

            # Split vertically in half
            sketch = full_img.crop((0, 0, w // 2, h))  # Left half
            rembrandt = full_img.crop((w // 2, 0, w, h))  # Right half

            # Save
            sketch.save(f"{output_dir}/train/A/{i:05d}.jpg")
            rembrandt.save(f"{output_dir}/train/B/{i:05d}.jpg")

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # Optional: create a small test set (e.g., last 30 images)
    os.makedirs(f"{output_dir}/test/A", exist_ok=True)
    os.makedirs(f"{output_dir}/test/B", exist_ok=True)

    test_start = max(0, len(jpg_files) - 30)
    for i in range(test_start, len(jpg_files)):
        os.rename(
            f"{output_dir}/train/A/{i:05d}.jpg",
            f"{output_dir}/test/A/{i - test_start:05d}.jpg"
        )
        os.rename(
            f"{output_dir}/train/B/{i:05d}.jpg",
            f"{output_dir}/test/B/{i - test_start:05d}.jpg"
        )

    print(f"✅ Done! {len(jpg_files) - 30} train pairs, 30 test pairs")


if __name__ == "__main__":
    prepare_rembrandt_from_side_by_side()