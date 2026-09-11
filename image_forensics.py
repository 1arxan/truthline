from PIL import Image, ImageChops, ImageEnhance
import os

print("IMAGE FORENSICS SCRIPT STARTED")

def generate_ela_image(image_path, quality=90):
    print(f"Opening image: {image_path}")

    if not os.path.exists(image_path):
        print(f"ERROR: File not found at {image_path}")
        return None

    original = Image.open(image_path).convert("RGB")
    print("Image opened successfully")

    resaved_path = "temp_resaved.jpg"
    original.save(resaved_path, "JPEG", quality=quality)
    resaved = Image.open(resaved_path)
    print("Resaved copy created")

    diff = ImageChops.difference(original, resaved)

    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    print(f"Max pixel difference found: {max_diff}")

    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    os.remove(resaved_path)
    print("Temp file cleaned up")

    return diff

if __name__ == "__main__":
    try:
        image_path = "test_images/WIN_20260912_00_20_23_Pro.jpg"
        ela_image = generate_ela_image(image_path)

        if ela_image is not None:
            output_path = "test_images/ela_output.png"
            ela_image.save(output_path)
            print(f"SUCCESS: ELA image saved to {output_path}")
        else:
            print("Skipped saving because image failed to load.")

    except Exception as e:
        print("AN ERROR OCCURRED:")
        print(e)

print("IMAGE FORENSICS SCRIPT FINISHED")