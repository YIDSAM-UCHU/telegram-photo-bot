from PIL import Image, ImageOps
import cv2
import numpy as np

from config import Config


def _detect_face(image):
    """Return the largest frontal face box (x, y, width, height), if found."""
    rgb = image.convert("RGB")
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return max(faces, key=lambda box: box[2] * box[3]) if len(faces) else None


def _photo_area_height(overlay):
    """Find the transparent portrait area above the first visible overlay row."""
    alpha = np.asarray(overlay.getchannel("A"))
    width, height = overlay.size
    for y in range(height):
        if np.count_nonzero(alpha[y] > 12) > width * 0.01:
            return max(1, y)
    return height


def _crop_for_poster(image, target_size):
    target_width, target_height = target_size
    face = _detect_face(image)
    aspect = target_width / target_height

    if face is None:
        # Keep the center of the image when no frontal face is found.
        width, height = image.size
        if width / height > aspect:
            crop_width = int(height * aspect)
            left = (width - crop_width) // 2
            return image.crop((left, 0, left + crop_width, height))
        crop_height = int(width / aspect)
        top = (height - crop_height) // 2
        return image.crop((0, top, width, top + crop_height))

    x, y, face_width, face_height = map(int, face)
    width, height = image.size
    # Leave room around the head while giving the face a prominent position.
    crop_height = min(height, max(face_height * 2, int(face_height * 2.5)))
    crop_width = min(width, int(crop_height * aspect))
    if crop_width / crop_height > aspect:
        crop_width = int(crop_height * aspect)
    else:
        crop_height = min(height, int(crop_width / aspect))

    center_x = x + face_width / 2
    center_y = y + face_height * 0.42
    left = int(center_x - crop_width / 2)
    top = int(center_y - crop_height * 0.42)
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))
    return image.crop((left, top, left + crop_width, top + crop_height))


def edit_image(input_path, output_path):
    photo = ImageOps.exif_transpose(Image.open(input_path)).convert("RGBA")
    overlay = Image.open(Config.OVERLAY_PATH).convert("RGBA")

    canvas_size = overlay.size
    photo_height = _photo_area_height(overlay)
    photo_area = (canvas_size[0], photo_height)
    photo = _crop_for_poster(photo, photo_area).resize(photo_area, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    canvas.alpha_composite(photo, (0, 0))
    canvas.alpha_composite(overlay)
    canvas.convert("RGB").save(output_path, "JPEG", quality=95)
