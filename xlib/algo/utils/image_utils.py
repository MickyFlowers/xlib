import io

import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image


def pil_image_to_np_buffer(img_pil: Image.Image, format: str = "PNG") -> np.ndarray:
    with io.BytesIO() as output:
        img_pil.save(output, format="PNG")
        png_data = output.getvalue()
    return np.frombuffer(png_data, dtype="uint8")


def np_buffer_to_pil_image(img_bytes: np.ndarray) -> Image.Image:
    with io.BytesIO(img_bytes.tobytes()) as input:
        img_pil = Image.open(input).convert("RGB")
    return img_pil


def compressed_msg_to_bytes(msg):
    return bytes(msg.data)


def compressed_msg_to_pil(msg):
    return Image.open(io.BytesIO(msg.data))


def bytes_to_cv2_image(img_bytes: bytes) -> np.ndarray:
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img_pil)

def save_video(output_path, frames, fps=30, codec="libx264"):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec=codec)
