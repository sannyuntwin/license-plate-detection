import easyocr
import cv2

class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read_text(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if height < 64:
            scale = max(2, round(64 / height))
            gray = cv2.resize(
                gray,
                (width * scale, height * scale),
                interpolation=cv2.INTER_CUBIC,
            )
            height, width = gray.shape

        result = self.reader.recognize(
            gray,
            horizontal_list=[[0, width, 0, height]],
            free_list=[],
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        return " ".join(result) if result else ""
