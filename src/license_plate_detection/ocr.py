import easyocr
import cv2

class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read_text(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        result = self.reader.readtext(gray, detail=0)

        if len(result) > 0:
            return " ".join(result)
        return ""
