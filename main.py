from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
import fitz  # PyMuPDF
from io import BytesIO
import logging
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://38.242.137.214:8888"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def extract_image_or_text(file_bytes: bytes, content_type: str):
    """
    Extracts either an image or text from a given file.
    If an image is found in a PDF, it extracts the image.
    Otherwise, it extracts text with font size.
    """
    try:
        if content_type == "application/pdf":
            doc = fitz.open(stream=BytesIO(file_bytes))
            if len(doc) == 0:
                raise ValueError("Empty PDF document")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                images = page.get_images(full=True)

                if images:
                    # Extract first image found in the document
                    img_index = images[0][0]  # First image index
                    base_image = doc.extract_image(img_index)
                    image_bytes = base_image["image"]
                    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return "image", image

            # If no images found, extract text with font size
            text_data = []
            for page_layout in extract_pages(BytesIO(file_bytes)):
                for element in page_layout:
                    if isinstance(element, (LTTextBox, LTTextLine)):
                        for text_line in element:
                            if isinstance(text_line, LTTextLine):
                                for char in text_line:
                                    if isinstance(char, LTChar) and char.get_text().strip():
                                        text_data.append({
                                            "letter": char.get_text(),
                                            "x": char.bbox[0],
                                            "y": char.bbox[1],
                                            "w": char.bbox[2] - char.bbox[0],
                                            "h": char.bbox[3] - char.bbox[1],
                                            "font_size": char.size
                                        })
            
            if not text_data:
                raise ValueError("No images or text found in the PDF")

            return "text", text_data

        if content_type in ["image/jpeg", "image/png"]:
            image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return "image", image

        raise ValueError("Unsupported file format")
    except Exception as e:
        logging.error(f"File extraction error: {str(e)}")
        raise

def scale_text_data(letter_boxes, target_width, target_height):
    """
    Scales letters extracted from text data.
    The letter with the maximum font size is set to 100%, and others are scaled accordingly.
    """
    if not letter_boxes:
        return []

    max_font_size = max(letter["font_size"] for letter in letter_boxes)

    # Calculate total width of text
    total_width = sum(letter["w"] for letter in letter_boxes)
    width_ratio = target_width / total_width if total_width > 0 else 1
    height_ratio = target_height / max_font_size if max_font_size > 0 else 1

    letters = []
    for letter in letter_boxes:
        letters.append({
            "letter_number": letter["letter"].upper(),
            "scaled_width": int(letter["w"] * width_ratio),
            "scaled_height": int(letter["font_size"] * height_ratio)
        })

    return letters

def process_image(image: np.ndarray, target_width: int, target_height: int) -> list:
    """
    Processes an image to detect individual letters and scale them accordingly.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        letter_boxes = []
        min_area = 500
        max_aspect_ratio = 3.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)

            if area > min_area and aspect_ratio < max_aspect_ratio:
                letter_boxes.append((x, y, w, h))

        letter_boxes.sort(key=lambda b: (b[0], b[1]))

        scaled_widths = [int(w * (target_width / sum(w for _, _, w, _ in letter_boxes))) for _, _, w, _ in letter_boxes]
        heights = [h for _, _, _, h in letter_boxes]

        letters = []
        for idx, (w, h) in enumerate(zip(scaled_widths, heights)):
            letters.append({
                "letter_number": idx + 1,
                "scaled_width": w,
                "scaled_height": int((target_height / max(heights)) * h)
            })

        return letters
    except Exception as e:
        logging.error(f"Image processing error: {str(e)}")
        raise

@app.post("/detect-letters/")
async def detect_letters(
    file: UploadFile = File(...),
    target_width: int = Form(200),
    target_height: int = Form(100)
):
    try:
        file_bytes = await file.read()
        data_type, data = extract_image_or_text(file_bytes, file.content_type)

        if data_type == "image":
            letters = process_image(data, target_width, target_height)
        else:
            letters = scale_text_data(data, target_width, target_height)

        return {"letters": letters}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, detail="Internal server error")
    finally:
        await file.close()

@app.get("/")
async def startup():
    return {"message": "API is Running"}

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
