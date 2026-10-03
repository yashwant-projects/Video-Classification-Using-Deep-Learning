import fitz
import json

doc = fitz.open(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790960783980.pdf")
print("Page 0 images:")
for img in doc[0].get_images(full=True):
    print(img)
print("Page 1 images:")
for img in doc[1].get_images(full=True):
    print(img)

doc.close()
