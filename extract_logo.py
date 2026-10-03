import pymupdf

doc = pymupdf.open(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790960783980.pdf")
logo = doc.extract_image(9)
with open(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\college_logo.png", "wb") as f:
    f.write(logo["image"])

# Let's extract the bone fracture image just in case
img2 = doc.extract_image(21)
with open(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bone_fracture.png", "wb") as f:
    f.write(img2["image"])

print("Extracted images.")
doc.close()
