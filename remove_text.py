import pymupdf
import re

doc = pymupdf.open(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790960783980.pdf")

for page in doc:
    # clean contents to ensure stream structure is normalized somewhat
    page.clean_contents()
    for xref in page.get_contents():
        stream = doc.xref_stream(xref)
        if stream:
            # Remove text blocks (BT to ET)
            new_stream = re.sub(b'BT.*?ET', b'', stream, flags=re.DOTALL)
            doc.update_stream(xref, new_stream)

doc.save(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\blank_bg.pdf")
print("Saved blank_bg.pdf")

# Now let's convert the first two pages to images to use as backgrounds
blank_doc = pymupdf.open(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\blank_bg.pdf")
pix0 = blank_doc[0].get_pixmap(dpi=150)
pix0.save(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_title.png")

pix1 = blank_doc[1].get_pixmap(dpi=150)
pix1.save(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_content.png")

print("Saved background images.")
