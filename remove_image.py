import pymupdf
import re

doc = pymupdf.open(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790960783980.pdf")

# Page 0
page = doc[0]
page.clean_contents()
for xref in page.get_contents():
    stream = doc.xref_stream(xref)
    if stream:
        # Remove text blocks
        new_stream = re.sub(b'BT.*?ET', b'', stream, flags=re.DOTALL)
        # Remove Image21
        new_stream = re.sub(b'/Image21 Do', b'', new_stream)
        doc.update_stream(xref, new_stream)

pix0 = doc[0].get_pixmap(dpi=150)
pix0.save(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_title.png")

print("Saved clean bg_title.png")
doc.close()
