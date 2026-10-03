import pymupdf
import re

doc = pymupdf.open(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790960783980.pdf")
last_page_idx = len(doc) - 1
page = doc[last_page_idx]

# Clean text from the last page
page.clean_contents()
for xref in page.get_contents():
    stream = doc.xref_stream(xref)
    if stream:
        new_stream = re.sub(b'BT.*?ET', b'', stream, flags=re.DOTALL)
        doc.update_stream(xref, new_stream)

# Save as bg_thank_you.png
pix_thank_you = page.get_pixmap(dpi=150)
pix_thank_you.save(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_thank_you.png")
print("Saved bg_thank_you.png")
doc.close()
