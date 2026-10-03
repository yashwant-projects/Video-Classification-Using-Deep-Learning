import collections
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

blank_slide_layout = prs.slide_layouts[6] 

bg_title = r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_title.png"
bg_content = r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_content.png"

def add_slide_with_bg(bg_image_path):
    slide = prs.slides.add_slide(blank_slide_layout)
    pic = slide.shapes.add_picture(bg_image_path, 0, 0, width=prs.slide_width, height=prs.slide_height)
    return slide

def add_text(slide, text, left, top, width, height, font_size, bold=False, color=RGBColor(0,0,0), alignment=PP_ALIGN.CENTER):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.clear() 
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = alignment
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = 'Times New Roman'
    return p

slide1 = add_slide_with_bg(bg_title)

add_text(slide1, "Shri Shankaracharya Institute of Professional", Inches(2), Inches(0.2), Inches(9.3), Inches(0.6), 36, True)
add_text(slide1, "Management & Technology, Raipur", Inches(2), Inches(0.8), Inches(9.3), Inches(0.6), 36, True)

add_text(slide1, "Major Project (Phase I)", Inches(3), Inches(1.8), Inches(7.33), Inches(0.5), 28, True)
add_text(slide1, "Presentation on", Inches(3), Inches(2.2), Inches(7.33), Inches(0.5), 24, True)

add_text(slide1, "Video Classification System and Intelligent Border", Inches(1.5), Inches(2.8), Inches(10.3), Inches(0.5), 32, True, RGBColor(0, 32, 96))
add_text(slide1, "Video Analytics Platform(IBVAP) Using Deep Learning", Inches(1.5), Inches(3.3), Inches(10.3), Inches(0.5), 32, True, RGBColor(0, 32, 96))

add_text(slide1, "Department of Artificial Intelligence & Machine Learning", Inches(2), Inches(4), Inches(9.33), Inches(0.5), 24, True)
add_text(slide1, "B.Tech 7th Semester Batch 2023-2027", Inches(2), Inches(4.5), Inches(9.33), Inches(0.5), 24, True)
add_text(slide1, "Session July – December 2026", Inches(2), Inches(5), Inches(9.33), Inches(0.5), 22, True)

add_text(slide1, "Guided By:", Inches(1.5), Inches(5.4), Inches(4), Inches(0.3), 22, True, alignment=PP_ALIGN.LEFT)
add_text(slide1, "Dr. Anjali Chandra", Inches(1.5), Inches(5.7), Inches(4), Inches(0.3), 22, True, alignment=PP_ALIGN.LEFT)
add_text(slide1, "Head of Department, Department of", Inches(1.5), Inches(6.0), Inches(5), Inches(0.3), 16, False, alignment=PP_ALIGN.LEFT)
add_text(slide1, "Artificial Intelligence and Machine Learning,", Inches(1.5), Inches(6.25), Inches(5), Inches(0.3), 16, False, alignment=PP_ALIGN.LEFT)
add_text(slide1, "SSIPMT Raipur", Inches(1.5), Inches(6.5), Inches(5), Inches(0.3), 16, False, alignment=PP_ALIGN.LEFT)

add_text(slide1, "Project Group Members:", Inches(8), Inches(5.4), Inches(4.5), Inches(0.3), 22, True, alignment=PP_ALIGN.LEFT)
add_text(slide1, "YASHWANT SUSHIL", Inches(8), Inches(5.7), Inches(4.5), Inches(0.3), 20, True, alignment=PP_ALIGN.LEFT)
add_text(slide1, "HARISH AGRAWAL", Inches(8), Inches(6.0), Inches(4.5), Inches(0.3), 20, True, alignment=PP_ALIGN.LEFT)
add_text(slide1, "HARSH KANOJIYA", Inches(8), Inches(6.3), Inches(4.5), Inches(0.3), 20, True, alignment=PP_ALIGN.LEFT)
add_text(slide1, "PRAVEEN SHANKAR", Inches(8), Inches(6.6), Inches(4.5), Inches(0.3), 20, True, alignment=PP_ALIGN.LEFT)

def add_content_slide(title, bullet_points, img_path=None, img_left=None, img_top=None, img_height=None):
    slide = add_slide_with_bg(bg_content)
    add_text(slide, title, Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
    
    txBox = slide.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, pt in enumerate(bullet_points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = "• " + pt
        p.font.size = Pt(28)
        p.font.name = 'Times New Roman'
        p.space_after = Pt(14)
        
    if img_path:
        slide.shapes.add_picture(img_path, img_left, img_top, height=img_height)
        
    return slide

add_content_slide("Presentation Layout :", [
    "Introduction",
    "Features",
    "Dataset Details",
    "Performance Metrics",
    "System Screenshots & Results",
    "Conclusion"
])

add_content_slide("Introduction", [
    "The project implements a Deep Learning-based video classification and surveillance system.",
    "It aims to automate complex security and monitoring tasks.",
    "Capabilities include action recognition, hybrid inference, and automated dossier generation.",
    "Features a robust live surveillance mode with an interactive virtual fence."
])

add_content_slide("Features", [
    "Deep Learning Video Classification: Classifies videos into 7 categories using PyTorch and LSTM.",
    "Hybrid Inference & Analytics: Leverages YOLOv8 and CLIP for object detection and tracking.",
    "Live Surveillance Mode: Real-time surveillance featuring a Virtual Fence.",
    "Telegram Bot Integration: Automatically sends alert messages upon breach.",
    "Automated Dossier Generation: Generates PDF reports and plots analyzing the inferences."
])

add_content_slide("Dataset Details", [
    "Trained on a meticulously curated dataset of 1050 unique videos (150 per class):",
    "Traffic: Filtered from the Traffic Video Dataset for highest quality.",
    "Wildlife: Extracted from the Animal-Kingdom Dataset via automated script.",
    "Violence: Handpicked from the Real Life Violence Situations Dataset.",
    "Education, Entertainment, Human Activities, Sports: Extracted from UCF101."
])

add_content_slide("Performance Metrics", [
    "Test Accuracy: Achieved a massive global peak testing accuracy of 99.05% on unseen holdout videos.",
    "Precision & Recall: Critical security classes like Violence, Wildlife, and Traffic achieved flawless 1.0000 scores.",
    "The model demonstrates exceptional reliability and discriminative power in real-time scenarios."
])

s7 = add_slide_with_bg(bg_content)
add_text(s7, "Video Classification System", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s7.shapes.add_picture(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790965221124.png", Inches(3.5), Inches(1.5), height=Inches(5.5))

s8 = add_slide_with_bg(bg_content)
add_text(s8, "Intelligent Border Video Analytics Platform", Inches(1.5), Inches(0.4), Inches(11), Inches(0.8), 40, True)
s8.shapes.add_picture(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790965221145.png", Inches(2.5), Inches(1.5), height=Inches(5.5))

s9 = add_slide_with_bg(bg_content)
add_text(s9, "Telegram Messaging Alert", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s9.shapes.add_picture(r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded\media_1790965221144.jpg", Inches(5), Inches(1.5), height=Inches(5.5))

pm_dir = r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\presentation_metrics"

s10 = add_slide_with_bg(bg_content)
add_text(s10, "Model Architecture", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s10.shapes.add_picture(pm_dir + r"\slide_9_model_architecture.png", Inches(1.5), Inches(1.5), height=Inches(5.5))

s11 = add_slide_with_bg(bg_content)
add_text(s11, "Results: Loss & Accuracy", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s11.shapes.add_picture(pm_dir + r"\slide_12_loss_accuracy.png", Inches(1.5), Inches(1.5), height=Inches(5.5))

s12 = add_slide_with_bg(bg_content)
add_text(s12, "Results: Sensitivity & AUC", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s12.shapes.add_picture(pm_dir + r"\slide_13_sensitivity_auc.png", Inches(1.5), Inches(1.5), height=Inches(5.5))

s13 = add_slide_with_bg(bg_content)
add_text(s13, "Results: Confusion Matrix", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s13.shapes.add_picture(pm_dir + r"\slide_14_confusion_matrix.png", Inches(1.5), Inches(1.5), height=Inches(5.5))

s14 = add_slide_with_bg(bg_content)
add_text(s14, "Results Bulletin", Inches(3.5), Inches(0.4), Inches(8), Inches(0.8), 40, True)
s14.shapes.add_picture(pm_dir + r"\slide_11_results_bulletin.png", Inches(1.5), Inches(1.5), height=Inches(5.5))


add_content_slide("Conclusion", [
    "The proposed Deep Learning model provides an efficient, accurate solution for video classification and real-time surveillance.",
    "Automation of dossier generation and telegram alerts significantly enhances rapid response and security operations.",
    "Future work may involve:",
    "  > Scaling the Virtual Fence to multi-camera setups.",
    "  > Integrating further optimization for edge deployment."
])

# ----------------- Slide 17: References -----------------
add_content_slide("References", [
    'Jocher, G. et al. "Ultralytics YOLOv8." (2023).',
    'Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML (2021).',
    'Soomro, K., Zamir, A. R., & Shah, M. "UCF101: A dataset of 101 human actions classes from videos in the wild." arXiv preprint arXiv:1212.0402 (2012).',
    'Hochreiter, S., & Schmidhuber, J. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.',
    'Goyal, K. et al. "Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding." CVPR (2022).'
])

# ----------------- Slide 18: Thank You -----------------
bg_thank_you = r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\bg_thank_you.png"
slide18 = add_slide_with_bg(bg_thank_you)
# Removed programmatic "Thank You!" since the text is already in the background image.

prs.save(r"c:\Users\yashw\Downloads\Video Classification Using Deep Learning\IBVAP_Presentation_v6.pptx")
print("Presentation generated successfully.")
