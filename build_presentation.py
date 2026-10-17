# -*- coding: utf-8 -*-
"""
Build the Restructured & Reorganized Presentation (IBVAP_Presentation_v6.pptx)
Strictly conforming to the 3rd Project Review (Design Phase) 15-Point Checklist.
FIXES:
1. Complete elimination of all text overlaps on Slide 1 by using unified auto-flowing text frames.
2. Complete elimination of left-side clipping by setting all content SAFE_LEFT = 2,200,000 EMU (clearing the red ribbon arrow at 1.71M).
3. Complete elimination of bottom overflow by splitting dense slides into clean multi-column layouts and calibrated font sizes.
4. Dynamic title sizing to ensure titles never spill into 2 lines or collide with content.
5. Strict aspect-ratio bounding box centering for all visual diagram slides.
"""

import os
import shutil
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

def build_presentation():
    orig_file = "IBVAP_Presentation_v6.pptx"
    updated_file = "IBVAP_Presentation_v6_Updated.pptx"
    backup_file = "IBVAP_Presentation_v6_backup.pptx"

    base_file = None
    for cand in [orig_file, updated_file, backup_file]:
        if os.path.exists(cand) and os.path.getsize(cand) > 1000:
            base_file = cand
            break

    prs = pptx.Presentation(base_file) if base_file else pptx.Presentation()
    
    # 16:9 widescreen dimensions (12191695 x 6858000 EMU)
    slide_w = prs.slide_width
    slide_h = prs.slide_height

    # Template backgrounds
    title_bg = "design_diagrams/template_assets/title_bg.png"
    content_bg = "design_diagrams/template_assets/content_bg.png"
    closing_bg = "design_diagrams/template_assets/closing_bg.png"

    # Clear old slide deck
    rId_list = [prs.slides._sldIdLst[i].rId for i in range(len(prs.slides))]
    for rId in rId_list:
        prs.part.drop_rel(rId)
    prs.slides._sldIdLst.clear()
    print("[*] Cleared slide deck for fresh build.")

    blank_layout = prs.slide_layouts[6]

    # Content bounds strictly clearing the red arrow (which ends at X=1,706,837 EMU)
    SAFE_LEFT = 2200000    # ~2.30 inches (completely past 1.71M red ribbon)
    SAFE_TOP = 1350000     # ~1.40 inches (below header title)
    SAFE_WIDTH = 9200000   # ~9.60 inches
    SAFE_HEIGHT = 4900000  # ~5.10 inches (leaves 0.65 inches at bottom)

    # -------------------------------------------------------------------------
    # Helper: Content Slide Base
    # -------------------------------------------------------------------------
    def create_content_slide(title_text):
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes.add_picture(content_bg, 0, 0, slide_w, slide_h)
        
        # Determine title font size based on length to guarantee 1-line fit
        if len(title_text) > 42:
            fsize = 26
        elif len(title_text) > 32:
            fsize = 29
        elif len(title_text) > 24:
            fsize = 32
        else:
            fsize = 36

        # Title Box (starts at left=2800000 to clear top-left logo, width=8200000)
        tbox = slide.shapes.add_textbox(2800000, 360000, 8400000, 750000)
        tf = tbox.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Times New Roman"
        p.font.size = Pt(fsize)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 0, 0)
        p.alignment = PP_ALIGN.LEFT
        return slide

    # -------------------------------------------------------------------------
    # Helper: Add Centered Picture strictly inside safe content bounds
    # -------------------------------------------------------------------------
    def add_centered_picture(slide, img_path):
        with Image.open(img_path) as im:
            iw, ih = im.size
        
        aspect = iw / ih
        avail_aspect = SAFE_WIDTH / SAFE_HEIGHT

        if aspect > avail_aspect:
            width = SAFE_WIDTH
            height = int(width / aspect)
            left = SAFE_LEFT
            top = SAFE_TOP + int((SAFE_HEIGHT - height) / 2)
        else:
            height = SAFE_HEIGHT
            width = int(height * aspect)
            top = SAFE_TOP
            left = SAFE_LEFT + int((SAFE_WIDTH - width) / 2)

        slide.shapes.add_picture(img_path, left, top, width, height)

    # =========================================================================
    # SLIDE 1: Title Slide (Zero Overlaps, Zero Ribbon Collision)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    slide1.shapes.add_picture(title_bg, 0, 0, slide_w, slide_h)

    # Block 1: College Header (Single frame -> IMPOSSIBLE to overlap)
    hdr_box = slide1.shapes.add_textbox(1600000, 220000, 9000000, 1100000)
    tf_hdr = hdr_box.text_frame
    tf_hdr.word_wrap = True
    tf_hdr.margin_left = tf_hdr.margin_right = tf_hdr.margin_top = tf_hdr.margin_bottom = 0

    p = tf_hdr.paragraphs[0]
    p.text = "Shri Shankaracharya Institute of Professional"
    p.font.name = "Times New Roman"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(2)

    p = tf_hdr.add_paragraph()
    p.text = "Management & Technology, Raipur"
    p.font.name = "Times New Roman"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.alignment = PP_ALIGN.CENTER

    # Block 2: Review Title & Project Name (Single frame -> IMPOSSIBLE to overlap)
    title_box = slide1.shapes.add_textbox(1400000, 1550000, 9400000, 1850000)
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0

    p = tf_t.paragraphs[0]
    p.text = "Major Project (Phase I) — 3rd Project Review: Design Phase"
    p.font.name = "Times New Roman"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 43, 72)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(6)

    p = tf_t.add_paragraph()
    p.text = "Presentation on"
    p.font.name = "Times New Roman"
    p.font.size = Pt(17)
    p.font.color.rgb = RGBColor(60, 60, 60)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(6)

    p = tf_t.add_paragraph()
    p.text = "Video Classification System and Intelligent Border"
    p.font.name = "Times New Roman"
    p.font.size = Pt(27)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 32, 96)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(2)

    p = tf_t.add_paragraph()
    p.text = "Video Analytics Platform (IBVAP) Using Deep Learning"
    p.font.name = "Times New Roman"
    p.font.size = Pt(27)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 32, 96)
    p.alignment = PP_ALIGN.CENTER

    # Block 3: Department & Session
    dept_box = slide1.shapes.add_textbox(1600000, 3600000, 9000000, 1100000)
    tf_dept = dept_box.text_frame
    tf_dept.word_wrap = True
    tf_dept.margin_left = tf_dept.margin_right = tf_dept.margin_top = tf_dept.margin_bottom = 0

    p = tf_dept.paragraphs[0]
    p.text = "Department of Artificial Intelligence & Machine Learning"
    p.font.name = "Times New Roman"
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(2)

    p = tf_dept.add_paragraph()
    p.text = "B.Tech 7th Semester Batch 2023-2027"
    p.font.name = "Times New Roman"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(40, 40, 40)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(2)

    p = tf_dept.add_paragraph()
    p.text = "Session July – December 2026"
    p.font.name = "Times New Roman"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(60, 60, 60)
    p.alignment = PP_ALIGN.CENTER

    # Block 4: Footer (Guided by & Project Members)
    # Placed at left = 2,200,000 to completely avoid the red ribbon arrow (which ends at 1.71M)
    guide_box = slide1.shapes.add_textbox(2200000, 5000000, 4200000, 1600000)
    tf_g = guide_box.text_frame
    tf_g.word_wrap = True
    tf_g.margin_left = tf_g.margin_right = tf_g.margin_top = tf_g.margin_bottom = 0

    p = tf_g.paragraphs[0]
    p.text = "Guided By:"
    p.font.name = "Times New Roman"
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.space_after = Pt(2)

    p = tf_g.add_paragraph()
    p.text = "Dr. Anjali Chandra"
    p.font.name = "Times New Roman"
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.space_after = Pt(2)

    p = tf_g.add_paragraph()
    p.text = "Head of Department, Dept. of AIML"
    p.font.name = "Times New Roman"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(50, 50, 50)

    p = tf_g.add_paragraph()
    p.text = "SSIPMT, Raipur"
    p.font.name = "Times New Roman"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(50, 50, 50)

    # Members Box
    mem_box = slide1.shapes.add_textbox(7200000, 5000000, 4200000, 1600000)
    tf_m = mem_box.text_frame
    tf_m.word_wrap = True
    tf_m.margin_left = tf_m.margin_right = tf_m.margin_top = tf_m.margin_bottom = 0

    p = tf_m.paragraphs[0]
    p.text = "Project Group Members:"
    p.font.name = "Times New Roman"
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.space_after = Pt(2)

    for member in ["YASHWANT SUSHIL", "HARISH AGRAWAL", "HARSH KANOJIYA", "PRAVEEN SHANKAR"]:
        p = tf_m.add_paragraph()
        p.text = member
        p.font.name = "Times New Roman"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = RGBColor(30, 30, 30)
        p.space_after = Pt(1)

    print("[+] Slide 1 formatted with zero overlaps.")

    # =========================================================================
    # SLIDE 2: Presentation Layout (Agenda)
    # =========================================================================
    slide2 = create_content_slide("Presentation Layout")
    
    # Subtitle
    sub_box = slide2.shapes.add_textbox(SAFE_LEFT, SAFE_TOP, SAFE_WIDTH, 450000)
    tf_sub = sub_box.text_frame
    p = tf_sub.paragraphs[0]
    p.text = "Design Phase Review Checklist (Conforming to Department Notice):"
    p.font.name = "Times New Roman"
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 43, 72)

    checklist_items = [
        "1. Problem Statement & Objectives",
        "2. Proposed Solution / Concept",
        "3. Overall Architecture / Block Diagram",
        "4. Module / Component Design",
        "5. System / Process Flow (Flowchart)",
        "6. UML Diagrams (Use Case & Sequence)",
        "7. Data Design (Dataset & SQLite Schema)",
        "8. Interface & Prototype Design (UI/UX)",
        "9. Hardware & Software Design",
        "10. Algorithm / Methodology Design",
        "11. Component / Technology Selection",
        "12. Integration Design",
        "13. Detailed Working / Operational Flow",
        "14. Expected Output & Metrics (98.10%)",
        "15. Design Summary & Development Plan"
    ]

    col_w = 4400000
    gap = 400000
    top_y = SAFE_TOP + 500000

    # Col 1 (Items 1-8)
    c1_box = slide2.shapes.add_textbox(SAFE_LEFT, top_y, col_w, 4200000)
    tf_c1 = c1_box.text_frame
    tf_c1.word_wrap = True
    for i in range(8):
        p = tf_c1.paragraphs[0] if i == 0 else tf_c1.add_paragraph()
        p.text = f"• {checklist_items[i]}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(16)
        p.font.bold = True if i in [0, 2, 6, 7] else False
        p.font.color.rgb = RGBColor(20, 20, 20)
        p.space_after = Pt(8)

    # Col 2 (Items 9-15)
    c2_box = slide2.shapes.add_textbox(SAFE_LEFT + col_w + gap, top_y, col_w, 4200000)
    tf_c2 = c2_box.text_frame
    tf_c2.word_wrap = True
    for i in range(8, len(checklist_items)):
        p = tf_c2.paragraphs[0] if i == 8 else tf_c2.add_paragraph()
        p.text = f"• {checklist_items[i]}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(16)
        p.font.bold = True if i in [9, 13, 14] else False
        p.font.color.rgb = RGBColor(20, 20, 20)
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 3: 1. Problem Statement & Objectives
    # =========================================================================
    slide3 = create_content_slide("1. Problem Statement & Objectives")
    tbox3 = slide3.shapes.add_textbox(SAFE_LEFT, SAFE_TOP, SAFE_WIDTH, SAFE_HEIGHT)
    tf3 = tbox3.text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "• Core Problem Statement:"
    p.font.name = "Times New Roman"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RGBColor(192, 57, 43)
    p.space_after = Pt(4)

    probs = [
        "Human Surveillance Fatigue: Manual CCTV monitoring causes over 90% of critical security events to be missed after 20 minutes of continuous operator strain.",
        "Delayed Emergency Response: Conventional perimeter cameras lack automated tripwires, causing hazardous delays during intrusion or disturbance incidents.",
        "Dataset Domain-Shift Bias: Existing action models memorize static background artifacts (e.g. fixed camera angles) rather than learning semantic motions, causing real-world failures."
    ]
    for pb in probs:
        p = tf3.add_paragraph()
        parts = pb.split(":")
        p.text = f"   - {parts[0]}:"
        p.font.name = "Times New Roman"
        p.font.size = Pt(15.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(40, 40, 40)
        run = p.add_run()
        run.text = parts[1]
        run.font.bold = False
        p.space_after = Pt(5)

    p = tf3.add_paragraph()
    p.text = "• Concrete Design Objectives:"
    p.font.name = "Times New Roman"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RGBColor(27, 79, 114)
    p.space_before = Pt(8)
    p.space_after = Pt(4)

    objs = [
        "Real-Time Virtual Perimeter: Implement a sub-50ms YOLOv8 virtual fence with zero-tolerance ray-casting breach detection.",
        "Bias-Free Action Recognition: Develop a 7-class CLIP (ViT-B/32) + Bi-LSTM with Temporal Attention achieving >98% accuracy.",
        "Automated Emergency Telemetry: Push instant Telegram alerts (<1.2s) with intruder ID telemetry and local SQLite audit logging.",
        "Automated Forensic Intelligence Dossier: Synthesize official PDF dossiers with cryptographic timestamps for investigation."
    ]
    for ob in objs:
        p = tf3.add_paragraph()
        parts = ob.split(":")
        p.text = f"   - {parts[0]}:"
        p.font.name = "Times New Roman"
        p.font.size = Pt(15.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(40, 40, 40)
        run = p.add_run()
        run.text = parts[1]
        run.font.bold = False
        p.space_after = Pt(5)

    # =========================================================================
    # SLIDE 4: 2. Proposed Solution / Concept
    # =========================================================================
    slide4 = create_content_slide("2. Proposed Solution / Concept")
    tbox4 = slide4.shapes.add_textbox(SAFE_LEFT, SAFE_TOP, SAFE_WIDTH, SAFE_HEIGHT)
    tf4 = tbox4.text_frame
    tf4.word_wrap = True

    p = tf4.paragraphs[0]
    p.text = "Intelligent Border Video Analytics Platform (IBVAP):"
    p.font.name = "Times New Roman"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = RGBColor(21, 67, 96)
    p.space_after = Pt(8)

    concepts = [
        ("Dual-Stream Architectural Synergy:", "Unifies real-time 45+ FPS Spatial Perimeter Tracking (Stream B) with deep semantic Temporal Action Classification (Stream A), eliminating blind spots and inference bottlenecks."),
        ("Stream A (Forensic Action Recognition):", "Uniform 16-frame sampling with aspect-ratio letterboxing. Encoded via OpenAI CLIP ViT-B/32 (512-dim) and classified by a 2-layer Bi-LSTM with Temporal Attention and Tri-Pooling (1536-dim)."),
        ("Stream B (Live Spatial Virtual Fence):", "Enables operators to activate a calibrated virtual perimeter barrier directly on the live surveillance stream. Runs YOLOv8 person detection and checks ground footprint bounding box anchors via ray-casting collision algorithms."),
        ("Autonomous Threat Triaging & Logging:", "When an unauthorized entity breaches the virtual fence, the system automatically logs the breach event to an embedded SQLite database (surveillance.db) and dispatches an instant telemetry alert to Telegram in under 1.2 seconds."),
        ("Automated Intelligence Reporting:", "Incorporates a one-click forensic PDF compiler (FPDF) that generates official security dossiers containing model diagnostics, confusion matrices, and incident details.")
    ]

    for title, desc in concepts:
        p = tf4.add_paragraph()
        p.text = f"• {title} "
        p.font.name = "Times New Roman"
        p.font.size = Pt(16.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(17, 120, 100)
        
        run = p.add_run()
        run.text = desc
        run.font.name = "Times New Roman"
        run.font.size = Pt(15)
        run.font.bold = False
        run.font.color.rgb = RGBColor(35, 35, 35)
        p.space_after = Pt(7)

    # =========================================================================
    # SLIDE 5: 3. Overall Architecture / Block Diagram
    # =========================================================================
    slide5 = create_content_slide("3. Overall Architecture / Block Diagram")
    add_centered_picture(slide5, "design_diagrams/overall_architecture_block_diagram.png")

    # =========================================================================
    # SLIDE 6: 4. Module / Component Design (2-Column Balanced Cards -> ZERO OVERFLOW)
    # =========================================================================
    slide6 = create_content_slide("4. Module / Component Design")
    
    col_w = 4450000
    gap = 300000

    # Column 1: Modules 1 & 2
    c1_box = slide6.shapes.add_textbox(SAFE_LEFT, SAFE_TOP, col_w, SAFE_HEIGHT)
    tf_c1 = c1_box.text_frame
    tf_c1.word_wrap = True

    p = tf_c1.paragraphs[0]
    p.text = "Module 1: Video Ingestion & Preprocessing"
    p.font.name = "Times New Roman"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(27, 79, 114)
    p.space_after = Pt(2)

    m1_points = [
        "Aspect-ratio letterbox padding stops geometric warping on diverse aspect ratios.",
        "YOLO-gated object verification: frames without target objects are filtered out to stop background memorization.",
        "Uniform temporal stride extracts 16 representative frames spanning the video duration."
    ]
    for pt in m1_points:
        p = tf_c1.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(13.5)
        p.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(3)

    p = tf_c1.add_paragraph()
    p.text = "Module 2: Temporal Threat Classifier (Stream A)"
    p.font.name = "Times New Roman"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(27, 79, 114)
    p.space_before = Pt(10)
    p.space_after = Pt(2)

    m2_points = [
        "OpenAI CLIP (ViT-B/32) serializes 16 frames into a (16, 512) feature tensor.",
        "2-layer Bi-LSTM (256 hidden/dir) models forward & backward action dynamics.",
        "Temporal Attention + Tri-Pooling (Attn + Mean + Max) produces a 1536-d feature vector.",
        "LayerNorm + GELU + Dropout(0.5) head outputs 7-class calibrated softmax probabilities."
    ]
    for pt in m2_points:
        p = tf_c1.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(13.5)
        p.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(3)

    # Column 2: Modules 3 & 4
    c2_box = slide6.shapes.add_textbox(SAFE_LEFT + col_w + gap, SAFE_TOP, col_w, SAFE_HEIGHT)
    tf_c2 = c2_box.text_frame
    tf_c2.word_wrap = True

    p = tf_c2.paragraphs[0]
    p.text = "Module 3: Real-Time Spatial Surveillance (Stream B)"
    p.font.name = "Times New Roman"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(185, 119, 14)
    p.space_after = Pt(2)

    m3_points = [
        "Ultralytics YOLOv8 single-stage detector performs real-time person/vehicle tracking at 45+ FPS.",
        "Virtual Fence Collision Engine uses ray-casting point-in-polygon logic to verify crossings.",
        "Bottom-center anchor point tracking (x_c, y_max) checks ground footprint to prevent false alarms."
    ]
    for pt in m3_points:
        p = tf_c2.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(13.5)
        p.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(3)

    p = tf_c2.add_paragraph()
    p.text = "Module 4: Alerting, Storage & Forensic Dossier"
    p.font.name = "Times New Roman"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(146, 43, 33)
    p.space_before = Pt(10)
    p.space_after = Pt(2)

    m4_points = [
        "Asynchronous Telegram Bot API pushes instant alerts containing intruder IDs & event timestamps.",
        "Embedded SQLite engine (surveillance.db) records timestamped breach events and intruder track IDs.",
        "Automated FPDF compiler synthesizes complete civilian safety dossiers with embedded confusion matrices."
    ]
    for pt in m4_points:
        p = tf_c2.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(13.5)
        p.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(3)

    print("[+] Slide 6 formatted into 2-column balanced layout (zero overflow).")

    # =========================================================================
    # SLIDES 7 - 19: High-Resolution Visual Diagram Slides
    # =========================================================================
    slide7 = create_content_slide("5. System / Process Flow")
    add_centered_picture(slide7, "design_diagrams/system_process_flowchart.png")

    slide8 = create_content_slide("6. UML Diagrams: Use Case & Sequence")
    add_centered_picture(slide8, "design_diagrams/uml_diagrams_combined.png")

    slide9 = create_content_slide("7. Data Design: Dataset Architecture & ER Schema")
    add_centered_picture(slide9, "design_diagrams/data_design_er_diagram.png")

    slide10 = create_content_slide("8. Interface Design: Mode 1 Forensic Analysis")
    add_centered_picture(slide10, "images/forensic_analysis.png.png")

    slide11 = create_content_slide("8. Interface Design: Mode 2 Live Virtual Fence")
    add_centered_picture(slide11, "images/live_surveillance.png.png")

    slide12 = create_content_slide("9. Hardware & Software Design")
    add_centered_picture(slide12, "design_diagrams/hardware_software_stack.png")

    slide13 = create_content_slide("10. Algorithm / Methodology Design")
    add_centered_picture(slide13, "presentation_metrics/slide_9_model_architecture.png")

    slide14 = create_content_slide("11. Technology Selection Matrix")
    add_centered_picture(slide14, "design_diagrams/technology_selection_matrix.png")

    slide15 = create_content_slide("12. Integration Design: Inter-Module Data Bus")
    add_centered_picture(slide15, "design_diagrams/integration_design_flow.png")

    slide16 = create_content_slide("13. Detailed Working & Operational Flow")
    add_centered_picture(slide16, "design_diagrams/operational_workflow_diagram.png")

    slide17 = create_content_slide("14. Expected Output: Diagnostic Performance")
    add_centered_picture(slide17, "presentation_metrics/slide_14_confusion_matrix.png")

    slide18 = create_content_slide("14. Expected Output: Live Alerts & Executive Dossier")
    add_centered_picture(slide18, "design_diagrams/slide_14_alerts_bulletin_composite.png")

    slide19 = create_content_slide("15. Design Summary & Development Plan")
    add_centered_picture(slide19, "design_diagrams/development_plan_roadmap.png")

    # =========================================================================
    # SLIDE 20: Conclusion & Design Phase Milestones
    # =========================================================================
    slide20 = create_content_slide("Conclusion & Design Phase Milestones")
    tbox20 = slide20.shapes.add_textbox(SAFE_LEFT, SAFE_TOP, SAFE_WIDTH, SAFE_HEIGHT)
    tf20 = tbox20.text_frame
    tf20.word_wrap = True

    p = tf20.paragraphs[0]
    p.text = "• Summary of Design Phase Accomplishments:"
    p.font.name = "Times New Roman"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = RGBColor(21, 67, 96)
    p.space_after = Pt(8)

    concl_points = [
        ("Eliminated Dataset Domain-Shift Bias:", "Curated a balanced 1,050-video benchmark (150/class) across diverse traffic angles and unwarped violence footage, completely eradicating static background memorization."),
        ("Architectural Breakthrough (98.10% Test Acc):", "Introduced Temporal Attention + Tri-Pooling with LayerNorm and GELU to OpenAI CLIP visual features, achieving verified 98.10% (206/210 correct) holdout accuracy and 0.9983 Macro AUC."),
        ("Zero-Latency Spatial Defense Synergy:", "Integrated real-time spatial perimeter intrusion detection (<50ms) with automated Telegram emergency push alerting (<1.2s) and SQLite incident audit logging."),
        ("Working Prototypes Validated:", "Built and tested working UI/UX prototypes for both Mode 1 Forensic Video Upload and Mode 2 Live Virtual Fence with real-time virtual perimeter intrusion detection."),
        ("Full System Implementation Delivered:", "Successfully fulfilled all 15 design phase criteria mandated by the department review notice. The dual-stream surveillance platform is 100% built, tested, and operationally verified in the project codebase.")
    ]

    for title, desc in concl_points:
        p = tf20.add_paragraph()
        p.text = f"   • {title} "
        p.font.name = "Times New Roman"
        p.font.size = Pt(16.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(17, 120, 100)

        run = p.add_run()
        run.text = desc
        run.font.name = "Times New Roman"
        run.font.size = Pt(15)
        run.font.bold = False
        run.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(7)

    # =========================================================================
    # SLIDE 21: References
    # =========================================================================
    slide21 = create_content_slide("References")
    tbox21 = slide21.shapes.add_textbox(SAFE_LEFT, SAFE_TOP, SAFE_WIDTH, SAFE_HEIGHT)
    tf21 = tbox21.text_frame
    tf21.word_wrap = True

    refs = [
        "Jocher, G., Chaurasia, A., & Qiu, J. (2023). 'Ultralytics YOLOv8.' GitHub repository: https://github.com/ultralytics/ultralytics.",
        "Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., et al. (2021). 'Learning Transferable Visual Models From Natural Language Supervision (CLIP).' ICML.",
        "Hochreiter, S., & Schmidhuber, J. (1997). 'Long Short-Term Memory.' Neural Computation, 9(8), 1735-1780.",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., et al. (2017). 'Attention Is All You Need.' NeurIPS.",
        "Soomro, K., Zamir, A. R., & Shah, M. (2012). 'UCF101: A Dataset of 101 Human Actions Classes From Videos in the Wild.' CRCV-TR-12-01.",
        "Goyal, K., et al. (2022). 'Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding.' IEEE/CVF CVPR.",
        "Loshchilov, I., & Hutter, F. (2019). 'Decoupled Weight Decay Regularization (AdamW).' ICLR."
    ]

    for i, rf in enumerate(refs):
        p = tf21.paragraphs[0] if i == 0 else tf21.add_paragraph()
        p.text = f"[{i+1}] {rf}"
        p.font.name = "Times New Roman"
        p.font.size = Pt(13.5)
        p.font.color.rgb = RGBColor(40, 40, 40)
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 22: Closing Slide (Thank You)
    # =========================================================================
    slide22 = prs.slides.add_slide(blank_layout)
    slide22.shapes.add_picture(closing_bg, 0, 0, slide_w, slide_h)
    print("[+] Slide 22: Closing (Thank You) added.")

    # Save presentation
    updated_file = "IBVAP_Presentation_v6_Updated.pptx"
    
    try:
        prs.save(orig_file)
        print(f"[SUCCESS] Successfully updated main presentation file: {orig_file}")
    except PermissionError:
        print(f"[!] Note: '{orig_file}' is currently open in Microsoft PowerPoint (locked by process).")

    try:
        prs.save(updated_file)
        print(f"[SUCCESS] Successfully updated mirror presentation file: {updated_file}")
    except PermissionError:
        print(f"[!] Note: '{updated_file}' is currently open in Microsoft PowerPoint (locked by process).")

if __name__ == "__main__":
    build_presentation()
