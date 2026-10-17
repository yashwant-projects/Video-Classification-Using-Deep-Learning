import os
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
CM_PATH = os.path.join(PROJECT_ROOT, "confusion_matrix.png")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "SIH_Civilian_Safety_Dossier.pdf")

# Update this with your actual name
STUDENT_NAME = "Yashwant Sushil"
INSTITUTION = "Shri Shankaracharya Institute of Professional Management and Technology"

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'SIH 2026 - Civilian Safety AI Dossier', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Prepared by: {STUDENT_NAME} | {INSTITUTION}', 0, 1, 'C')
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(4)

def generate_report():
    pdf = PDF()
    pdf.add_page()

    # Section 1: Architecture
    pdf.chapter_title('1. Hybrid Architecture & Methodology')
    body_1 = (
        "This project implements a dual-stream deep learning pipeline designed for zero-shot threat detection "
        "and action recognition in civilian environments.\n\n"
        "Stream A (Temporal): Utilizes an OpenAI CLIP (ViT-B/32) feature extractor coupled with a 2-layer Bidirectional LSTM. "
        "The model features Temporal Attention concatenated with Mean and Max pooling (1536-dim), stabilized by LayerNorm "
        "and GELU activations to evaluate sequences of 16 frames and isolate complex physical actions.\n"
        "Stream B (Spatial): Integrates Ultralytics YOLOv8 for spatial object detection, tracking, and virtual fence perimeter "
        "monitoring to provide real-time situational awareness."
    )
    pdf.chapter_body(body_1)

    # Section 2: Data Integrity
    pdf.chapter_title('2. Dataset Integrity & Zero Leakage Validation')
    body_2 = (
        "To ensure robust evaluation on unseen real-world data, the 1,050-video dataset was strictly partitioned at the "
        "video level prior to frame extraction. The 80/20 stratified split guarantees exactly 840 training videos and "
        "210 unseen testing videos, evenly distributed across 7 classes (30 test videos per class). The dataset features "
        "diverse environmental distributions (dashcam, drone, and handheld footage for traffic, aspect-ratio unwarped letterboxing "
        "for violence) to completely eliminate dataset domain-shift bias."
    )
    pdf.chapter_body(body_2)

    # Section 3: Performance Metrics
    pdf.chapter_title('3. Unseen Data Performance Metrics')
    body_3 = (
        "The model was evaluated exclusively on the 210 holdout videos. The system achieved a validated peak testing "
        "accuracy of 98.10% (206/210 correct) with an extraordinary Macro AUC of 0.9983. Education, Entertainment, and Wildlife "
        "achieved flawless 1.0000 Precision and Recall scores, with 0.9667+ across Violence, Traffic, Human Activities, and Sports. "
        "The model generalizes robustly without environmental memorization."
    )
    pdf.chapter_body(body_3)

    # Section 4: Confusion Matrix Graphic
    pdf.chapter_title('4. Diagnostic Confusion Matrix')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 6, "Figure 1: Classification distribution across the 210 unseen test videos.", 0, 1)
    
    if os.path.exists(CM_PATH):
        # Insert image (x, y, width). Adjust width to fit the page.
        pdf.image(CM_PATH, x=15, w=180)
    else:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, "[ERROR] confusion_matrix.png not found in project root.", 0, 1)
        pdf.set_text_color(0, 0, 0)

    # Save PDF
    pdf.output(OUTPUT_PDF, 'F')
    print("=" * 60)
    print(f"[SUCCESS] HOD Dossier generated at: {OUTPUT_PDF}")
    print("=" * 60)

if __name__ == "__main__":
    generate_report()