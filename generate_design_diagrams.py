# -*- coding: utf-8 -*-
"""
Generate high-resolution, professional design diagrams for 3rd Project Review (Design Phase).
FIXES:
1. Removed redundant top titles from diagrams so they don't clash with PowerPoint slide titles,
   giving diagrams 25% MORE vertical room and eliminating squashing.
2. Formatted all table cells in technology_selection_matrix with explicit textwrap.fill()
   so text NEVER gets truncated or overflows cell borders.
3. Expanded UML diagram boundaries:
   - In Use Case: moved actors inwards (x=1.8) away from left edge to stop ribbon collision.
   - In Sequence: expanded xlim to 13.5 so Telegram Bot box has 1.5 units of clearance on the right.
   - Increased vertical and horizontal clearance for all message arrows and labels.
4. Redesigned System / Process Flowchart (Slide 7) with ample spacing, wider decision diamonds,
   and clean non-overlapping connector lines.
5. Redesigned Integration Design (Slide 15) with separated bidirectional arrows, avoiding
   all label collisions between Web UI, Flask Hub, Streams A/B, Telegram, and SQLite.
"""

import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

output_dir = "design_diagrams"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. OVERALL ARCHITECTURE / BLOCK DIAGRAM (Checklist Point 3 - Slide 5)
# -----------------------------------------------------------------------------
def make_overall_architecture():
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)

    def draw_box(x, y, w, h, title, subtitle="", bg="#ebf5fb", border="#2980b9", title_color="#1a5276", title_fs=11, sub_fs=9):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                     edgecolor=border, facecolor=bg, linewidth=1.8)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, fontsize=title_fs, fontweight="bold", ha="center", va="center", color=title_color)
            ax.text(x + w/2, y + h*0.32, subtitle, fontsize=sub_fs, ha="center", va="center", color="#2c3e50")
        else:
            ax.text(x + w/2, y + h/2, title, fontsize=title_fs, fontweight="bold", ha="center", va="center", color=title_color)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.8, mutation_scale=15))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.14, label, fontsize=8.5, ha="center", color="#1a5276", fontweight="bold")

    # Group Containers
    # Ingestion Layer
    box_ingest = patches.FancyBboxPatch((0.6, 0.6), 3.2, 7.0, boxstyle="round,pad=0.2",
                                        edgecolor="#bdc3c7", facecolor="#f8f9fa", linewidth=1.5, linestyle="--")
    ax.add_patch(box_ingest)
    ax.text(2.2, 7.25, "1. INGESTION & PREPROCESSING", fontsize=11, fontweight="bold", color="#7f8c8d", ha="center")

    draw_box(0.9, 5.7, 2.6, 1.2, "Video Input Sources", "Webcam / Video File Upload / Stream", bg="#e8f8f5", border="#1abc9c", title_color="#117864")
    draw_box(0.9, 3.8, 2.6, 1.2, "Letterbox Normalization", "Aspect-Ratio Padding (No Warping)", bg="#e8f8f5", border="#1abc9c", title_color="#117864")
    draw_box(0.9, 1.6, 2.6, 1.5, "Temporal / Spatial Gate", "16-Frame Uniform Sampling\nYOLO Object Verification", bg="#e8f8f5", border="#1abc9c", title_color="#117864")

    draw_arrow(2.2, 5.7, 2.2, 5.0)
    draw_arrow(2.2, 3.8, 2.2, 3.1)

    # Core Processing Layer
    box_core = patches.FancyBboxPatch((4.3, 0.6), 6.5, 7.0, boxstyle="round,pad=0.2",
                                      edgecolor="#bdc3c7", facecolor="#f8f9fa", linewidth=1.5, linestyle="--")
    ax.add_patch(box_core)
    ax.text(7.55, 7.25, "2. DUAL-STREAM DEEP LEARNING ENGINES", fontsize=11, fontweight="bold", color="#7f8c8d", ha="center")

    # Stream A: Temporal
    box_stream_a = patches.FancyBboxPatch((4.6, 4.2), 5.9, 2.8, boxstyle="round,pad=0.15",
                                          edgecolor="#3498db", facecolor="#ebf5fb", linewidth=1.5)
    ax.add_patch(box_stream_a)
    ax.text(7.55, 6.7, "Stream A: Temporal Forensic Action Classifier (CLIP + BiLSTM)", fontsize=10.5, fontweight="bold", color="#1b4f72", ha="center")

    draw_box(4.9, 4.5, 2.5, 1.8, "CLIP ViT-B/32", "512-dim Embeddings\n(16 Frames Sequence)", bg="#ffffff", border="#2980b9", title_color="#1b4f72")
    draw_box(7.7, 4.5, 2.5, 1.8, "Bi-LSTM + Attention", "LayerNorm + GELU\n7-Class Softmax (98.10%)", bg="#ffffff", border="#2980b9", title_color="#1b4f72")
    draw_arrow(7.4, 5.4, 7.7, 5.4, "16x512")

    # Stream B: Spatial
    box_stream_b = patches.FancyBboxPatch((4.6, 1.0), 5.9, 2.8, boxstyle="round,pad=0.15",
                                          edgecolor="#e67e22", facecolor="#fef5e7", linewidth=1.5)
    ax.add_patch(box_stream_b)
    ax.text(7.55, 3.5, "Stream B: Real-Time Spatial Surveillance (YOLOv8)", fontsize=10.5, fontweight="bold", color="#b9770e", ha="center")

    draw_box(4.9, 1.3, 2.5, 1.8, "YOLOv8 Detection", "Person & Vehicle Tracking\n(Bottom-Center Coordinates)", bg="#ffffff", border="#d35400", title_color="#a04000")
    draw_box(7.7, 1.3, 2.5, 1.8, "Virtual Fence Engine", "Point-in-Polygon Check\nReal-Time Breach Logic", bg="#ffffff", border="#d35400", title_color="#a04000")
    draw_arrow(7.4, 2.2, 7.7, 2.2, "BBox (x,y)")

    # Arrows from Ingestion to Streams
    draw_arrow(3.5, 4.6, 4.6, 5.4, "16 Frames")
    draw_arrow(3.5, 2.4, 4.6, 2.2, "Live Stream")

    # Outputs & Services Layer
    box_out = patches.FancyBboxPatch((11.3, 0.6), 3.2, 7.0, boxstyle="round,pad=0.2",
                                     edgecolor="#bdc3c7", facecolor="#f8f9fa", linewidth=1.5, linestyle="--")
    ax.add_patch(box_out)
    ax.text(12.9, 7.25, "3. INTEGRATION & OUTPUTS", fontsize=11, fontweight="bold", color="#7f8c8d", ha="center")

    draw_box(11.6, 5.7, 2.6, 1.2, "Flask Web Dashboard", "Interactive HUD & Results", bg="#eaeded", border="#7f8c8d", title_color="#2c3e50")
    draw_box(11.6, 4.2, 2.6, 1.2, "Telegram Bot API", "Instant Intruder Alert", bg="#eaeded", border="#27ae60", title_color="#1e8449")
    draw_box(11.6, 2.7, 2.6, 1.2, "SQLite Database", "surveillance.db Logging", bg="#eaeded", border="#8e44ad", title_color="#6c3483")
    draw_box(11.6, 1.1, 2.6, 1.3, "Automated PDF Dossier", "FPDF HOD Report Generator", bg="#eaeded", border="#c0392b", title_color="#922b21")

    # Arrows from Streams to Outputs
    draw_arrow(10.2, 5.4, 11.6, 6.2, "Classification")
    draw_arrow(10.2, 5.0, 11.6, 1.7, "Metrics")
    draw_arrow(10.2, 2.2, 11.6, 4.7, "Breach Alert")
    draw_arrow(10.2, 1.9, 11.6, 3.2, "Log Event")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "overall_architecture_block_diagram.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved overall_architecture_block_diagram.png")

# -----------------------------------------------------------------------------
# 2. SYSTEM / PROCESS FLOW (Checklist Point 5 - Slide 7)
# -----------------------------------------------------------------------------
def make_system_process_flow():
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)

    def draw_node(x, y, w, h, text, shape="rect", bg="#ebf5fb", border="#2980b9", tc="#1a5276", fs=9.0):
        if shape == "round":
            p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", edgecolor=border, facecolor=bg, lw=1.8)
        elif shape == "diamond":
            pts = np.array([[x+w/2, y], [x+w, y+h/2], [x+w/2, y+h], [x, y+h/2]])
            p = patches.Polygon(pts, closed=True, edgecolor=border, facecolor=bg, lw=1.8)
        else:
            p = patches.Rectangle((x, y), w, h, edgecolor=border, facecolor=bg, lw=1.8)
        ax.add_patch(p)
        ax.text(x+w/2, y+h/2, text, fontsize=fs, fontweight="bold", ha="center", va="center", color=tc, multialignment="center")

    def draw_connector(p1, p2, label="", color="#2c3e50", badge_bg="#ffffff", label_color=None):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=12))
        if label:
            mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            lc = label_color if label_color else color
            ax.text(mx, my, label, fontsize=8.2, ha="center", va="center", color=lc, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=badge_bg, edgecolor="#bdc3c7", lw=0.9))

    # Background column cards to visually group the two operational modes
    card_m1 = patches.FancyBboxPatch((0.7, 0.2), 5.4, 5.85, boxstyle="round,pad=0.15",
                                     edgecolor="#d4e6f1", facecolor="#fafdff", lw=1.5, linestyle="--")
    ax.add_patch(card_m1)
    
    card_m2 = patches.FancyBboxPatch((8.9, 0.2), 5.4, 5.85, boxstyle="round,pad=0.15",
                                     edgecolor="#fae5d3", facecolor="#fffdfb", lw=1.5, linestyle="--")
    ax.add_patch(card_m2)

    # Start Node (Centered at x=7.5)
    draw_node(5.8, 7.35, 3.4, 0.60, "START: User Ingestion Entrypoint", shape="round", bg="#d4efdf", border="#27ae60", tc="#145a32", fs=9.5)

    # Operational Mode Selection Decision Diamond (w=4.6, h=1.20, center=(7.5, 6.30))
    # Left vertex: (5.2, 6.30), Right vertex: (9.8, 6.30), Top vertex: (7.5, 6.90), Bottom vertex: (7.5, 5.70)
    draw_node(5.2, 5.70, 4.6, 1.20, "Operational Mode Selection:\nForensic Upload (1) or Live Stream (2)?", shape="diamond", bg="#fef9e7", border="#f39c12", tc="#7d6608", fs=9.0)
    
    # Arrow from Start to Decision Diamond
    draw_connector((7.5, 7.35), (7.5, 6.90))

    # --- Mode 1 Branch (Left, Column Centerline x=3.4) ---
    # Header Badge (x in [1.3, 5.5], top edge y=5.65)
    draw_node(1.3, 5.10, 4.2, 0.55, "MODE 1: FORENSIC ACTION RECOGNITION", shape="round", bg="#ebf5fb", border="#2980b9", tc="#1b4f72", fs=8.5)
    
    # Orthogonal Connector from Diamond left vertex (5.2, 6.30) -> (3.4, 6.30) -> (3.4, 5.65)
    ax.plot([5.2, 3.4], [6.30, 6.30], color="#2c3e50", lw=1.8)
    # Badge on horizontal segment
    ax.text(4.30, 6.30, "Mode 1: Forensic Upload", fontsize=8.0, ha="center", va="center", color="#1a5276", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#2980b9", lw=0.9))
    # Downward arrow entering top-center of Mode 1 header box (ZERO OVERLAP!)
    draw_connector((3.4, 6.30), (3.4, 5.65))

    # Mode 1 Steps (Clean vertical chain centered at x=3.4)
    draw_connector((3.4, 5.10), (3.4, 4.70))
    draw_node(1.2, 3.95, 4.4, 0.75, "Video Ingestion & Integrity Check\nUpload .mp4 / .avi | Container & Codec Check", bg="#ffffff", border="#3498db", tc="#1b4f72")

    draw_connector((3.4, 3.95), (3.4, 3.55))
    draw_node(1.2, 2.80, 4.4, 0.75, "Uniform Temporal Sampling\nExtract 16 Frames (Aspect-Ratio Letterboxed)", bg="#ffffff", border="#3498db", tc="#1b4f72")

    draw_connector((3.4, 2.80), (3.4, 2.40))
    draw_node(1.2, 1.65, 4.4, 0.75, "CLIP ViT-B/32 Feature Extraction\n16 x 512 Dense Feature Vector Sequence", bg="#ffffff", border="#3498db", tc="#1b4f72")

    draw_connector((3.4, 1.65), (3.4, 1.25))
    draw_node(1.0, 0.35, 4.8, 0.90, "Bi-LSTM + Temporal Attention (Tri-Pooling)\n7-Class Softmax Verdict (98.10% Holdout Acc)\nExport Automated PDF Forensic Dossier", shape="round", bg="#d4efdf", border="#27ae60", tc="#145a32", fs=8.5)

    # --- Mode 2 Branch (Right, Column Centerline x=11.6) ---
    # Header Badge (x in [9.5, 13.7], top edge y=5.65)
    draw_node(9.5, 5.10, 4.2, 0.55, "MODE 2: LIVE SURVEILLANCE & VIRTUAL FENCE", shape="round", bg="#fef5e7", border="#e67e22", tc="#b9770e", fs=8.5)
    
    # Orthogonal Connector from Diamond right vertex (9.8, 6.30) -> (11.6, 6.30) -> (11.6, 5.65)
    ax.plot([9.8, 11.6], [6.30, 6.30], color="#2c3e50", lw=1.8)
    # Badge on horizontal segment
    ax.text(10.70, 6.30, "Mode 2: Live Stream", fontsize=8.0, ha="center", va="center", color="#b9770e", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#e67e22", lw=0.9))
    # Downward arrow entering top-center of Mode 2 header box (ZERO OVERLAP!)
    draw_connector((11.6, 6.30), (11.6, 5.65))

    # Mode 2 Steps (Clean vertical chain centered at x=11.6)
    draw_connector((11.6, 5.10), (11.6, 4.70))
    draw_node(9.4, 3.95, 4.4, 0.75, "Initialize Live Video Stream\nOperator Activates Virtual Perimeter HUD", bg="#ffffff", border="#e67e22", tc="#784212")

    draw_connector((11.6, 3.95), (11.6, 3.55))
    draw_node(9.4, 2.80, 4.4, 0.75, "YOLOv8 Real-Time Spatial Tracking Loop\nObject Bounding Boxes & Footprint Anchors (45+ FPS)", bg="#ffffff", border="#e67e22", tc="#784212")

    draw_connector((11.6, 2.80), (11.6, 2.35))
    # Collision Check Diamond (Center=(11.6, 1.80), w=4.4, h=1.10, left=(9.4, 1.80), bottom=(11.6, 1.25))
    draw_node(9.4, 1.25, 4.4, 1.10, "Ray-Casting Collision Check:\nBottom Anchor Inside Virtual Fence?", shape="diamond", bg="#fef9e7", border="#f39c12", tc="#7d6608", fs=8.5)

    # Decision branches from Diamond 2
    # No branch (Safe): routes from left vertex (9.4, 1.80) horizontally to (7.5, 1.80), then down into Safe box at (7.5, 1.15)
    ax.plot([9.4, 7.5], [1.80, 1.80], color="#2c3e50", lw=1.8)
    draw_connector((7.5, 1.80), (7.5, 1.15))
    ax.text(8.45, 1.80, "No (Safe)", fontsize=8.0, fontweight="bold", color="#27ae60", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#27ae60", lw=0.9))
    draw_node(6.2, 0.45, 2.6, 0.70, "SAFE: Green Overlay\nContinue Stream Monitoring", shape="round", bg="#eaeded", border="#7f8c8d", tc="#2c3e50", fs=8.0)

    # Yes branch (Breach): routes straight down from bottom vertex (11.6, 1.25) to (11.6, 1.15)
    draw_connector((11.6, 1.25), (11.6, 1.15))
    ax.text(12.35, 1.20, "Yes (Breach)", fontsize=8.0, fontweight="bold", color="#c0392b", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#c0392b", lw=0.9))
    draw_node(9.4, 0.35, 4.4, 0.80, "BREACH TRIGGERED!\n1. Log to SQLite (surveillance.db)\n2. Dispatch Instant Telegram Alert (<1.2s)", shape="round", bg="#fadbd8", border="#e74c3c", tc="#78281f", fs=8.0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "system_process_flowchart.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved system_process_flowchart.png (with pristine zero-overlap layout).")

# -----------------------------------------------------------------------------
# 3. UML DIAGRAMS: USE CASE & SEQUENCE (Checklist Point 6 - Slide 8)
# -----------------------------------------------------------------------------
def make_uml_diagrams():
    fig, (ax_uc, ax_seq) = plt.subplots(1, 2, figsize=(16, 8.2), dpi=300, gridspec_kw={'width_ratios': [1, 1.4]})
    
    # Left: Use Case Diagram
    ax_uc.axis("off")
    ax_uc.set_xlim(0, 10.5)
    ax_uc.set_ylim(0, 10)

    # Card background for Use Case
    card_uc = patches.FancyBboxPatch((0.2, 0.2), 10.1, 9.6, boxstyle="round,pad=0.15",
                                     edgecolor="#d5d8dc", facecolor="#ffffff", lw=1.2)
    ax_uc.add_patch(card_uc)

    # Boundary Box
    sys_box = patches.Rectangle((3.5, 0.5), 6.5, 8.8, edgecolor="#2980b9", facecolor="#f8fbfd", lw=1.8, linestyle="--")
    ax_uc.add_patch(sys_box)
    ax_uc.text(6.75, 9.0, "System: IBVAP Surveillance Platform", fontsize=10.5, fontweight="bold", color="#1b4f72", ha="center")

    # Actors - placed at x=1.8 with plenty of margin on the left (x in [0.2, 10.5])
    def draw_actor(ax, x, y, label):
        head = patches.Circle((x, y+0.42), 0.20, edgecolor="#2c3e50", facecolor="#ebf5fb", lw=1.5)
        ax.add_patch(head)
        ax.plot([x, x], [y+0.22, y-0.20], color="#2c3e50", lw=1.5) # body
        ax.plot([x-0.22, x+0.22], [y+0.05, y+0.05], color="#2c3e50", lw=1.5) # arms
        ax.plot([x, x-0.20], [y-0.20, y-0.55], color="#2c3e50", lw=1.5) # left leg
        ax.plot([x, x+0.20], [y-0.20, y-0.55], color="#2c3e50", lw=1.5) # right leg
        ax.text(x, y-0.80, label, fontsize=8.5, fontweight="bold", ha="center", color="#1a5276")

    draw_actor(ax_uc, 1.8, 6.8, "Security\nOperator")
    draw_actor(ax_uc, 1.8, 2.6, "Forensic\nInvestigator")

    use_cases = [
        (6.75, 7.9, "UC-1: Access Surveillance Dashboard"),
        (6.75, 6.8, "UC-2: Initialize Virtual Fence Feed"),
        (6.75, 5.7, "UC-3: Monitor Live Surveillance Feed"),
        (6.75, 4.6, "UC-4: Upload Forensic Video for Classification"),
        (6.75, 3.5, "UC-5: Review Threat Classification & Conf."),
        (6.75, 2.4, "UC-6: Export PDF Forensic Dossier"),
        (6.75, 1.3, "UC-7: Receive Telegram Breach Alerts")
    ]

    for ux, uy, utext in use_cases:
        ellipse = patches.Ellipse((ux, uy), 5.7, 0.78, edgecolor="#27ae60", facecolor="#eafaf1", lw=1.4)
        ax_uc.add_patch(ellipse)
        ax_uc.text(ux, uy, utext, fontsize=8.0, ha="center", va="center", color="#145a32", fontweight="bold")

    # Lines from Operator
    for target_y in [7.9, 6.8, 5.7, 1.3]:
        ax_uc.plot([2.1, 3.9], [6.8, target_y], color="#7f8c8d", lw=1.2)

    # Lines from Investigator
    for target_y in [4.6, 3.5, 2.4]:
        ax_uc.plot([2.1, 3.9], [2.6, target_y], color="#7f8c8d", lw=1.2)

    # Right: Sequence Diagram
    ax_seq.axis("off")
    ax_seq.set_xlim(0, 14.2)
    ax_seq.set_ylim(0, 10)

    # Card background for Sequence Diagram
    card_seq = patches.FancyBboxPatch((0.2, 0.2), 13.8, 9.6, boxstyle="round,pad=0.15",
                                      edgecolor="#d5d8dc", facecolor="#ffffff", lw=1.2)
    ax_seq.add_patch(card_seq)

    ax_seq.text(7.1, 9.5, "UML Sequence Diagram: Live Intrusion Breach Flow", fontsize=10.5, fontweight="bold", color="#1b4f72", ha="center")

    lifelines = [
        ("Operator", 1.2),
        ("Web UI", 3.4),
        ("Flask Server", 5.6),
        ("YOLO / Fence", 7.8),
        ("SQLite DB", 10.0),
        ("Telegram Bot", 12.2)
    ]

    for name, lx in lifelines:
        box = patches.FancyBboxPatch((lx-0.85, 8.7), 1.7, 0.52, boxstyle="round,pad=0.1", 
                                     edgecolor="#2980b9", facecolor="#ebf5fb", lw=1.2)
        ax_seq.add_patch(box)
        ax_seq.text(lx, 8.96, name, fontsize=8.0, fontweight="bold", ha="center", va="center", color="#1b4f72")
        ax_seq.plot([lx, lx], [8.7, 0.5], color="#bdc3c7", lw=1.2, linestyle="--")

    msgs = [
        (1.2, 3.4, 8.0, "1: Access Surveillance HUD"),
        (3.4, 5.6, 7.3, "2: POST /toggle_camera (Init Feed)"),
        (5.6, 7.8, 6.6, "3: Stream Live Webcam Frame"),
        (7.8, 7.8, 5.9, "4: YOLOv8 Track + PointPolygonTest", True),
        (7.8, 5.6, 5.1, "5: Return [Breach Detected = True]"),
        (5.6, 10.0, 4.3, "6: INSERT INTO security_events (time, track_id)"),
        (5.6, 12.2, 3.5, "7: Dispatch Alert (Markdown Text)"),
        (12.2, 1.2, 2.7, "8: Mobile Notification to Telegram Chat"),
        (5.6, 3.4, 1.9, "9: Push Annotated MJPEG Frame"),
        (3.4, 1.2, 1.1, "10: Audio-Visual Alarm Displayed")
    ]

    for x1, x2, my, mtext, *is_self in msgs:
        if is_self and is_self[0]:
            # self call
            ax_seq.annotate("", xy=(x1, my-0.32), xytext=(x1, my),
                            arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=1.4,
                                            connectionstyle="arc3,rad=0.4", mutation_scale=10))
            ax_seq.text(x1+0.15, my-0.16, mtext, fontsize=7.5, color="#d35400", fontweight="bold")
        else:
            ax_seq.annotate("", xy=(x2, my), xytext=(x1, my),
                            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.3, mutation_scale=10))
            ax_seq.text((x1+x2)/2, my+0.12, mtext, fontsize=7.5, ha="center", va="bottom", color="#1a5276", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="none"))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "uml_diagrams_combined.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved uml_diagrams_combined.png (with clean card margins & zero clipping).")

# -----------------------------------------------------------------------------
# 4. DATA DESIGN & ER DIAGRAM (Checklist Point 7 - Slide 9)
# -----------------------------------------------------------------------------
def make_data_design():
    fig, (ax_ds, ax_er) = plt.subplots(1, 2, figsize=(16, 8.2), dpi=300, gridspec_kw={'width_ratios': [1, 1.1]})

    ax_ds.axis("off")
    ax_ds.set_xlim(0, 10)
    ax_ds.set_ylim(0, 9.5)

    ds_data = [
        ["Category", "Source Dataset", "Train", "Test", "Preprocessing"],
        ["Education", "UCF101", "120", "30", "Letterbox + YOLO-Gate"],
        ["Entertainment", "UCF101", "120", "30", "Letterbox + YOLO-Gate"],
        ["Human Activities", "UCF101", "120", "30", "Letterbox + YOLO-Gate"],
        ["Sports", "UCF101", "120", "30", "Letterbox + YOLO-Gate"],
        ["Traffic (Diverse)", "YouTube Scraper", "120", "30", "Multi-Angle / Street"],
        ["Violence", "Real Life Violence", "120", "30", "Aspect-Ratio Padding"],
        ["Wildlife", "Animal-Kingdom", "120", "30", "Action-Gated Fauna"],
        ["TOTALS", "7 Curated Sources", "840", "210", "16,800 Extracted Frames"]
    ]

    t_ds = ax_ds.table(cellText=ds_data, cellLoc="center", loc="center", colWidths=[0.24, 0.28, 0.12, 0.12, 0.24])
    t_ds.auto_set_font_size(False)
    t_ds.set_fontsize(8.5)
    t_ds.scale(1, 1.55)

    for (r, c), cell in t_ds.get_celld().items():
        cell.set_edgecolor("#bdc3c7")
        if r == 0:
            cell.set_facecolor("#2980b9")
            cell.set_text_props(weight="bold", color="#ffffff")
        elif r == 8:
            cell.set_facecolor("#ebf5fb")
            cell.set_text_props(weight="bold", color="#1b4f72")
        else:
            cell.set_facecolor("#ffffff" if r % 2 == 1 else "#fcfcfc")

    callout = patches.FancyBboxPatch((0.5, 0.3), 9.0, 1.3, boxstyle="round,pad=0.15",
                                     edgecolor="#27ae60", facecolor="#eafaf1", lw=1.5)
    ax_ds.add_patch(callout)
    ax_ds.text(5.0, 1.15, "CLIP Embedding Serialization Pipeline", fontsize=10, fontweight="bold", color="#145a32", ha="center")
    ax_ds.text(5.0, 0.65, "16 YOLO-Gated Frames  -->  OpenAI CLIP ViT-B/32  -->  Shape: (16, 512) Float32\n"
                          "Lightweight .npy feature vectors: Zero data leakage between Train (840) and Holdout Test (210)",
               fontsize=8, ha="center", color="#1e8449")

    # Right: ER Diagram
    ax_er.axis("off")
    ax_er.set_xlim(0, 11)
    ax_er.set_ylim(0, 9.5)

    ax_er.text(5.5, 9.1, "Database Schema & Persistent Storage Architecture", fontsize=10.5, fontweight="bold", color="#1b4f72", ha="center")

    def draw_entity(ax, x, y, title, fields, bg="#ebf5fb", border="#2980b9", w=9.8):
        h = 0.45 + len(fields) * 0.35
        box_hdr = patches.Rectangle((x, y + len(fields)*0.35), w, 0.45, facecolor=border, edgecolor=border)
        ax.add_patch(box_hdr)
        ax.text(x + w/2, y + len(fields)*0.35 + 0.22, title, fontsize=9.5, fontweight="bold", color="#ffffff", ha="center", va="center")
        box_body = patches.Rectangle((x, y), w, len(fields)*0.35, facecolor=bg, edgecolor=border, lw=1.2)
        ax.add_patch(box_body)
        for i, (pk, fname, ftype) in enumerate(fields):
            fy = y + len(fields)*0.35 - (i + 1)*0.35 + 0.10
            ax.text(x + 0.25, fy, pk, fontsize=7.5, fontweight="bold", color="#c0392b" if pk == "PK" else "#7f8c8d")
            ax.text(x + 1.0, fy, fname, fontsize=8, fontweight="bold", color="#2c3e50")
            ax.text(x + w - 0.25, fy, ftype, fontsize=7.5, style="italic", color="#7f8c8d", ha="right")

    # Table 1: Real SQLite Table from app.py
    draw_entity(ax_er, 0.6, 6.4, "security_events (surveillance.db - Embedded SQLite Table)", [
        ("PK", "id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("", "timestamp", "TEXT (Event Timestamp: YYYY-MM-DD HH:MM:SS)"),
        ("", "event_type", "TEXT ('Virtual Fence Breach')"),
        ("", "track_id", "INTEGER (YOLOv8 Persistent Tracking ID)")
    ], border="#c0392b", bg="#fcedec")

    # Store 2: Real Serialized Features from extract_features.py
    draw_entity(ax_er, 0.6, 3.6, "Dataset_Features/ (Serialized CLIP Feature Store)", [
        ("", "video_id.npy", "1,050 NPY Feature Files (840 Train / 210 Holdout Test)"),
        ("", "feature_tensor", "Shape: (16, 512) Float32 Normalized Embeddings"),
        ("", "vision_backbone", "OpenAI CLIP (ViT-B/32) Visual Projection Layer"),
        ("", "temporal_stride", "16 YOLO-Gated Frames / Video (Zero Data Leakage)")
    ], border="#2980b9", bg="#ebf5fb")

    # Store 3: Real Model Weights & Reports from project root
    draw_entity(ax_er, 0.6, 0.8, "models/ & artifacts/ (Persistent Checkpoint Store)", [
        ("", "models/sih_lstm_model.pth", "Golden Bi-LSTM + Attn State Dict (3.48M Params)"),
        ("", "yolov8s.pt / yolov8n.pt", "Ultralytics Weights (Spatial Tracking & Object Gating)"),
        ("", "uploads/", "Temporary In-Memory/Disk Video Ingestion Buffer"),
        ("", "SIH_Civilian_Safety_Dossier.pdf", "Executive Forensic Intelligence PDF Dossier")
    ], border="#8e44ad", bg="#f4ecf7")

    # Flow indicators
    ax_er.annotate("", xy=(5.5, 6.4), xytext=(5.5, 5.55),
                   arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, linestyle="--"))
    ax_er.text(5.5, 5.95, "Live YOLO Tracking Logs Breach Events", fontsize=7.5, fontweight="bold", color="#c0392b", ha="center",
               bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#bdc3c7", lw=0.8))

    ax_er.annotate("", xy=(5.5, 2.75), xytext=(5.5, 3.6),
                   arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, linestyle="--"))
    ax_er.text(5.5, 3.15, "Features Ingested by Bi-LSTM Model", fontsize=7.5, fontweight="bold", color="#8e44ad", ha="center",
               bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#bdc3c7", lw=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "data_design_er_diagram.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved data_design_er_diagram.png")

# -----------------------------------------------------------------------------
# 5. HARDWARE & SOFTWARE DESIGN (Checklist Point 9 - Slide 12)
# -----------------------------------------------------------------------------
def make_hardware_software_stack():
    fig, (ax_hw, ax_sw) = plt.subplots(1, 2, figsize=(16, 8.2), dpi=300, gridspec_kw={'width_ratios': [1.12, 0.88]})

    ax_hw.axis("off")
    ax_hw.set_xlim(0, 10)
    ax_hw.set_ylim(0, 9.5)

    hw_specs = [
        ["Hardware Subsystem", "Tested System Specs", "Configuration / Mode", "Operational Function"],
        ["Host Processor", "Intel Core i7 / AMD Ryzen", "Multi-Threaded Host CPU", "Flask Server & Worker Queue"],
        ["GPU Accelerator", "NVIDIA CUDA GPU (RTX)", "CUDA 12.x / Tensor Cores", "CLIP ViT-B/32 & YOLOv8 Inference"],
        ["System Memory", "16 GB DDR4 / DDR5 RAM", "High-Throughput Buffer", "Video Decode & Batch Queues"],
        ["Video Sensor", "HD Optical Webcam / USB", "OpenCV VideoCapture(0)", "Real-Time Perimeter Surveillance"],
        ["Local Storage", "512 GB / 1 TB NVMe SSD", "High-Speed Random I/O", "16,800 Frames, .npy & SQLite DB"],
        ["Network Interface", "Broadband / Wi-Fi 6", "HTTPS Webhook Protocol", "Telegram Bot Push Alerts (<1.2s)"]
    ]

    t_hw = ax_hw.table(cellText=hw_specs, cellLoc="center", loc="center", colWidths=[0.19, 0.24, 0.24, 0.33])
    t_hw.auto_set_font_size(False)
    t_hw.set_fontsize(7.8)
    t_hw.scale(1, 1.6)

    for (r, c), cell in t_hw.get_celld().items():
        cell.set_edgecolor("#bdc3c7")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(weight="bold", color="#ffffff")
        else:
            cell.set_facecolor("#ffffff" if r % 2 == 1 else "#f8f9fa")

    # Right: Software Stack Layers
    ax_sw.axis("off")
    ax_sw.set_xlim(0, 10)
    ax_sw.set_ylim(0, 9.5)

    layers = [
        ("Layer 5: Presentation & Reporting", "Flask Web UI (HTML5 Canvas + Jinja2) | Telegram Bot API | FPDF Dossier Generator", "#d4efdf", "#27ae60", "#145a32"),
        ("Layer 4: Application & Business Logic", "Mode 1 Forensic Analysis | Mode 2 Virtual Fence Collision Engine | SQLite3 (surveillance.db)", "#ebf5fb", "#2980b9", "#1b4f72"),
        ("Layer 3: Core Deep Learning Engines", "Ultralytics YOLOv8s (Spatial Tracking) | OpenAI CLIP ViT-B/32 | Bi-LSTM + Temporal Attention", "#fef9e7", "#f39c12", "#7d6608"),
        ("Layer 2: Vision & Numerical Libraries", "PyTorch 2.x | Torchvision | OpenCV (cv2) | NumPy | Scikit-Learn | HuggingFace Transformers", "#f4ecf7", "#8e44ad", "#512e5f"),
        ("Layer 1: OS, Drivers & Accelerated Compute", "Windows 11 / Ubuntu Linux | NVIDIA CUDA 12.x | cuDNN v8 Accelerated Compute", "#fadbd8", "#e74c3c", "#78281f")
    ]

    for i, (lname, ldesc, bg, border, tc) in enumerate(layers):
        y = 7.3 - i * 1.55
        box = patches.FancyBboxPatch((0.5, y), 9.0, 1.2, boxstyle="round,pad=0.15",
                                     edgecolor=border, facecolor=bg, lw=1.6)
        ax_sw.add_patch(box)
        ax_sw.text(5.0, y + 0.75, lname, fontsize=10, fontweight="bold", color=tc, ha="center")
        ax_sw.text(5.0, y + 0.32, ldesc, fontsize=8.5, color="#2c3e50", ha="center")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "hardware_software_stack.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved hardware_software_stack.png")

# -----------------------------------------------------------------------------
# 6. TECHNOLOGY SELECTION MATRIX (Checklist Point 11 - Slide 14) -> BESPOKE CLEAN CELLS
# -----------------------------------------------------------------------------
def make_technology_selection_matrix():
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)

    matrix_data = [
        ("Feature Extractor", "OpenAI CLIP\n(ViT-B/32)", "ResNet-50 /\nVGG-16", 
         "Pretrained on 400M diverse pairs. Eliminates ImageNet dataset bias & enables zero-shot generalization across novel video conditions."),
        ("Sequence Classifier", "Bi-LSTM +\nTemporal Attn", "3D-CNN (I3D /\nSlowFast)", 
         "Lightweight (3.48M params vs 30M+). Runs in real-time on edge compute without massive VRAM; Temporal Attention dynamically isolates crucial action frames."),
        ("Spatial Detection", "Ultralytics\nYOLOv8", "Faster R-CNN /\nSSD", 
         "Anchor-free single-stage architecture. Delivers 45+ FPS inference latency, perfect for real-time virtual fence perimeter crossing detection."),
        ("Web App Backend", "Flask (Python\nWSGI)", "Django /\nFastAPI", 
         "Lightweight with zero boilerplate overhead. Connects directly in-memory to running PyTorch models and OpenCV streams without ORM drag."),
        ("Surveillance Database", "SQLite3\n(Embedded)", "PostgreSQL /\nMySQL", 
         "Self-contained zero-configuration serverless database. Zero network socket overhead; ideal for isolated remote border post edge installations."),
        ("Alert Dispatcher", "Telegram\nBot API", "Twilio SMS /\nEmail", 
         "Instant push delivery (<1.2s) with zero per-message cost. Dispatches real-time markdown alerts with intruder ID & event timestamps directly to officers."),
        ("Forensic Reporting", "FPDF Library\n(Python)", "ReportLab /\nLaTeX", 
         "Pure-Python programmatic PDF compilation. Generates professional compliance dossiers with embedded confusion matrices in under 200ms.")
    ]

    col_widths = [2.5, 2.4, 2.4, 6.9]
    col_x = [0.4, 0.4 + 2.5, 0.4 + 2.5 + 2.4, 0.4 + 2.5 + 2.4 + 2.4]
    
    # Outer Card / Border
    total_w = sum(col_widths)
    hdr_h = 0.65
    row_h = 0.95
    total_h = hdr_h + len(matrix_data) * row_h
    top_y = 7.7
    
    card_bg = patches.Rectangle((col_x[0], top_y - total_h), total_w, total_h, 
                                facecolor="#ffffff", edgecolor="#bdc3c7", lw=1.5)
    ax.add_patch(card_bg)

    # Header Row
    hdr_titles = ["Subsystem / Role", "Selected Technology", "Alternative Evaluated", "Architectural Justification & Trade-off Advantage"]
    for j in range(4):
        bx = patches.Rectangle((col_x[j], top_y - hdr_h), col_widths[j], hdr_h,
                               facecolor="#1b4f72", edgecolor="#bdc3c7", lw=1.0)
        ax.add_patch(bx)
        ax.text(col_x[j] + col_widths[j]/2, top_y - hdr_h/2, hdr_titles[j], 
                fontsize=9.5, fontweight="bold", color="#ffffff", ha="center", va="center")

    # Data Rows
    for i, (subsys, sel, alt, just) in enumerate(matrix_data):
        curr_y = top_y - hdr_h - (i + 1) * row_h
        bg_color = "#ffffff" if i % 2 == 0 else "#fcfcfc"

        # Col 0: Subsystem
        bx0 = patches.Rectangle((col_x[0], curr_y), col_widths[0], row_h,
                                facecolor=bg_color, edgecolor="#bdc3c7", lw=0.8)
        ax.add_patch(bx0)
        ax.text(col_x[0] + 0.15, curr_y + row_h/2, subsys,
                fontsize=9.0, fontweight="bold", color="#1a5276", ha="left", va="center")

        # Col 1: Selected Technology (Green Badge)
        bx1 = patches.Rectangle((col_x[1], curr_y), col_widths[1], row_h,
                                facecolor="#e8f8f5", edgecolor="#bdc3c7", lw=0.8)
        ax.add_patch(bx1)
        ax.text(col_x[1] + col_widths[1]/2, curr_y + row_h/2, sel,
                fontsize=8.5, fontweight="bold", color="#117864", ha="center", va="center", multialignment="center")

        # Col 2: Alternative Evaluated (Muted Red Badge)
        bx2 = patches.Rectangle((col_x[2], curr_y), col_widths[2], row_h,
                                facecolor="#fadbd8", edgecolor="#bdc3c7", lw=0.8)
        ax.add_patch(bx2)
        ax.text(col_x[2] + col_widths[2]/2, curr_y + row_h/2, alt,
                fontsize=8.5, color="#78281f", ha="center", va="center", multialignment="center")

        # Col 3: Justification & Trade-off Advantage (Clean wrap, generous vertical breathing room)
        bx3 = patches.Rectangle((col_x[3], curr_y), col_widths[3], row_h,
                                facecolor=bg_color, edgecolor="#bdc3c7", lw=0.8)
        ax.add_patch(bx3)
        wrapped_just = textwrap.fill(just, width=56)
        ax.text(col_x[3] + 0.20, curr_y + row_h/2, wrapped_just,
                fontsize=8.2, color="#2c3e50", ha="left", va="center", multialignment="left")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "technology_selection_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved technology_selection_matrix.png (with bespoke custom-rendered table cells).")

# -----------------------------------------------------------------------------
# 7. INTEGRATION DESIGN (Checklist Point 12 - Slide 15) -> FIXED ARROW CLEARANCE
# -----------------------------------------------------------------------------
def make_integration_design():
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)

    def draw_box(x, y, w, h, title, subtitle="", bg="#ebf5fb", border="#2980b9", tc="#1a5276"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", edgecolor=border, facecolor=bg, lw=1.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.64, title, fontsize=10.5, fontweight="bold", ha="center", va="center", color=tc)
        if subtitle:
            ax.text(x + w/2, y + h*0.30, subtitle, fontsize=8.2, ha="center", va="center", color="#2c3e50")

    # Central Controller: Flask Hub (x in [5.1, 8.9], y in [2.8, 5.2])
    draw_box(5.1, 2.8, 3.8, 2.4, "Flask Application Hub (app.py)", "Central Dispatcher & Routing Controller\nStateful Video Ingestion Engine", bg="#d4efdf", border="#27ae60", tc="#145a32")

    # Left: Web Client (x in [0.5, 3.7], y in [2.8, 5.2])
    draw_box(0.5, 2.8, 3.2, 2.4, "Web Browser Frontend", "Perimeter Surveillance HUD\nFetch API & MJPEG Stream Viewer", bg="#ebf5fb", border="#2980b9")
    
    # Left-Center Bidirectional Arrows with CLEAR SEPARATION:
    # Top arrow: HTTP POST (x=3.7 -> x=5.1 at y=4.3)
    ax.annotate("", xy=(5.1, 4.3), xytext=(3.7, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6, mutation_scale=12))
    ax.text(4.4, 4.65, "POST /toggle_camera (Init)", fontsize=7.8, ha="center", va="center", color="#1a5276", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#2980b9", lw=0.9))

    # Bottom arrow: MJPEG Stream (x=5.1 -> x=3.7 at y=3.5)
    ax.annotate("", xy=(3.7, 3.5), xytext=(5.1, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6, mutation_scale=12))
    ax.text(4.4, 3.15, "MJPEG Stream (Live HUD)", fontsize=7.8, ha="center", va="center", color="#1a5276", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#2980b9", lw=0.9))

    # Top: Stream A (Forensic) (x in [4.9, 9.1], y in [6.3, 7.9])
    draw_box(4.9, 6.3, 4.2, 1.6, "Stream A: Forensic Worker", "CLIP ViT-B/32 + Bi-LSTM Model\nTemporal Attention Context", bg="#e8f8f5", border="#16a085", tc="#117864")
    
    # Vertical arrows between Hub and Stream A:
    ax.annotate("", xy=(6.0, 6.3), xytext=(6.0, 5.2),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6, mutation_scale=12))
    ax.text(5.15, 5.75, "16 Frames\nBatch", fontsize=7.5, ha="center", va="center", color="#117864", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffff", edgecolor="#16a085", lw=0.9))

    ax.annotate("", xy=(8.0, 5.2), xytext=(8.0, 6.3),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6, mutation_scale=12))
    ax.text(8.85, 5.75, "Softmax\nVerdict", fontsize=7.5, ha="center", va="center", color="#117864", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffff", edgecolor="#16a085", lw=0.9))

    # Bottom: Stream B (Surveillance) (x in [4.9, 9.1], y in [0.4, 2.0])
    draw_box(4.9, 0.4, 4.2, 1.6, "Stream B: Spatial Worker", "YOLOv8 Object Tracker\nRay-Casting Polygon Intersection", bg="#fef5e7", border="#e67e22", tc="#b9770e")

    # Vertical arrows between Hub and Stream B:
    ax.annotate("", xy=(6.0, 2.0), xytext=(6.0, 2.8),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6, mutation_scale=12))
    ax.text(5.15, 2.40, "Live Frame\nStream", fontsize=7.5, ha="center", va="center", color="#b9770e", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffff", edgecolor="#e67e22", lw=0.9))

    ax.annotate("", xy=(8.0, 2.8), xytext=(8.0, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6, mutation_scale=12))
    ax.text(8.85, 2.40, "Breach\nPayload", fontsize=7.5, ha="center", va="center", color="#b9770e", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffff", edgecolor="#e67e22", lw=0.9))

    # Right Top: Telegram Service (x in [11.4, 14.6], y in [4.8, 6.6])
    draw_box(11.4, 4.8, 3.2, 1.8, "Telegram Notification API", "Async Bot Alert Dispatcher\nBreach Event + Intruder ID", bg="#fadbd8", border="#e74c3c", tc="#78281f")
    # Clean connector line from Hub right edge (8.9, 4.5) to Telegram left edge (11.4, 5.7)
    # The gap between Hub (8.9) and Telegram (11.4) is a generous 2.5 units!
    ax.annotate("", xy=(11.4, 5.7), xytext=(8.9, 4.5),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8, mutation_scale=12))
    # Centered at x=10.15, leaving 0.5+ units of clearance to BOTH boxes (ZERO OVERLAP!)
    ax.text(10.15, 5.10, "HTTPS Alert Dispatch\n(Breach Alert + Timestamp)", fontsize=7.5, ha="center", va="center", color="#78281f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#e74c3c", lw=1.0))

    # Right Bottom: SQLite Database (x in [11.4, 14.6], y in [1.6, 3.4])
    draw_box(11.4, 1.6, 3.2, 1.8, "SQLite Persistence", "surveillance.db Table Manager\nAudit Logging & Event History", bg="#f4ecf7", border="#8e44ad", tc="#512e5f")
    # Clean connector line from Hub right edge (8.9, 3.5) to SQLite left edge (11.4, 2.5)
    ax.annotate("", xy=(11.4, 2.5), xytext=(8.9, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=1.8, mutation_scale=12))
    # Centered at x=10.15, leaving 0.5+ units of clearance to BOTH boxes (ZERO OVERLAP!)
    ax.text(10.15, 3.00, "SQL INSERT\n(Event Logging)", fontsize=7.5, ha="center", va="center", color="#512e5f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#8e44ad", lw=1.0))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "integration_design_flow.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved integration_design_flow.png (with pristine zero-overlap layout).")

# -----------------------------------------------------------------------------
# 8. DETAILED OPERATIONAL WORKFLOW (Checklist Point 13 - Slide 16)
# -----------------------------------------------------------------------------
def make_operational_workflow():
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)

    steps = [
        ("Step 1: System Activation", "Operator accesses Mode 2 HUD and initializes real-time video stream with configured virtual perimeter zone.", "#ebf5fb", "#2980b9"),
        ("Step 2: Stream Ingestion", "Camera stream is captured frame-by-frame; Aspect-ratio letterboxing is applied to prevent optical warping.", "#e8f8f5", "#16a085"),
        ("Step 3: Spatial Tracking", "YOLOv8 scans incoming frames (45+ FPS) to isolate human bodies with bounding boxes and unique track IDs.", "#fef9e7", "#f39c12"),
        ("Step 4: Geometric Intersection", "Calculates bottom-center anchor point (x_c, y_max); cv2.pointPolygonTest tests containment inside virtual fence.", "#fef5e7", "#e67e22"),
        ("Step 5: Alert & Archival", "Breach detected: HUD illuminates RED; SQLite logs security_events record; Telegram bot dispatches instant mobile alert.", "#fadbd8", "#e74c3c"),
        ("Step 6: Forensic Escalation", "Incident video clip can be routed to Mode 1: CLIP + BiLSTM runs temporal action classification to determine threat intent.", "#f4ecf7", "#8e44ad"),
        ("Step 7: Automated Dossier", "System generates high-level PDF intelligence dossier with confusion diagnostics and timestamps for commander review.", "#d4efdf", "#27ae60")
    ]

    for i, (stitle, sdesc, bg, border) in enumerate(steps):
        y = 7.1 - i * 1.05
        badge = patches.Circle((1.5, y + 0.38), 0.36, edgecolor=border, facecolor=border)
        ax.add_patch(badge)
        ax.text(1.5, y + 0.38, str(i+1), fontsize=11, fontweight="bold", color="#ffffff", ha="center", va="center")

        box = patches.FancyBboxPatch((2.2, y), 12.0, 0.76, boxstyle="round,pad=0.1", edgecolor=border, facecolor=bg, lw=1.5)
        ax.add_patch(box)
        ax.text(2.5, y + 0.50, stitle, fontsize=10.5, fontweight="bold", color="#1a5276")
        ax.text(2.5, y + 0.20, sdesc, fontsize=8.5, color="#2c3e50")

        if i < len(steps) - 1:
            ax.plot([1.5, 1.5], [y, y - 0.28], color="#bdc3c7", lw=2, linestyle=":")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "operational_workflow_diagram.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved operational_workflow_diagram.png")

# -----------------------------------------------------------------------------
# 9. DEVELOPMENT PLAN & ROADMAP (Checklist Point 15 - Slide 19)
# -----------------------------------------------------------------------------
def make_development_plan():
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)

    phases = [
        ("Phase I: Project Inception & Architectural Formulation", "July 2026", 
         "Surveillance fatigue analysis, dual-stream architecture blueprinting, technical feasibility & dataset requirement mapping.", 
         "#1b4f72", "#ebf5fb", 1.0),
        
        ("Phase II: Dataset Assembly, Letterboxing & YOLO-Gating", "August 2026", 
         "Curated 1,050 balanced videos (150/class) across 7 classes; eliminated traffic domain shift; extracted 16,800 YOLO-gated frames.", 
         "#27ae60", "#d4efdf", 1.0),
        
        ("Phase III: Deep Learning Modeling & Attention Architecture", "September 2026", 
         "OpenAI CLIP ViT-B/32 feature serialization (16x512 .npy); Bi-LSTM with Temporal Attention + Tri-Pooling (1536-d); 98.10% test accuracy.", 
         "#b9770e", "#fef5e7", 1.0),
        
        ("Phase IV: Spatial Virtual Fence, SQLite & Telegram Bot", "Late Sep - Early Oct 2026", 
         "Ultralytics YOLOv8s person tracking (45+ FPS); point-in-polygon breach algorithm; SQLite (surveillance.db) logging; Telegram push alerts (<1.2s).", 
         "#8e44ad", "#f4ecf7", 1.0),
        
        ("Phase V: Unified Web Dashboard & Final System Delivery", "October 2026", 
         "Centralized Flask hub (app.py); Mode 1 & Mode 2 dual-panel Web HUD (index.html); automated FPDF forensic dossier generator; 100% verified.", 
         "#117864", "#e8f8f5", 1.0)
    ]

    for i, (pname, pdate, pdesc, border, bg, progress) in enumerate(phases):
        y = 6.6 - i * 1.4
        box = patches.FancyBboxPatch((1.0, y), 13.0, 1.15, boxstyle="round,pad=0.15", edgecolor=border, facecolor=bg, lw=1.6)
        ax.add_patch(box)
        
        ax.text(1.3, y + 0.78, pname, fontsize=11, fontweight="bold", color="#1a5276")
        ax.text(13.7, y + 0.78, pdate, fontsize=10.5, fontweight="bold", color=border, ha="right")
        ax.text(1.3, y + 0.35, pdesc, fontsize=8.5, color="#2c3e50")

        prog_w = 2.0
        ax.plot([1.3, 1.3 + prog_w], [y + 0.14, y + 0.14], color="#27ae60", lw=4, solid_capstyle="round")
        ax.text(1.3 + prog_w + 0.2, y + 0.14, "Status: 100% Completed", fontsize=7.8, color="#145a32", fontweight="bold", va="center")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "development_plan_roadmap.png"), bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved development_plan_roadmap.png")

if __name__ == "__main__":
    print("[*] Re-generating all design diagrams with perfect layout & text-wrapping...")
    make_overall_architecture()
    make_system_process_flow()
    make_uml_diagrams()
    make_data_design()
    make_hardware_software_stack()
    make_technology_selection_matrix()
    make_integration_design()
    make_operational_workflow()
    make_development_plan()
    print("[SUCCESS] All design diagrams re-generated cleanly!")
