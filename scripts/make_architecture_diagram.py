"""Generate solutions/architecture_diagram.png."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT = Path(__file__).resolve().parent.parent / "solutions" / "architecture_diagram.png"
OUTPUT.parent.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_AWS     = "#FF9900"   # AWS orange
C_AWSBG   = "#FFF3E0"
C_DATA    = "#1565C0"   # blue – data / S3
C_DATABG  = "#E3F2FD"
C_MODEL   = "#2E7D32"   # green – ML
C_MODELBG = "#E8F5E9"
C_APP     = "#6A1B9A"   # purple – app
C_APPBG   = "#F3E5F5"
C_USER    = "#37474F"
C_USERBG  = "#ECEFF1"
C_ARROW   = "#455A64"

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── helpers ───────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel="", facecolor="#ECEFF1", edgecolor="#90A4AE",
        fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
                          zorder=2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    cy = y + h / 2 + (0.12 if sublabel else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#212121", zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel, ha="center", va="center",
                fontsize=7.5, color="#555555", zorder=3, style="italic")


def group(ax, x, y, w, h, title, facecolor="#FFF8E1", edgecolor="#FFB300", lw=1.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
                          zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.18, y + h - 0.22, title, ha="left", va="top",
            fontsize=8.5, fontweight="bold", color=edgecolor, zorder=3)


def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=14),
                zorder=4)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom",
                fontsize=7, color=color, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Layout (left → right flow)
# ══════════════════════════════════════════════════════════════════════════════

# ── Developer / Local ─────────────────────────────────────────────────────────
group(ax, 0.3, 6.6, 3.6, 4.0, "Local / Developer", C_USERBG, C_USER, lw=1.2)
box(ax, 0.55, 9.0, 3.1, 0.9,  "Raw CSV data",      "bronze / silver / gold parquet",
    C_DATABG, C_DATA, bold=False)
box(ax, 0.55, 7.8, 3.1, 0.9,  "XGBoost Training",  "notebooks / train.py",
    C_MODELBG, C_MODEL, bold=False)
box(ax, 0.55, 6.75, 3.1, 0.8, "model_metadata.json\nxgboost_bath_predictor.json",
    sublabel="", facecolor=C_MODELBG, edgecolor=C_MODEL, fontsize=7.8)

# ── AWS cloud group ───────────────────────────────────────────────────────────
group(ax, 4.3, 0.4, 13.4, 10.2, "AWS (eu-west-1)", C_AWSBG, C_AWS, lw=2)

# S3
box(ax, 4.6, 8.7,  2.8, 1.0, "S3 Bucket",        "vaultech-lorenzo\nmodel artefacts",
    C_DATABG, C_DATA)

# ECR
box(ax, 4.6, 6.9,  2.8, 1.4, "Amazon ECR",
    "vaultech-serving\nvaultech-streamlit",
    C_AWSBG, C_AWS)

# SageMaker group
group(ax, 7.8, 5.2, 5.6, 5.1, "Amazon SageMaker", C_MODELBG, C_MODEL, lw=1.5)
box(ax, 8.1, 9.0, 2.4, 0.9,  "Model Registry",   "ModelPackageGroup", C_MODELBG, C_MODEL)
box(ax, 10.7, 9.0, 2.4, 0.9, "Model Package",    "v1 — Approved", C_MODELBG, C_MODEL)
box(ax, 8.1, 7.5, 4.8, 1.0,  "Real-time Endpoint",
    "vaultech-bath-predictor  (ml.m5.large)", C_MODELBG, C_MODEL, bold=True)
box(ax, 8.1, 6.0, 4.8, 1.0,  "Serving Container",
    "Flask + gunicorn  (xgboost 1.7)", C_AWSBG, C_AWS)
box(ax, 8.1, 5.4, 4.8, 0.5,  "IAM: VaultechSageMakerRole", sublabel="",
    facecolor="#FFF8E1", edgecolor="#FFB300", fontsize=7.5)

# ECS group
group(ax, 7.8, 0.6, 5.6, 4.2, "Amazon ECS / Fargate", C_APPBG, C_APP, lw=1.5)
box(ax, 8.1, 3.4, 4.8, 1.0,  "Streamlit App",
    "vaultech-streamlit  (port 8501)", C_APPBG, C_APP, bold=True)
box(ax, 8.1, 2.2, 4.8, 0.9,  "Task Definition",
    "512 CPU / 1024 MB  ·  awsvpc", C_APPBG, C_APP)
box(ax, 8.1, 1.1, 4.8, 0.8,  "Security Group  +  Public IP",
    "TCP 8501 open  ·  SAGEMAKER_ENDPOINT_NAME env", C_APPBG, C_APP, fontsize=7.5)
box(ax, 8.1, 0.7, 2.2, 0.35, "IAM: VaultechECSTaskRole", sublabel="",
    facecolor="#F3E5F5", edgecolor=C_APP, fontsize=7)
box(ax, 10.5, 0.7, 2.4, 0.35, "IAM: ecsTaskExecutionRole", sublabel="",
    facecolor="#F3E5F5", edgecolor=C_APP, fontsize=7)

# CloudWatch
box(ax, 13.6, 6.9, 3.7, 1.4, "CloudWatch Logs",
    "/ecs/vaultech-streamlit\ncontainer stdout / stderr",
    C_AWSBG, C_AWS)

# User browser
box(ax, 13.6, 3.9, 3.7, 1.2, "User Browser",
    "http://<public-ip>:8501", C_USERBG, C_USER, bold=True)

# ── Arrows ────────────────────────────────────────────────────────────────────
# Data pipeline
arrow(ax, 2.1, 9.0,  2.1, 8.7, "ETL")           # CSV → training
arrow(ax, 2.1, 7.8,  2.1, 7.55, "train")         # training → model files

# Upload to S3
arrow(ax, 3.65, 8.9,  4.6, 9.0, "upload artefacts")

# ECR pushes
arrow(ax, 3.65, 7.2,  4.6, 7.5, "docker push")

# S3 → SageMaker (model download)
arrow(ax, 6.15, 9.1,  7.8, 9.2, "s3:GetObject")

# Model Registry
arrow(ax, 9.3, 9.9,  10.7, 9.9, "register")      # registry → package (same row)
arrow(ax, 10.7, 9.0, 10.7, 8.5, "deploy")         # package → endpoint
arrow(ax, 9.5,  8.5, 9.5, 8.0, "")                # endpoint ↓

# ECR → Serving container
arrow(ax, 7.4, 7.2,  8.1, 6.5, "pull image")

# Serving container → endpoint
arrow(ax, 10.5, 6.5, 10.5, 7.5, "")

# ECR → ECS
arrow(ax, 7.4, 6.9,  8.1, 3.9, "pull image")

# ECS → SageMaker invoke
arrow(ax, 13.0, 3.9, 13.3, 8.0,
      "InvokeEndpoint\n(text/csv)", color=C_MODEL)

# ECS → CloudWatch
arrow(ax, 13.0, 3.7, 13.6, 7.5, "logs", color=C_AWS)

# User → ECS
arrow(ax, 15.45, 3.9, 13.0, 3.9, "HTTP :8501")

# deploy scripts
ax.text(1.6, 6.5, "deploy/deploy_sagemaker.py\ndeploy/deploy_ecs.py",
        ha="center", va="top", fontsize=7, color="#78909C", style="italic")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9.0, 10.75, "VaultTech — Forging Line Bath-Time Predictor  |  Cloud Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#212121")

plt.tight_layout(pad=0.3)
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT}")
