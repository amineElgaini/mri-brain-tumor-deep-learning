import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.inference import CLASSES, load_classifier, load_detector, predict_class, detect

COLORS = {"Glioma": "#FF6B5B", "Meningioma": "#F5B83D", "No Tumor": "#7BD88F", "Pituitary": "#45C8E8"}
SAMPLE_DIR = ROOT / "dashboard" / "samples"
SAMPLES = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if SAMPLE_DIR.exists() else []

st.set_page_config(page_title="Brain MRI, read two ways", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@400;500;700&display=swap');
header[data-testid="stHeader"], #MainMenu, footer { display: none; }.stApp { background: #0F1B24; color: #E6EEF2; }
.stApp { background: #0F1B24; color: #E6EEF2; }
header[data-testid="stHeader"], #MainMenu, footer { display: none; }
.block-container { padding-top: 2.4rem; max-width: 1180px; }

.hero { font-family: 'Instrument Serif', serif !important; font-weight: 400; font-size: clamp(2.8rem, 6vw, 4.6rem);
          line-height: 1.02; letter-spacing: -0.01em; margin: 0 0 .7rem; }
.lede { color: #8FA6B3; font-size: 1.1rem; line-height: 1.5; max-width: 36rem; margin: 0 0 1rem; }
.notice { display: inline-block; border: 1px solid #2A404D; border-radius: 999px; padding: .3rem .85rem;
          font-size: .85rem; color: #8FA6B3; margin-bottom: 1.4rem; }
.section { font-size: 1.05rem; font-weight: 700; margin: 1.6rem 0 .3rem; }
.sub { color: #8FA6B3; margin: .3rem 0 1.2rem; }

[data-testid="stFileUploaderDropzone"] { background: #16262F; border: 1px dashed #3A5666; border-radius: 6px; }
button[data-baseweb="tab"] { font-size: 1rem; }

.viewer { position: relative; background: #000; border: 1px solid #2A404D; border-radius: 6px; overflow: hidden; }
.viewer img { display: block; width: 100%; }
.viewer span { position: absolute; padding: .6rem .8rem; font-size: .78rem; color: #CFE0E8; text-shadow: 0 1px 3px #000; }
.viewer .tl { top: 0; left: 0; }  .viewer .tr { top: 0; right: 0; }
.viewer .bl { bottom: 0; left: 0; } .viewer .br { bottom: 0; right: 0; }

.verdict { font-family: 'Instrument Serif', serif !important; font-size: clamp(3rem, 5vw, 4.4rem); line-height: 1; }
.row { display: grid; grid-template-columns: 6.5rem 1fr 3.6rem; align-items: center; gap: .8rem; margin: .55rem 0; opacity: .6; }
.row.top { opacity: 1; }
.track { height: 10px; background: #16262F; border-radius: 5px; overflow: hidden; }
.fill { height: 100%; border-radius: 5px; transform-origin: left; }
.pc { text-align: right; font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: no-preference) { .fill { animation: grow .9s ease-out; } }
@keyframes grow { from { transform: scaleX(0); } }

.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .5rem; }
.chip { display: inline-flex; align-items: center; gap: .45rem; border: 1px solid; border-radius: 999px; padding: .25rem .75rem; font-size: .9rem; }
.dot { display: inline-block; width: .6rem; height: .6rem; border-radius: 50%; margin-right: .5rem; }
.chip .dot { margin-right: 0; }

table.res { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.res th { text-align: left; font-weight: 500; color: #8FA6B3; padding: .5rem .6rem; border-bottom: 1px solid #2A404D; }
.res td { padding: .6rem; border-bottom: 1px solid #1E313C; }
.res .n { text-align: right; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return load_classifier(), load_detector()


def draw_boxes(image, boxes):
    out = image.copy()
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(14, image.width // 32))
    lw = max(2, image.width // 170)
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        color = COLORS[b["class"]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{b['class']} {b['conf']:.2f}"
        l, t, r, bt = d.textbbox((0, 0), label, font=font)
        top = max(0, y1 - (bt - t) - 10)
        d.rectangle([x1, top, x1 + (r - l) + 12, top + (bt - t) + 10], fill=color)
        d.text((x1 + 6, top + 5 - t), label, fill="#0F1B24", font=font)
    return out


def data_uri(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def viewer(shown, boxes, conf):
    plural = "" if len(boxes) == 1 else "s"
    return (
        f"<div class='viewer'><img src='{data_uri(shown)}' alt='MRI slice with detected regions'>"
        f"<span class='tl'>{shown.width} × {shown.height} px</span>"
        f"<span class='tr'>Threshold {conf:.2f}</span>"
        f"<span class='bl'>{len(boxes)} region{plural} found</span>"
        f"<span class='br'>YOLOv8n detector</span></div>"
    )


def reading(probs):
    top = int(probs.argmax())
    name = CLASSES[top]
    rows = ""
    for i, c in enumerate(CLASSES):
        cls = "row top" if i == top else "row"
        rows += (
            f"<div class='{cls}'><span>{c}</span>"
            f"<div class='track'><div class='fill' style='width:{probs[i] * 100:.1f}%;background:{COLORS[c]}'></div></div>"
            f"<span class='pc'>{probs[i]:.1%}</span></div>"
        )
    return (
        f"<div class='verdict' style='color:{COLORS[name]}'>{name}</div>"
        f"<div class='sub'>The classifier puts {probs[top]:.1%} on this class</div>{rows}"
    )


def chips(boxes):
    items = "".join(
        f"<span class='chip' style='border-color:{COLORS[b['class']]}'>"
        f"<i class='dot' style='background:{COLORS[b['class']]}'></i>{b['class']} {b['conf']:.0%}</span>"
        for b in boxes
    )
    return f"<div class='chips'>{items}</div>"


def table(headers, rows):
    head = "<th>Class</th>" + "".join(f"<th class='n'>{h}</th>" for h in headers)
    body = ""
    for name, vals in rows:
        cells = "".join(f"<td class='n'>{v:.3f}</td>" for v in vals)
        body += f"<tr><td><i class='dot' style='background:{COLORS[name]}'></i>{name}</td>{cells}</tr>"
    return f"<table class='res'><tr>{head}</tr>{body}</table>"


clf, det = get_models()

st.markdown(
    "<div class='hero' role='heading' aria-level='1'>Brain MRI, read two ways</div>"
    "<p class='lede'>One model names the tumor type. Another draws where it sees one.</p>"
    "<span class='notice'>Educational prototype. Not a medical device, not for diagnosis.</span>",
    unsafe_allow_html=True,
)

tab_analyze, tab_results = st.tabs(["Analyze a scan", "How well it works"])

with tab_analyze:
    c1, c2 = st.columns([3, 2], gap="large")
    file = c1.file_uploader("Drop an MRI slice here", type=["jpg", "jpeg", "png"])
    sample = None
    if SAMPLES:
        names = {p.stem.replace("_", " ").capitalize(): p for p in SAMPLES}
        pick = c2.selectbox("Or try a sample", list(names), index=None, placeholder="Pick a sample")
        sample = names.get(pick)
    conf = c2.slider("Detection threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Lower finds fainter regions. Higher keeps only confident ones.")

    image = None
    if file:
        image = Image.open(file).convert("RGB")
    elif sample:
        image = Image.open(sample).convert("RGB")

    if image is None:
        st.markdown("<p class='lede'>Upload a slice or pick a sample to see both models at work.</p>",
                    unsafe_allow_html=True)
    else:
        probs = predict_class(clf, image)
        boxes = detect(det, image, conf)

        left, right = st.columns([3, 2], gap="large")
        left.markdown(viewer(draw_boxes(image, boxes), boxes, conf), unsafe_allow_html=True)
        with right:
            st.markdown(reading(probs), unsafe_allow_html=True)
            st.markdown("<div class='section'>Regions the detector found</div>", unsafe_allow_html=True)
            if boxes:
                st.markdown(chips(boxes), unsafe_allow_html=True)
            else:
                st.markdown("<p class='sub'>No region found at this threshold. Lower it to look for fainter regions.</p>",
                            unsafe_allow_html=True)
            if CLASSES[int(probs.argmax())] == "No Tumor" and boxes:
                st.info("The two models disagree here. The classifier read No Tumor, but the detector found a region.")

with tab_results:
    path = ROOT / "results" / "metrics.json"
    if not path.exists():
        st.info("results/metrics.json is missing. Run the evaluation to create it.")
    else:
        m = json.loads(path.read_text())
        a, b = st.columns(2, gap="large")

        with a:
            k = m["classifier"]
            st.markdown("<div class='section'>Classifier on the test set</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='sub'>{k['images']} images. Accuracy {k['accuracy']:.1%}, macro F1 {k['macro_f1']:.3f}.</p>",
                        unsafe_allow_html=True)
            rows = [(c, [v["precision"], v["recall"], v["f1"]]) for c, v in k["per_class"].items()]
            st.markdown(table(["Precision", "Recall", "F1"], rows), unsafe_allow_html=True)
            img = ROOT / "results" / "confusion_matrix_classifier_test.png"
            if img.exists():
                st.image(str(img), width="stretch")

        with b:
            k = m["detector"]
            st.markdown("<div class='section'>Detector on the test set</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='sub'>{k['images']} images. mAP50 {k['map50']:.3f}, mAP50-95 {k['map50_95']:.3f}.</p>",
                        unsafe_allow_html=True)
            rows = [(c, [v]) for c, v in k["per_class_map50"].items()]
            st.markdown(table(["mAP50"], rows), unsafe_allow_html=True)
            st.markdown("<p class='sub'>No Tumor images are used by the classifier only. Their boxes cover the whole brain, "
                        "so the detector learns the three tumor classes.</p>", unsafe_allow_html=True)
            img = ROOT / "results" / "yolo_confusion_matrix_normalized.png"
            if img.exists():
                st.image(str(img), width="stretch")