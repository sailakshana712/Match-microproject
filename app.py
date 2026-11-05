"""
Reduce Dimensionality of Images While Preserving the Most Important Features
Maths Micro Project - Principal Component Analysis (PCA) on images

Run:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Image PCA Lab", page_icon="🧬", layout="wide")

MAX_FILE_MB = 5
MAX_SIDE = 600  # large images are scaled down so PCA stays fast

INK = "#0E2A3B"
TEAL = "#1F8A9E"
AMBER = "#F2A541"
PAGES = [
    "🏠 Home",
    "📤 Upload",
    "🔬 PCA & Comparison",
    "📊 Results",
    "📐 Mathematics",
    "📈 Visualization",
    "ℹ️ About & Conclusion",
]

CSS = f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hero {{
    background: linear-gradient(135deg, {INK} 0%, #174B63 100%);
    border-radius: 18px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.4rem;
    border-bottom: 5px solid {AMBER};
}}
.hero h1 {{ color: #fff; font-size: 2.3rem; line-height: 1.2; margin: 0 0 .6rem 0; }}
.hero p {{ color: #d7e6ee; font-size: 1.08rem; margin: 0; max-width: 760px; }}
.card {{
    border: 1px solid rgba(128,128,128,.28); border-left: 5px solid {TEAL};
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    background: rgba(31,138,158,.06);
}}
.card h4 {{ margin: 0 0 .3rem 0; }}
.card p {{ margin: 0; opacity: .9; }}
.member {{
    border: 1px solid rgba(128,128,128,.28); border-top: 4px solid {AMBER};
    border-radius: 12px; padding: .9rem 1.1rem; height: 100%;
}}
.member b {{ display: block; margin-bottom: .2rem; }}
.member span {{ opacity: .85; font-size: .93rem; }}
div[data-testid="stMetric"] {{
    border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .7rem 1rem;
}}
</style>
"""


# ----------------------------------------------------------------------------
# Core PCA logic (real mathematics, NumPy only)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running PCA (SVD) on the image...")
def run_pca(arr: np.ndarray):
    """
    arr : float array of shape (H, W, C), values 0..255.

    Each colour channel is an H x W matrix. Every pixel ROW is treated as one
    observation with W features. For every channel we:
        1. compute the mean of each column (feature mean)
        2. mean-center the matrix:      Xc = X - mean
        3. SVD:                         Xc = U S Vt
           -> rows of Vt are principal directions (eigenvectors of covariance)
           -> S^2 / (H - 1) are the eigenvalues of the covariance matrix
    Returns a list of (mean, U, S, Vt), one tuple per channel.
    """
    models = []
    for c in range(arr.shape[2]):
        X = arr[:, :, c].astype(np.float64)
        mean = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
        models.append((mean, U, S, Vt))
    return models


def reconstruct(models, k: int) -> np.ndarray:
    """Rebuild the image from the first k principal components.
       X_reduced = Xc W_k  (= U_k S_k)     X_hat = X_reduced W_k^T + mean"""
    channels = []
    for mean, U, S, Vt in models:
        reduced = U[:, :k] * S[:k]  # scores, shape (H, k)
        channels.append(reduced @ Vt[:k] + mean)
    return np.clip(np.stack(channels, axis=-1), 0, 255)


def explained_variance_ratio(models) -> np.ndarray:
    """Pooled over channels: share of total variance carried by each component."""
    var = sum(S ** 2 for _, _, S, _ in models)
    return var / var.sum()


def mse_curve(models, shape) -> np.ndarray:
    """Exact MSE (before 0-255 clipping) for k = 0..r using discarded eigenvalues."""
    H, W, C = shape
    tail = 0
    for _, _, S, _ in models:
        sq = S ** 2
        # tail[k] = sum of sq[k:]  (energy that is thrown away when keeping k comps)
        tail = tail + np.concatenate([np.cumsum(sq[::-1])[::-1], [0.0]])
    return tail / (H * W * C)


def compression_stats(shape, k: int):
    """Storage assumption: one stored number = one pixel value.
       Original  : H * W * C numbers
       Compressed: per channel  k*H (scores) + k*W (components) + W (mean)"""
    H, W, C = shape
    original = H * W * C
    compressed = C * (k * H + k * W + W)
    return original, compressed, original / compressed, 1 - compressed / original


def compute_metrics(original: np.ndarray, rec: np.ndarray) -> dict:
    mse = float(np.mean((original - rec) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return {"mse": mse, "psnr": psnr}


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def load_image(file):
    """Returns (image scaled for PCA, original (w, h))."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    orig_size = img.size
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img, orig_size


def make_sample_image() -> Image.Image:
    """Synthetic demo image so the app works without uploading anything."""
    W, H = 480, 360
    yy, xx = np.mgrid[0:H, 0:W]
    sky = np.stack([40 + 120 * yy / H, 90 + 110 * yy / H, 170 + 70 * yy / H], -1)
    img = Image.fromarray(sky.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse((350, 40, 430, 120), fill=(250, 210, 90))
    d.polygon([(0, 360), (110, 170), (220, 360)], fill=(60, 80, 70))
    d.polygon([(120, 360), (270, 130), (440, 360)], fill=(45, 65, 60))
    d.polygon([(270, 130), (250, 165), (270, 155), (290, 170)], fill=(245, 245, 250))
    d.rectangle((0, 300, 480, 360), fill=(35, 55, 45))
    for x in range(20, 460, 55):
        d.polygon([(x, 330), (x + 14, 285), (x + 28, 330)], fill=(25, 90, 55))
    d.rectangle((200, 255, 260, 300), fill=(150, 90, 60))
    d.polygon([(192, 255), (230, 220), (268, 255)], fill=(120, 40, 40))
    d.rectangle((222, 272, 238, 300), fill=(250, 220, 120))
    return img


def to_array(img: Image.Image, grayscale: bool) -> np.ndarray:
    if grayscale:
        return np.asarray(img.convert("L"), dtype=np.float64)[:, :, None]
    return np.asarray(img, dtype=np.float64)


def to_display(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return arr[:, :, 0] if arr.shape[2] == 1 else arr


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(to_display(arr)).save(buf, format="PNG")
    return buf.getvalue()


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def need_image(ctx):
    if ctx is None:
        st.warning("No image yet. Go to **📤 Upload** and add a photo, or load the sample image.")
        st.button("Go to Upload", on_click=lambda: st.session_state.update(page=PAGES[1]))
        st.stop()


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1>Shrink an image. Keep what matters.</h1>
          <p>Principal Component Analysis (PCA) finds the directions that carry the most
          information in an image, so we can store far fewer numbers and still rebuild a
          picture that looks almost the same. Upload a photo and see it happen live.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("Get started: upload an image", type="primary",
              on_click=lambda: st.session_state.update(page=PAGES[1]))

    st.subheader("What is dimensionality reduction?")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("The problem", "A 600 x 400 colour photo is 720,000 numbers. Most of them repeat "
             "the same patterns, so much of that data is redundant.")
    with c2:
        card("The idea", "Rotate the data to new axes (principal components) ordered by how much "
             "variance they capture, then keep only the first k.")
    with c3:
        card("The payoff", "Less storage and faster computation, with the strongest features "
             "of the image preserved. This is used in face recognition and medical imaging.")

    st.subheader("Why PCA?")
    st.markdown(
        "- It keeps the **highest-variance** directions, which hold the most visible structure.\n"
        "- It is **optimal** in the least-squares sense: no other k-dimensional linear "
        "representation has lower reconstruction error.\n"
        "- It is a standard pre-processing step in computer vision and machine learning pipelines."
    )

    st.subheader("Project team")
    team = [
        ("Member 1: Frontend", "Interface, navigation, comparison screens"),
        ("Member 2: Backend & PCA", "Image processing, PCA, reconstruction"),
        ("Member 3: Maths & Visualization", "Metrics, graphs, formulas, testing"),
        ("Member 4: Integration & Deployment", "Repository, deployment, documentation"),
    ]
    cols = st.columns(4)
    for col, (name, role) in zip(cols, team):
        col.markdown(f'<div class="member"><b>{name}</b><span>{role}</span></div>',
                     unsafe_allow_html=True)
    st.caption("Edit the team list in `page_home()` to add your names.")


def page_upload():
    st.title("📤 Upload an image")
    st.write(f"Accepted formats: JPG, JPEG, PNG. Maximum size: {MAX_FILE_MB} MB. "
             f"Images larger than {MAX_SIDE}px are scaled down so PCA stays quick.")

    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if up is not None:
        if up.size > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is {up.size / 1e6:.1f} MB. Please upload an image under {MAX_FILE_MB} MB.")
        else:
            try:
                if st.session_state.get("source") != up.file_id:
                    img, orig_size = load_image(up)
                    st.session_state.img = img
                    st.session_state.source = up.file_id
                    st.session_state.orig_size = orig_size
                    st.session_state.name = up.name
                    st.rerun()  # refresh so the sidebar slider appears
            except Exception as e:
                st.error(f"Could not read this file as an image ({e}). Try another JPG or PNG.")

    if st.button("Use the sample image instead"):
        st.session_state.img = make_sample_image()
        st.session_state.source = "sample"
        st.session_state.orig_size = st.session_state.img.size
        st.session_state.name = "sample image"
        st.rerun()

    img = st.session_state.get("img")
    if img is None:
        st.info("Upload an image or load the sample to begin.")
        return

    left, right = st.columns([2, 1])
    left.image(img, caption=st.session_state.get("name", ""), use_container_width=True)
    ow, oh = st.session_state.get("orig_size", img.size)
    w, h = img.size
    right.metric("Original size", f"{ow} x {oh} px")
    right.metric("Size used for PCA", f"{w} x {h} px")
    right.metric("Maximum components", f"{min(w, h)}")
    right.success("Image ready. Open **🔬 PCA & Comparison** and move the slider in the sidebar.")


def page_pca(ctx):
    st.title("🔬 PCA & Comparison")
    need_image(ctx)
    models, arr, k = ctx["models"], ctx["arr"], ctx["k"]
    rec = ctx["rec"]
    H, W, C = arr.shape
    m = ctx["metrics"]
    retained = ctx["evr"][:k].sum() * 100
    _, _, ratio, saving = ctx["comp"]

    st.write(f"Using **k = {k}** of {min(H, W)} principal components. "
             "Change **k** in the sidebar and the result updates instantly.")

    a, b = st.columns(2)
    a.image(to_display(arr), caption=f"Original ({W} x {H})", use_container_width=True)
    b.image(to_display(rec), caption=f"Reconstructed with {k} components", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Information retained", f"{retained:.2f}%")
    c2.metric("MSE", f"{m['mse']:.2f}")
    c3.metric("PSNR", "inf" if np.isinf(m["psnr"]) else f"{m['psnr']:.1f} dB")
    c4.metric("Compression ratio", f"{ratio:.2f} : 1")

    st.subheader("What changed?")
    if retained >= 99:
        msg = "Nearly all detail is kept. The two images look almost identical to the eye."
    elif retained >= 95:
        msg = "Edges and main shapes are sharp. Only very fine texture is lost."
    elif retained >= 85:
        msg = "The overall scene is clear but fine detail looks soft or slightly streaky."
    else:
        msg = ("Only the coarsest structure survives: broad shapes and colour regions. "
               "Streaks appear because each component captures one direction of pixel columns.")
    st.info(msg)

    with st.expander("Show the difference image (error amplified x4)"):
        diff = np.clip(np.abs(arr - rec) * 4, 0, 255)
        st.image(to_display(diff), use_container_width=True)
        st.caption("Brighter areas are where the reconstruction differs most from the original.")

    st.download_button("Download reconstructed image (PNG)", png_bytes(rec),
                       file_name=f"reconstructed_k{k}.png", mime="image/png")


def page_results(ctx):
    st.title("📊 Results & Analysis")
    need_image(ctx)
    arr, k = ctx["arr"], ctx["k"]
    H, W, C = arr.shape
    original, compressed, ratio, saving = ctx["comp"]
    retained = ctx["evr"][:k].sum() * 100
    m = ctx["metrics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Original dimensions", f"{H} x {W} x {C}")
    c2.metric("Reduced dimensions", f"{H} x {k} (scores) per channel")
    c3.metric("Principal components kept", f"{k} / {min(H, W)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Explained variance ratio (cumulative)", f"{retained:.3f}%")
    c2.metric("Mean Squared Error", f"{m['mse']:.3f}")
    c3.metric("Compression ratio", f"{ratio:.2f} : 1")

    if ratio < 1:
        st.warning("With this many components the PCA representation is **larger** than the "
                   "original image. Lower k to actually compress.")
    else:
        st.success(f"Storage drops from {original:,} to {compressed:,} numbers "
                   f"({saving * 100:.1f}% saved).")

    st.caption("Storage assumption: one stored number = one pixel value. PCA stores, per channel, "
               "the scores (H x k), the components (k x W) and the mean row (W). "
               "Compression ratio = (H·W·C) / (C·(kH + kW + W)).")

    st.subheader("How the number of components changes everything")
    r = min(H, W)
    ks = sorted({x for x in [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, k] if x <= r})
    mse_all = ctx["mse_all"]
    rows = []
    for kk in ks:
        _, _, ra, sv = compression_stats(arr.shape, kk)
        rows.append({
            "Components (k)": kk,
            "Variance retained (%)": round(ctx["evr"][:kk].sum() * 100, 3),
            "MSE": round(float(mse_all[kk]), 3),
            "Compression ratio": round(ra, 2),
            "Storage saved (%)": round(sv * 100, 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_math():
    st.title("📐 Mathematical Explanation")

    st.header("1. Matrix representation")
    st.write("A greyscale image is an H x W matrix of brightness values 0 to 255. A colour image "
             "is three such matrices (R, G, B). We run PCA on each channel. "
             "Each **row** of pixels is one observation, each **column** is one feature.")
    st.latex(r"X \in \mathbb{R}^{H \times W}")

    st.header("2. Mean centering")
    st.write("Subtract the average of each column so the data is centred at the origin.")
    st.latex(r"\mu_j = \frac{1}{H}\sum_{i=1}^{H} X_{ij}, \qquad X_c = X - \mathbf{1}\mu^{T}")

    st.header("3. Covariance matrix")
    st.write("It tells us how the columns vary together. It is W x W and symmetric.")
    st.latex(r"C = \frac{1}{H-1}\, X_c^{T} X_c")

    st.header("4. Eigenvalues and eigenvectors")
    st.latex(r"C\,v_j = \lambda_j\, v_j")
    st.write("Each eigenvector v_j is a **principal direction**. Its eigenvalue λ_j is the "
             "variance of the data along that direction. We sort so λ_1 ≥ λ_2 ≥ ... Large "
             "eigenvalue means a direction that carries a lot of the image's information.")

    st.header("5. Principal Component Analysis via SVD")
    st.write("This app computes PCA with the Singular Value Decomposition, which is numerically "
             "more stable than forming C directly:")
    st.latex(r"X_c = U\,\Sigma\,V^{T}, \qquad \lambda_j = \frac{\sigma_j^{2}}{H-1}")
    st.write("The columns of V are the eigenvectors of C, and σ_j are the singular values.")

    st.header("6. Dimensionality reduction")
    st.write("Keep the first k eigenvectors as W_k (a W x k matrix) and project:")
    st.latex(r"X_{reduced} = X_c\,W_k \quad (H \times k)")

    st.header("7. Image reconstruction")
    st.latex(r"X_{reconstructed} = X_{reduced}\,W_k^{T} + \mathbf{1}\mu^{T}")

    st.header("8. Error and quality measures")
    st.latex(r"MSE = \frac{1}{N}\sum_{i=1}^{N}\left(X_i - \hat{X}_i\right)^2")
    st.latex(r"\text{Explained variance ratio}_k = \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{W}\lambda_j}")
    st.latex(r"\text{Compression ratio} = \frac{H\cdot W}{kH + kW + W}")

    st.subheader("What each symbol means")
    st.markdown(
        "| Symbol | Meaning |\n|---|---|\n"
        "| X | Original channel matrix (H x W) |\n"
        "| μ | Row vector of column means |\n"
        "| X_c | Mean-centred matrix |\n"
        "| C | Covariance matrix |\n"
        "| λ_j, v_j | j-th eigenvalue and eigenvector |\n"
        "| W_k | First k eigenvectors as columns |\n"
        "| X_reduced | Reduced representation, H x k |\n"
        "| X̂ (X_reconstructed) | Rebuilt image |\n"
        "| N | Total number of pixel values compared |\n"
        "| k | Number of principal components kept |"
    )
    st.info("Eckart-Young theorem: keeping the top k components gives the best possible rank-k "
            "approximation of the image in the least-squares (MSE) sense. "
            "The MSE equals the sum of the discarded eigenvalue energy divided by N.")


def page_visualization(ctx):
    st.title("📈 Visualization")
    need_image(ctx)
    models, arr, k = ctx["models"], ctx["arr"], ctx["k"]
    H, W, C = arr.shape
    evr = ctx["evr"]
    r = len(evr)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # 1. Explained variance
    st.subheader("1. Explained variance")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    show = min(r, 50)
    ax[0].bar(range(1, show + 1), evr[:show] * 100, color=TEAL)
    ax[0].set_title(f"Variance per component (first {show})")
    ax[0].set_xlabel("Component"); ax[0].set_ylabel("% of variance")
    ax[1].plot(range(1, r + 1), np.cumsum(evr) * 100, color=INK, lw=2)
    ax[1].axvline(k, color=AMBER, ls="--", label=f"k = {k}")
    ax[1].axhline(95, color="gray", ls=":", lw=1)
    ax[1].set_title("Cumulative variance"); ax[1].set_xlabel("Components")
    ax[1].set_ylabel("% retained"); ax[1].legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    k95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1)
    st.caption(f"{k95} components are enough to retain 95% of the variance for this image.")

    # 2. Principal components
    st.subheader("2. Principal components (eigenvectors)")
    name = "Gray" if C == 1 else "Red"
    Vt = models[0][3]
    n_show = min(5, Vt.shape[0])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i in range(n_show):
        ax.plot(Vt[i], lw=1.4, label=f"PC {i + 1}")
    ax.set_title(f"First {n_show} principal directions ({name} channel): weight of each pixel column")
    ax.set_xlabel("Column index (feature)"); ax.set_ylabel("Weight")
    ax.legend(ncol=n_show)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("PC 1 is usually smooth (overall brightness layout). Later components wiggle more: "
               "they encode finer detail.")

    # 3. Reconstruction error
    st.subheader("3. Reconstruction error")
    mse_all = ctx["mse_all"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(range(0, r + 1), mse_all, color=INK, lw=2)
    ax.set_yscale("log") if mse_all[:-1].min() > 0 else None
    ax.axvline(k, color=AMBER, ls="--", label=f"k = {k}")
    ax.set_title("MSE vs number of components (log scale)")
    ax.set_xlabel("Components kept"); ax.set_ylabel("MSE"); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 4. Gallery
    st.subheader("4. Original vs reconstructed at different k")
    ks = [x for x in [1, 5, 10, 25, 50, 100] if x < r]
    cols = st.columns(len(ks) + 1)
    cols[0].image(to_display(arr), caption="Original", use_container_width=True)
    for col, kk in zip(cols[1:], ks):
        col.image(to_display(reconstruct(models, kk)),
                  caption=f"k={kk} ({evr[:kk].sum() * 100:.1f}%)", use_container_width=True)


def page_about():
    st.title("ℹ️ About & Conclusion")
    st.header("Objectives")
    st.markdown(
        "- Understand how PCA reduces the dimensionality of image data.\n"
        "- Preserve the most important features (highest-variance directions).\n"
        "- Measure the trade-off between quality (MSE, PSNR, variance retained) and "
        "storage (compression ratio)."
    )
    st.header("Methodology")
    st.markdown(
        "1. Load the image and convert it to numeric matrices (RGB or greyscale).\n"
        "2. Mean-center each channel and compute the SVD.\n"
        "3. Keep the first k principal components.\n"
        "4. Reconstruct the image and compute MSE, PSNR, explained variance and compression ratio."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.header("Applications")
        st.markdown(
            "- Image compression and storage reduction\n"
            "- Face recognition (eigenfaces)\n"
            "- Medical imaging (MRI/CT noise reduction)\n"
            "- Feature extraction for machine learning\n"
            "- Faster training of computer vision models"
        )
        st.header("Advantages")
        st.markdown(
            "- Mathematically optimal linear reduction (least squares)\n"
            "- No labels needed (unsupervised)\n"
            "- Easy to interpret: eigenvalues show importance"
        )
    with c2:
        st.header("Limitations")
        st.markdown(
            "- Linear only: cannot capture curved structure\n"
            "- Lossy: fine detail is lost at low k\n"
            "- Storing U and V can cost more than the saving for small k-benefit images\n"
            "- Not as efficient as JPEG for pure photo compression"
        )
        st.header("Future scope")
        st.markdown(
            "- Block-wise PCA for better local detail\n"
            "- Kernel PCA and autoencoders for non-linear reduction\n"
            "- Eigenface-based face recognition on a dataset\n"
            "- Combining PCA with quantisation and entropy coding"
        )
    st.header("Conclusion")
    st.write("PCA shows that an image's essential structure lives in a small number of principal "
             "components. By keeping only those, we reduce the data we store and process while "
             "preserving the features that matter, which is the core idea behind many "
             "computer vision and machine learning pipelines.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_context():
    """Run PCA for the current image and sidebar settings. None if no image."""
    img = st.session_state.get("img")
    if img is None:
        return None
    gray = st.session_state.get("mode", "Colour (RGB)") == "Greyscale"
    arr = to_array(img, gray)
    H, W, C = arr.shape
    kmax = min(H, W)

    if "k" not in st.session_state:
        st.session_state.k = min(20, kmax)
    st.session_state.k = max(1, min(st.session_state.k, kmax))
    k = st.sidebar.slider("Principal components (k)", 1, kmax, key="k")

    models = run_pca(arr)
    rec = reconstruct(models, k)
    return {
        "arr": arr, "models": models, "k": k, "rec": rec,
        "evr": explained_variance_ratio(models),
        "mse_all": mse_curve(models, arr.shape),
        "metrics": compute_metrics(arr, rec),
        "comp": compression_stats(arr.shape, k),
    }


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    st.sidebar.title("🧬 Image PCA Lab")
    st.sidebar.radio("Navigate", PAGES, key="page")
    st.sidebar.divider()
    st.sidebar.radio("Image mode", ["Colour (RGB)", "Greyscale"], key="mode")
    ctx = build_context()
    if ctx is None:
        st.sidebar.caption("Add an image to unlock the component slider.")

    page = st.session_state.page
    if page == PAGES[0]:
        page_home()
    elif page == PAGES[1]:
        page_upload()
    elif page == PAGES[2]:
        page_pca(ctx)
    elif page == PAGES[3]:
        page_results(ctx)
    elif page == PAGES[4]:
        page_math()
    elif page == PAGES[5]:
        page_visualization(ctx)
    else:
        page_about()


main()
