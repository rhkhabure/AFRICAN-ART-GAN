import torch
import sys

def patch_lightweight_gan():
    """
    lightweight_gan asserts CUDA availability at module level (line 38).
    We must patch torch.cuda.is_available to return True DURING the import,
    then immediately restore it. This fools the assert without affecting
    anything else.
    """
    if torch.cuda.is_available():
        return  # real GPU present, no patch needed

    # Temporarily make CUDA appear available just for the import
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: True

    try:
        import lightweight_gan.lightweight_gan as lgm

        # Now patch .cuda() so it doesn't actually try to move to GPU
        original_cuda = lgm.LightweightGAN.cuda
        def safe_cuda(self, rank=0):
            return self  # no-op on CPU
        lgm.LightweightGAN.cuda = safe_cuda

    finally:
        # Always restore the real function
        torch.cuda.is_available = original_is_available

patch_lightweight_gan()

import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import os
import io
import zipfile
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "African Fabric GAN",
    page_icon   = "🎨",
    layout      = "wide",
)

# ── Constants ─────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
MODEL_PATH = hf_hub_download(
    repo_id  = "Devaime/african-fabric-gan",
    filename = "model_final_step010000.pt"
)
GENERATED_DIR = "generated"
IMAGE_SIZE    = 256
LATENT_DIM    = 256
os.makedirs(GENERATED_DIR, exist_ok=True)

# ── Minimal Generator (matches LightweightGAN's G architecture)
# We re-implement only what's needed for inference so the app
# does NOT require lightweight_gan to be installed on the server.
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load generator weights from full checkpoint."""
    try:
        from lightweight_gan import LightweightGAN

        device = torch.device("cpu")

        GAN = LightweightGAN(
            latent_dim        = LATENT_DIM,
            image_size        = IMAGE_SIZE,
            fmap_max          = 512,
            fmap_inverse_coef = 12,
            attn_res_layers   = [32],
        ).to(device)

        ckpt = torch.load(MODEL_PATH, map_location=device)
        GAN.load_state_dict(ckpt["model"])
        GAN.eval()
        return GAN, device

    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()


def generate_images(GAN, device, num_images: int, seed: int = None):
    """Generate `num_images` fabric images. Returns list of PIL Images."""
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, device=device)
        imgs  = GAN.G(noise)
        imgs  = (imgs.clamp(-1, 1) + 1) / 2   # → [0, 1]

    pil_images = []
    for i in range(num_images):
        tensor = imgs[i].permute(1, 2, 0).cpu().numpy()
        tensor = (tensor * 255).astype("uint8")
        pil_images.append(Image.fromarray(tensor))

    return pil_images


def images_to_zip(image_list):
    """Pack a list of PIL Images into an in-memory ZIP and return bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(image_list):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"fabric_{i+1:03d}.png", img_buf.getvalue())
    buf.seek(0)
    return buf.read()


# ── UI ────────────────────────────────────────────────────────

st.title("🎨 African Fabric GAN")
st.markdown(
    "Generate synthetic African fabric patterns using a "
    "Lightweight GAN trained on 1,059 fabric images."
)
st.divider()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")

    num_images = st.slider(
        "Number of images to generate",
        min_value = 1,
        max_value = 16,
        value     = 8,
        step      = 1,
    )

    use_seed = st.checkbox("Fix random seed (reproducible results)", value=False)
    seed     = None
    if use_seed:
        seed = st.number_input("Seed value", min_value=0, max_value=99999, value=42)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Model: Lightweight GAN  \n"
        "Dataset: ~1,000 African fabric images  \n"
        "Resolution: 256 × 256  \n"
        "Training steps: 10,000"
    )

# Load model
GAN, device = load_model()
st.success("✅ Model loaded and ready")

# Generate button
col_btn, col_info = st.columns([1, 3])
with col_btn:
    generate_clicked = st.button(
        "🎲 Generate Images",
        use_container_width=True,
        type="primary",
    )

if generate_clicked:
    with st.spinner(f"Generating {num_images} fabric image(s)..."):
        images = generate_images(GAN, device, num_images, seed)

    # Save to disk with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []
    for i, img in enumerate(images):
        path = os.path.join(GENERATED_DIR, f"{timestamp}_{i+1:03d}.png")
        img.save(path)
        saved_paths.append(path)

    st.success(f"✅ Generated {num_images} image(s)")
    st.session_state["last_images"]  = images
    st.session_state["last_paths"]   = saved_paths

# ── Gallery of current generation ─────────────────────────────
if "last_images" in st.session_state:
    st.subheader("🖼️ Generated Images")

    images = st.session_state["last_images"]
    cols   = st.columns(4)

    for i, img in enumerate(images):
        with cols[i % 4]:
            st.image(img, caption=f"Image {i+1}", use_container_width=True)

    st.divider()

    # Download options
    st.subheader("⬇️ Download")
    dl_col1, dl_col2 = st.columns(2)

    # Download all as ZIP
    with dl_col1:
        zip_bytes = images_to_zip(images)
        st.download_button(
            label     = "📦 Download all as ZIP",
            data      = zip_bytes,
            file_name = f"african_fabric_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime      = "application/zip",
            use_container_width = True,
        )

    # Download individual images
    with dl_col2:
        selected = st.selectbox(
            "Download individual image",
            options = [f"Image {i+1}" for i in range(len(images))],
        )
        idx = int(selected.split(" ")[1]) - 1
        buf = io.BytesIO()
        images[idx].save(buf, format="PNG")
        st.download_button(
            label     = f"⬇️ Download {selected}",
            data      = buf.getvalue(),
            file_name = f"fabric_{idx+1:03d}.png",
            mime      = "image/png",
            use_container_width = True,
        )

# ── Pre-generated gallery (images already in /generated folder) ──
st.divider()
st.subheader("📁 Previously Generated Images")

all_saved = sorted([
    f for f in os.listdir(GENERATED_DIR)
    if f.endswith(".png")
])

if all_saved:
    st.markdown(f"_{len(all_saved)} image(s) saved in the `generated/` folder_")
    show_prev = st.checkbox("Show previously generated images", value=False)

    if show_prev:
        prev_cols = st.columns(4)
        for i, fname in enumerate(reversed(all_saved[-16:])):  # show last 16
            with prev_cols[i % 4]:
                img_path = os.path.join(GENERATED_DIR, fname)
                st.image(img_path, caption=fname[:16], use_container_width=True)
else:
    st.info("No previously generated images yet. Click **Generate Images** above.")
