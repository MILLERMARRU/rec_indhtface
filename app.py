import streamlit as st
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

# Cargar modelo una vez
@st.cache_resource
def load_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

app = load_model()

st.set_page_config(layout="wide")
st.title("🧠 Detector Facial con InsightFace")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📸 Toma una foto o carga una imagen")
    camera_image = st.camera_input("Tomar foto")
    uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

with col2:
    if camera_image or uploaded_file:
        image_slot = st.empty()
        result_slot = st.empty()

        with st.spinner("🔍 Detectando rostros..."):
            if camera_image:
                image = Image.open(camera_image).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")

            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            faces = app.get(img_bgr)

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_slot.image(img_np, caption="Resultado", channels="RGB", use_container_width=True)
        result_slot.success(f"✅ Rostros detectados: {len(faces)}")
    else:
        st.info("📷 Toma una foto o carga una imagen para comenzar")
