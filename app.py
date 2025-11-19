# import os
# import traceback
# import torch
# import streamlit as st
# import pandas as pd
#
# from prediction import predict
# from model.pred_func import load_genconvit, set_result
# from model.config import load_config
#
# # -------------------------------------------------------
# # Streamlit setup
# # -------------------------------------------------------
# st.set_page_config(page_title="Vid Deepfake Detector", layout="wide")
# st.title("🎥 Vid Deepfake Detection")
#
# # -------------------------------------------------------
# # Device detection (GPU if available)
# # -------------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# st.sidebar.write(f"🔌 Device: **{device}**")
# if torch.cuda.is_available():
#     st.sidebar.write(f"⚡ GPU: **{torch.cuda.get_device_name(0)}**")
#
# # -------------------------------------------------------
# # Load models (GPU if available)
# # -------------------------------------------------------
# @st.cache_resource
# def load_model(model_type="vae"):
#     """Load ED, VAE, or Both — automatically move to GPU if available."""
#     config = load_config()
#     ed_model = vae_model = None
#
#     if model_type in ["ed", "both"]:
#         ed_model = load_genconvit(config, "ed", "genconvit_ed_inference", None, fp16=False)
#         ed_model = ed_model.to(device)
#
#     if model_type in ["vae", "both"]:
#         vae_model = load_genconvit(config, "vae", None, "genconvit_vae_inference", fp16=False)
#         vae_model = vae_model.to(device)
#
#     return ed_model, vae_model
#
# # -------------------------------------------------------
# # Sidebar controls
# # -------------------------------------------------------
# st.sidebar.header("⚙️ Settings")
#
# model_choice = st.sidebar.radio(
#     "Select Model(s):",
#     ["vae", "ed", "both"],
#     index=0,
#     help="Choose which model(s) to run predictions with.",
# )
#
# num_frames = st.sidebar.slider(
#     "Number of Frames to Process per Video",
#     min_value=5,
#     max_value=60,
#     value=15,
#     step=5,
#     help="Higher frame counts improve accuracy but increase inference time.",
# )
#
# st.sidebar.write(" ")
#
# ed_model, vae_model = load_model(model_choice)
#
# mode = st.radio("Select Mode", ["Single Video", "Folder of Videos"], horizontal=True)
#
# # -------------------------------------------------------
# # SINGLE VIDEO MODE
# # -------------------------------------------------------
# if mode == "Single Video":
#
#     uploaded_file = st.file_uploader("Upload a video (.mp4)", type=["mp4"])
#
#     if uploaded_file:
#         video_path = "temp_single.mp4"
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.read())
#
#         st.video(video_path, format="video/mp4")
#
#         if st.button("🔍 Predict"):
#             st.info("Running deepfake detection... Please wait.")
#
#             try:
#                 result_dict = set_result()
#                 results = []
#
#                 # ---------------------------------------------
#                 # Run VAE model
#                 # ---------------------------------------------
#                 if vae_model is not None:
#                     result, acc, count, pred = predict(
#                         video_path,
#                         vae_model.to(device),
#                         False,
#                         result_dict,
#                         num_frames,
#                         "vae",
#                         "uncategorized"
#                     )
#                     _, conf = pred[0], float(pred[1])
#                     label = "🟥 FAKE" if conf >= 0.5 else "🟩 REAL"
#                     results.append(("VAE", label, conf))
#
#                 # ---------------------------------------------
#                 # Run ED model
#                 # ---------------------------------------------
#                 if ed_model is not None:
#                     result, acc, count, pred = predict(
#                         video_path,
#                         ed_model.to(device),
#                         False,
#                         result_dict,
#                         num_frames,
#                         "ed",
#                         "uncategorized"
#                     )
#                     _, conf = pred[0], float(pred[1])
#                     label = "🟥 FAKE" if conf >= 0.5 else "🟩 REAL"
#                     results.append(("ED", label, conf))
#
#                 # ---------------------------------------------
#                 # Display results
#                 # ---------------------------------------------
#                 st.subheader("Results")
#                 for name, label, conf in results:
#                     st.write(f"**{name} Model:** {label} (Confidence: {conf*100:.2f}%)")
#
#             except Exception:
#                 st.error("❌ Error processing video:")
#                 st.code(traceback.format_exc())
#
# # -------------------------------------------------------
# # FOLDER MODE
# # -------------------------------------------------------
# else:
#     st.write("📂 Paste or drag the folder path that contains your .mp4 videos below.")
#
#     folder_path = st.text_input(
#         "Folder path",
#         value="",
#         placeholder=r"e.g. C:\Users\hadee\PycharmProjects\DeepFake\sample_prediction_data",
#     )
#
#     if folder_path and os.path.isdir(folder_path):
#
#         video_files = [
#             os.path.join(folder_path, f)
#             for f in os.listdir(folder_path)
#             if f.lower().endswith(".mp4")
#         ]
#
#         if not video_files:
#             st.warning("No .mp4 videos found in this folder.")
#         else:
#             st.info(f"Found {len(video_files)} videos.")
#
#             if st.button("🔍 Predict All"):
#                 st.info("Running predictions... Please wait.")
#
#                 results = []
#                 progress = st.progress(0)
#
#                 for i, vid in enumerate(video_files):
#
#                     try:
#                         result_dict = set_result()
#                         row = {"Video": os.path.basename(vid)}
#
#                         # VAE prediction
#                         if vae_model is not None:
#                             result, acc, count, pred = predict(
#                                 vid,
#                                 vae_model.to(device),
#                                 False,
#                                 result_dict,
#                                 num_frames,
#                                 "vae",
#                                 "uncategorized"
#                             )
#                             _, conf = pred[0], float(pred[1])
#                             label = "FAKE" if conf >= 0.5 else "REAL"
#                             row["VAE"] = f"{label} ({conf:.3f})"
#
#                         # ED prediction
#                         if ed_model is not None:
#                             result, acc, count, pred = predict(
#                                 vid,
#                                 ed_model.to(device),
#                                 False,
#                                 result_dict,
#                                 num_frames,
#                                 "ed",
#                                 "uncategorized"
#                             )
#                             _, conf = pred[0], float(pred[1])
#                             label = "FAKE" if conf >= 0.5 else "REAL"
#                             row["ED"] = f"{label} ({conf:.3f})"
#
#                         results.append(row)
#
#                     except Exception:
#                         st.error(f"❌ Error with {os.path.basename(vid)}:")
#                         st.code(traceback.format_exc())
#
#                     progress.progress((i + 1) / len(video_files))
#
#                 df = pd.DataFrame(results)
#                 st.dataframe(df, use_container_width=True)
#                 st.success(f"✅ Done! Processed {len(video_files)} videos.")
#
#                 # Save results
#                 os.makedirs("result", exist_ok=True)
#                 csv_path = os.path.join("result", "streamlit_results.csv")
#                 df.to_csv(csv_path, index=False)
#
#                 with open(csv_path, "rb") as f:
#                     st.download_button(
#                         "📥 Download CSV",
#                         data=f,
#                         file_name="results.csv"
#                     )
#
#     elif folder_path:
#         st.error("⚠️ Invalid folder path. Please check and try again.")


import os
import traceback
import torch
import streamlit as st
import pandas as pd

from prediction import predict
from model.pred_func import load_genconvit, set_result
from model.config import load_config

# -------------------------------------------------------
# Streamlit setup
# -------------------------------------------------------
st.set_page_config(page_title="Vid Deepfake Detector", layout="wide")
st.title("🎥 Vid Deepfake Detection")

# -------------------------------------------------------
# Device detection
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.write(f"🔌 Device: **{device}**")
if torch.cuda.is_available():
    st.sidebar.write(f"⚡ GPU: **{torch.cuda.get_device_name(0)}**")

# -------------------------------------------------------
# Load models
# -------------------------------------------------------
@st.cache_resource
def load_model(model_type="vae"):
    config = load_config()

    # VAE only
    if model_type == "vae":
        vae = load_genconvit(config, "vae", None, "genconvit_vae_inference", fp16=False)
        return None, vae.to(device), None

    # ED only
    if model_type == "ed":
        ed = load_genconvit(config, "ed", "genconvit_ed_inference", None, fp16=False)
        return ed.to(device), None, None

    # BOTH: fused model like offline script
    if model_type == "both":
        fused = load_genconvit(
            config,
            "genconvit",
            "genconvit_ed_inference",
            "genconvit_vae_inference",
            fp16=False
        )
        return None, None, fused.to(device)

    return None, None, None


# -------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.radio(
    "Select Model(s):",
    ["vae", "ed", "both"],
    index=0
)

num_frames = st.sidebar.slider(
    "Number of Frames to Process per Video",
    min_value=5,
    max_value=60,
    value=15,
    step=5
)

st.sidebar.write(" ")

# Updated assignment
ed_model, vae_model, fused_model = load_model(model_choice)

mode = st.radio("Select Mode", ["Single Video", "Folder of Videos"], horizontal=True)

# -------------------------------------------------------
# SINGLE VIDEO MODE
# -------------------------------------------------------
if mode == "Single Video":

    uploaded_file = st.file_uploader("Upload a video (.mp4)", type=["mp4"])

    if uploaded_file:
        video_path = "temp_single.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path, format="video/mp4")

        if st.button("🔍 Predict"):
            st.info("Running deepfake detection. Please wait.")

            try:
                result_dict = set_result()
                results = []

                # Combined model
                if fused_model is not None:
                    result, acc, count, pred = predict(
                        video_path,
                        fused_model.to(device),
                        False,
                        result_dict,
                        num_frames,
                        "genconvit",
                        "uncategorized"
                    )
                    _, conf = pred[0], float(pred[1])
                    label = "🟥 FAKE" if conf >= 0.5 else "🟩 REAL"
                    results.append(("Combined", label, conf))

                # VAE only
                if vae_model is not None and fused_model is None:
                    result, acc, count, pred = predict(
                        video_path,
                        vae_model.to(device),
                        False,
                        result_dict,
                        num_frames,
                        "vae",
                        "uncategorized"
                    )
                    _, conf = pred[0], float(pred[1])
                    label = "🟥 FAKE" if conf >= 0.5 else "🟩 REAL"
                    results.append(("VAE", label, conf))

                # ED only
                if ed_model is not None and fused_model is None:
                    result, acc, count, pred = predict(
                        video_path,
                        ed_model.to(device),
                        False,
                        result_dict,
                        num_frames,
                        "ed",
                        "uncategorized"
                    )
                    _, conf = pred[0], float(pred[1])
                    label = "🟥 FAKE" if conf >= 0.5 else "🟩 REAL"
                    results.append(("ED", label, conf))

                st.subheader("Results")
                for name, label, conf in results:
                    st.write(f"**{name} Model:** {label} (Confidence: {conf*100:.2f} percent)")

            except Exception:
                st.error("❌ Error processing video:")
                st.code(traceback.format_exc())

# -------------------------------------------------------
# FOLDER MODE
# -------------------------------------------------------
else:
    st.write("📂 Paste or drag the folder path that contains your .mp4 videos below.")

    folder_path = st.text_input(
        "Folder path",
        value="",
        placeholder=r"e.g. C:\Users\hadee\PycharmProjects\DeepFake\GenConViT\sample_prediction_data",
    )

    if folder_path and os.path.isdir(folder_path):

        video_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".mp4")
        ]

        if not video_files:
            st.warning("No .mp4 videos found in this folder.")
        else:
            st.info(f"Found {len(video_files)} videos.")

            if st.button("🔍 Predict All"):
                st.info("Running predictions. Please wait.")

                results = []
                progress = st.progress(0)

                for i, vid in enumerate(video_files):

                    try:
                        result_dict = set_result()
                        row = {"Video": os.path.basename(vid)}

                        # Combined model
                        if fused_model is not None:
                            result, acc, count, pred = predict(
                                vid,
                                fused_model.to(device),
                                False,
                                result_dict,
                                num_frames,
                                "genconvit",
                                "uncategorized"
                            )
                            _, conf = pred[0], float(pred[1])
                            label = "FAKE" if conf >= 0.5 else "REAL"
                            row["Combined"] = f"{label} ({conf:.3f})"

                        # VAE only
                        if vae_model is not None and fused_model is None:
                            result, acc, count, pred = predict(
                                vid,
                                vae_model.to(device),
                                False,
                                result_dict,
                                num_frames,
                                "vae",
                                "uncategorized"
                            )
                            _, conf = pred[0], float(pred[1])
                            label = "FAKE" if conf >= 0.5 else "REAL"
                            row["VAE"] = f"{label} ({conf:.3f})"

                        # ED only
                        if ed_model is not None and fused_model is None:
                            result, acc, count, pred = predict(
                                vid,
                                ed_model.to(device),
                                False,
                                result_dict,
                                num_frames,
                                "ed",
                                "uncategorized"
                            )
                            _, conf = pred[0], float(pred[1])
                            label = "FAKE" if conf >= 0.5 else "REAL"
                            row["ED"] = f"{label} ({conf:.3f})"

                        results.append(row)

                    except Exception:
                        st.error(f"❌ Error with {os.path.basename(vid)}:")
                        st.code(traceback.format_exc())

                    progress.progress((i + 1) / len(video_files))

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                st.success(f"✅ Done. Processed {len(video_files)} videos.")

                os.makedirs("result", exist_ok=True)
                csv_path = os.path.join("result", "streamlit_results.csv")
                df.to_csv(csv_path, index=False)

                with open(csv_path, "rb") as f:
                    st.download_button(
                        "📥 Download CSV",
                        data=f,
                        file_name="results.csv"
                    )

    elif folder_path:
        st.error("⚠️ Invalid folder path. Please check and try again.")
