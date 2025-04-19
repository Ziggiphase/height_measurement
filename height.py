import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load model
model = YOLO("yolov8m.pt")

st.title("People Height Estimator (Right-to-Left ID Order)")
st.write("Estimate people's height from an image or video, given the known height of one person.")

# Sidebar input
mode = st.sidebar.radio("Choose Input Type", ("Image", "Video"))
ref_index = st.sidebar.number_input("Index of person with known height (Right-to-Left)", min_value=0, step=1)
ref_height = st.sidebar.number_input("Known Height (in meters)", min_value=0.1, step=0.1)

def estimate_heights_from_image(image, ref_index, ref_height):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    
    # Sort people from right to left by x2
    person_boxes = sorted(
        [box for i, box in enumerate(boxes) if int(class_ids[i]) == 0],
        key=lambda b: b[2],
        reverse=True
    )
    
    if len(person_boxes) == 0 or ref_index >= len(person_boxes):
        st.error("No people detected or invalid reference index.")
        return None
    
    ref_box = person_boxes[ref_index]
    ref_pixel_height = ref_box[3] - ref_box[1]
    pixels_per_meter = ref_pixel_height / ref_height

    annotated = image.copy()

    for idx, box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, box)
        pixel_height = y2 - y1
        estimated_height = pixel_height / pixels_per_meter

        color = (0, 255, 0) if idx == ref_index else (255, 0, 0)
        label = f"Ref: {estimated_height:.2f}m" if idx == ref_index else f"{estimated_height:.2f}m"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def estimate_heights_from_video(video_path, ref_index, ref_height):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        person_boxes = sorted(
            [box for i, box in enumerate(boxes) if int(class_ids[i]) == 0],
            key=lambda b: b[2],
            reverse=True
        )

        if len(person_boxes) == 0 or ref_index >= len(person_boxes):
            out.write(frame)
            continue

        ref_box = person_boxes[ref_index]
        ref_pixel_height = ref_box[3] - ref_box[1]
        pixels_per_meter = ref_pixel_height / ref_height

        for idx, box in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, box)
            pixel_height = y2 - y1
            estimated_height = pixel_height / pixels_per_meter

            color = (0, 255, 0) if idx == ref_index else (255, 0, 0)
            label = f"Ref: {estimated_height:.2f}m" if idx == ref_index else f"{estimated_height:.2f}m"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    return out_path

# Image mode
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None and ref_height > 0:
        img = Image.open(uploaded_image)
        img = np.array(img)
        result_img = estimate_heights_from_image(img, ref_index, ref_height)
        if result_img is not None:
            st.image(result_img, caption="Height Estimation", use_column_width=True)

# Video mode
else:
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None and ref_height > 0:
        temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_vid.write(uploaded_video.read())
        result_vid_path = estimate_heights_from_video(temp_vid.name, ref_index, ref_height)
        st.video(result_vid_path)
