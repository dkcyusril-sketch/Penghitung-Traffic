import streamlit as st
import cv2
import pandas as pd
import tempfile
import os
from collections import defaultdict

st.set_page_config(page_title="Penghitung Traffic Video", page_icon="🚦", layout="wide")
st.title("Penghitung Kendaraan & Pejalan Kaki")
st.write("Unggah video rekaman HP, lalu aplikasi akan mendeteksi dan menghitung objek yang melintas.")

st.info(
    "Versi awal ini memakai model YOLO pra-latih. Kategori yang tersedia mengikuti kelas model: "
    "mobil, sepeda motor, sepeda, bus, truk, dan orang. Angkutan umum kecil seperti angkot "
    "bisa salah terklasifikasi. Hasil perlu diperiksa, terutama bila objek saling menutupi."
)

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

uploaded = st.file_uploader("Pilih video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
conf = st.slider("Ambang keyakinan deteksi", 0.10, 0.90, 0.35, 0.05)
frame_step = st.select_slider("Kecepatan analisis", options=[1, 2, 3, 5, 8], value=2,
                              help="1 = setiap frame; angka lebih besar lebih cepat tetapi dapat melewatkan objek.")
line_position = st.slider("Posisi garis hitung (% tinggi video)", 10, 90, 60, 5)

if uploaded:
    st.video(uploaded)
    if st.button("Mulai menghitung", type="primary"):
        try:
            model = load_model()
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                video_path = tmp.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Video tidak dapat dibuka. Coba format MP4.")
                os.unlink(video_path)
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count_line_y = int(height * line_position / 100)

            # COCO classes: bicycle=1, car=2, motorcycle=3, bus=5, truck=7, person=0
            mapping = {0: "Pejalan kaki", 1: "Sepeda", 2: "Mobil",
                       3: "Sepeda motor", 5: "Bus", 7: "Truk"}
            totals = defaultdict(int)
            previous_y = {}
            counted_ids = set()
            output_rows = []
            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()
            frame_index = 0
            processed = 0

            output_path = os.path.join(tempfile.gettempdir(), "hasil_traffic_annotated.mp4")
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps/frame_step), (width, height))

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue

                result = model.track(frame, persist=True, conf=conf, verbose=False, tracker="bytetrack.yaml")[0]
                cv2.line(frame, (0, count_line_y), (width, count_line_y), (0, 255, 255), 2)

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                    xyxy = boxes.xyxy.cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    for track_id, box, cls, score in zip(ids, xyxy, clss, confs):
                        if cls not in mapping:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cy = int((y1 + y2) / 2)
                        label = mapping[cls]
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        tag = f"{label} {score:.2f}" + (f" ID:{track_id}" if track_id is not None else "")
                        cv2.putText(frame, tag, (x1, max(20, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                        if track_id is not None:
                            if track_id in previous_y and track_id not in counted_ids:
                                old_y = previous_y[track_id]
                                if (old_y < count_line_y <= cy) or (old_y > count_line_y >= cy):
                                    totals[label] += 1
                                    counted_ids.add(track_id)
                                    seconds = frame_index / fps
                                    output_rows.append({
                                        "Waktu video (detik)": round(seconds, 2),
                                        "Kategori": label,
                                        "ID objek": track_id
                                    })
                            previous_y[track_id] = cy

                y0 = 28
                for name in ["Mobil", "Sepeda motor", "Sepeda", "Bus", "Truk", "Pejalan kaki"]:
                    cv2.putText(frame, f"{name}: {totals[name]}", (12, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    y0 += 27

                writer.write(frame)
                if processed % 5 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Pratinjau hasil deteksi")
                processed += 1
                frame_index += 1
                if total_frames:
                    progress.progress(min(1.0, frame_index / total_frames))
                status.write(f"Memproses frame {frame_index:,} dari {total_frames:,}…")

            cap.release()
            writer.release()
            progress.progress(1.0)
            status.success("Analisis selesai.")

            names = ["Mobil", "Sepeda motor", "Sepeda", "Bus", "Truk", "Pejalan kaki"]
            summary = pd.DataFrame([{"Kategori": n, "Jumlah terhitung": int(totals[n])} for n in names])
            st.subheader("Hasil penghitungan")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.metric("Total semua objek yang melintasi garis", int(sum(totals.values())))

            csv = summary.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Unduh ringkasan CSV", csv, "ringkasan_traffic.csv", "text/csv")

            detail = pd.DataFrame(output_rows, columns=["Waktu video (detik)", "Kategori", "ID objek"])
            xlsx_path = os.path.join(tempfile.gettempdir(), "detail_traffic.xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
                summary.to_excel(xw, index=False, sheet_name="Ringkasan")
                detail.to_excel(xw, index=False, sheet_name="Detail")
            with open(xlsx_path, "rb") as f:
                st.download_button("Unduh laporan Excel", f.read(),
                                   "laporan_traffic.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button("Unduh video dengan penanda", f.read(),
                                       "video_traffic_terdeteksi.mp4", "video/mp4")

            st.warning(
                "Penting: penghitungan ini menghitung objek yang berhasil dilacak saat melintasi garis, "
                "bukan semua objek yang hanya terlihat di video. Jika kamera bergerak, garis terlalu dekat "
                "dengan tepi, atau lalu lintas sangat padat, hasil dapat meleset. Periksa video hasil sebelum "
                "menggunakan angka untuk keputusan survei."
            )
            os.unlink(video_path)
        except Exception as e:
            st.error(f"Analisis gagal: {e}")
            st.write("Pastikan internet tersedia saat pertama kali model YOLO diunduh, dan gunakan video MP4 bila memungkinkan.")
else:
    st.markdown("### Cara menggunakan")
    st.markdown("1. Jalankan aplikasi di laptop/PC atau lingkungan Python yang mendukungnya.\n"
                "2. Unggah video rekaman HP berdurasi 5–15 menit.\n"
                "3. Atur ambang deteksi dan posisi garis hitung.\n"
                "4. Tekan **Mulai menghitung**.\n"
                "5. Unduh ringkasan CSV, laporan Excel, atau video yang diberi penanda.")
