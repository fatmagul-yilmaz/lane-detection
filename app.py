import os
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import deque

st.set_page_config(page_title="Otoyol Sürüş Güvenliği Portalı", layout="wide")
st.title("🚗 Otoyol Trafiğinde Şerit Sadakati ve Güvenli Takip Mesafesi Analiz Portalı")

st.sidebar.header("🛠️ Sistem Ayarları")
conf_threshold = st.sidebar.slider("YOLO Güven Eşiği (Conf)", 0.20, 0.90, 0.40, 0.05)
start_button = st.sidebar.button("Analizi Başlat / Yeniden Başlat")

st.sidebar.markdown("---")
st.sidebar.header("📊 Anlık Sürüş Raporu")
stat_status = st.sidebar.empty()
stat_offset = st.sidebar.empty()
stat_distance = st.sidebar.empty()
stat_fps = st.sidebar.empty()

video_placeholder = st.empty()

frames_root = "frames"
png_files = []

for root, dirs, files in os.walk(frames_root):
    for file in files:
        if file.lower().endswith(".png"):
            png_files.append(os.path.join(root, file))

def extract_number(path):
    name = os.path.basename(path)
    nums = ''.join(filter(str.isdigit, name))
    return int(nums) if nums != "" else 0

png_files.sort(key=extract_number)

if len(png_files) == 0:
    st.error("Hata: 'frames' klasöründe .png formatında frame bulunamadı!")
    st.stop()

kırpma_orani = 0.08  
atlanacak_frame_sayisi = int(len(png_files) * kırpma_orani)
png_files = png_files[atlanacak_frame_sayisi:]

def average_line(lines, height):
    if len(lines) == 0:
        return None
    slope_avg = np.mean([l[0] for l in lines])
    intercept_avg = np.mean([l[1] for l in lines])
    if slope_avg == 0:
        return None
    y1 = height
    y2 = int(height * 0.6)
    x1 = int((y1 - intercept_avg) / slope_avg)
    x2 = int((y2 - intercept_avg) / slope_avg)
    return (x1, y1, x2, y2)

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

if start_button or 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    
    model = load_yolo()
    
    sample_img = cv2.imread(png_files[0])
    h, w, _ = sample_img.shape
    
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon = np.array([
        [50, h],
        [w // 2 - 60, h // 2 + 40],
        [w // 2 + 60, h // 2 + 40],
        [w - 50, h]
    ])
    cv2.fillPoly(mask, [polygon], 255)
    
    last_left_lane = None
    last_right_lane = None
    fps_list = []
    prev_time = time.time()
    lane_history = deque(maxlen=10)
    PX_TO_CM = 0.5 

    for f_path in png_files:
        frame = cv2.imread(f_path)
        if frame is None:
            continue
            
        output = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = cv2.bitwise_and(edges, edges, mask=mask)

        lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)
        left, right = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                if slope < -0.5:
                    left.append((slope, intercept))
                elif slope > 0.5:
                    right.append((slope, intercept))

        left_lane = average_line(left, h)
        right_lane = average_line(right, h)

        if left_lane: last_left_lane = left_lane
        if right_lane: last_right_lane = right_lane

        lane_detected = 1 if (last_left_lane and last_right_lane) else 0
        lane_history.append(lane_detected)
        confidence = int((sum(lane_history) / len(lane_history)) * 100)

        status_text = "TETKİK EDİLİYOR"
        status_color = (0, 255, 255)
        offset_cm = 0.0

        if last_left_lane and last_right_lane:
            overlay = output.copy()
            pts = np.array([
                [last_left_lane[0], last_left_lane[1]],
                [last_left_lane[2], last_left_lane[3]],
                [last_right_lane[2], last_right_lane[3]],
                [last_right_lane[0], last_right_lane[1]]
            ], np.int32)
            
            lane_center = (last_left_lane[0] + last_right_lane[0]) // 2
            vehicle_center = w // 2
            offset_px = vehicle_center - lane_center
            offset_cm = offset_px * PX_TO_CM

            if abs(offset_cm) < 15:
                status_text = "MERKEZDE"
                status_color = (0, 255, 0)      
                fill_color = (255, 0, 0)        
            elif abs(offset_cm) < 35:
                status_text = "HAFIF KAYMA"
                status_color = (0, 255, 255)    
                fill_color = (0, 255, 255)      
            else:
                status_text = "SAGA KAYMA" if offset_cm > 0 else "SOLA KAYMA"
                status_color = (0, 0, 255)     
                fill_color = (0, 0, 255)       

            cv2.fillPoly(overlay, [pts], fill_color)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
            
            cv2.line(output, last_left_lane[:2], last_left_lane[2:], (255, 0, 0), 5)
            cv2.line(output, last_right_lane[:2], last_right_lane[2:], (255, 0, 0), 5)

        yolo_results = model.track(frame, classes=[2, 5, 7], conf=conf_threshold, verbose=False, persist=True)
        
        closest_car_dist = "Menzil Dışı"
        tracking_alert = "GÜVENLİ"
        min_distance_meters = 999  
        
        if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                c_conf = box.conf[0]
                c_id = int(box.id[0]) if box.id is not None else None
                
                car_center_x = int((x1 + x2) / 2)
                
                horizon = int(h * 0.52)
                dy = max(y2 - horizon, 1) 
                
                distance_meters = int(1200 / dy) 
                distance_meters = max(2, min(distance_meters, 80))
                
                in_our_lane = (w // 2 - 120) < car_center_x < (w // 2 + 120)
                
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 105, 180), 2)
                id_text = f" ID:{c_id}" if c_id is not None else ""
                cv2.putText(output, f"Arac{id_text} ({distance_meters}m)", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 105, 180), 2)
                
                if y2 > int(h * 0.55):
                    if in_our_lane and distance_meters < 8:
                        tracking_alert = f"⚠️ TEHLİKE: ID {c_id} YAKIN TAKİP!"
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.line(output, (w // 2, h), (car_center_x, y2), (0, 0, 255), 3)
                        cv2.putText(output, "FREN YAP!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.line(output, (w // 2, h), (car_center_x, y2), (0, 255, 0), 1)
                
                if in_our_lane and distance_meters < min_distance_meters:
                    min_distance_meters = distance_meters
                    closest_car_dist = f"{distance_meters} Metre (ID: {c_id})"

        if min_distance_meters == 999:
            closest_car_dist = "Menzil Dışı"

        cv2.putText(output, f"Lane Confidence: {confidence}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if last_left_lane and last_right_lane:
            cv2.putText(output, f"Sapma: {offset_cm:.1f} cm ({status_text})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        curr_time = time.time()
        delta = max(curr_time - prev_time, 1e-6)
        prev_time = current_time = curr_time
        fps_list.append(1 / delta)
        if len(fps_list) > 10: fps_list.pop(0)
        avg_fps = int(sum(fps_list) / len(fps_list))
        cv2.putText(output, f"FPS: {avg_fps}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stat_status.metric(label="Şerit Durumu", value=status_text)
        stat_offset.metric(label="Şeritten Sapma (cm)", value=f"{offset_cm:.1f} cm")
        stat_distance.metric(label="Öndeki Araç Mesafesi / Durum", value=closest_car_dist, delta=tracking_alert, delta_color="inverse" if "TEHLİKE" in tracking_alert else "normal")
        stat_fps.metric(label="Sistem Hızı (FPS)", value=avg_fps)

        frame_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        time.sleep(0.01)

    st.success("Tüm frame'lerin analizi başarıyla tamamlandı ve raporlandı!")