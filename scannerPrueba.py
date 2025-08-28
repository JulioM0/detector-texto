import cv2
import easyocr
import threading
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import re
from collections import defaultdict
import time

# Configuración de la cámara y medidas
reader = easyocr.Reader(['es'], gpu=False, verbose=False)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # Resolución balanceada
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Variables globales 
results = []
frame_to_process = None
lock = threading.Lock()
conteo_textos = {}
textos_en_pantalla = set()
texto_historico = defaultdict(list)  # Para validación temporal
frame_count = 0

def preprocess_image(image):
    """Preprocesamiento optimizado de la imagen para mejor OCR"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Suavizado ligero para reducir ruido
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Threshold adaptativo mejorado
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Operación morfológica suave para limpiar
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def is_valid_text(text, min_length=1):
    """Validar si el texto detectado es válido - más permisivo para párrafos"""
    if not text or len(text.strip()) < min_length:
        return False
    
    # Permitir texto con números y letras
    if not re.search(r'[A-Za-z0-9]', text):
        return False
    
    # Más permisivo con caracteres especiales para párrafos completos
    special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
    if special_char_ratio > 0.5:  # Máximo 50% de caracteres especiales
        return False
    
    return True

def group_nearby_text(ocr_results):
    """Agrupar texto que está cerca geográficamente para formar párrafos"""
    if not ocr_results:
        return []
    
    grouped = []
    used_indices = set()
    
    for i, item in enumerate(ocr_results):
        if i in used_indices:
            continue
        
        # Manejar diferentes formatos de resultado
        if len(item) == 3:
            bbox1, text1, conf1 = item
        elif len(item) == 2:
            bbox1, text1 = item
            conf1 = 1.0
        else:
            continue
            
        # Inicializar grupo con el texto actual
        group_bbox = bbox1
        group_text = text1
        group_conf = conf1
        used_indices.add(i)
        
        # Calcular centro del bbox actual
        center1_x = (bbox1[0][0] + bbox1[2][0]) / 2
        center1_y = (bbox1[0][1] + bbox1[2][1]) / 2
        
        # Buscar textos cercanos para agrupar
        for j, item2 in enumerate(ocr_results):
            if j in used_indices:
                continue
            
            # Manejar diferentes formatos de resultado
            if len(item2) == 3:
                bbox2, text2, conf2 = item2
            elif len(item2) == 2:
                bbox2, text2 = item2
                conf2 = 1.0
            else:
                continue
                
            # Calcular centro del bbox candidato
            center2_x = (bbox2[0][0] + bbox2[2][0]) / 2
            center2_y = (bbox2[0][1] + bbox2[2][1]) / 2
            
            # Calcular distancia
            distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
            
            # Si está cerca (menos de 100 píxeles), agrupar
            if distance < 100:
                # Determinar orden de lectura (izquierda a derecha, arriba a abajo)
                if center2_y < center1_y - 10:  # Está arriba
                    group_text = text2 + " " + group_text
                elif center2_y > center1_y + 10:  # Está abajo
                    group_text = group_text + " " + text2
                elif center2_x < center1_x:  # Está a la izquierda
                    group_text = text2 + " " + group_text
                else:  # Está a la derecha
                    group_text = group_text + " " + text2
                
                # Expandir bbox para incluir el nuevo texto
                min_x = min(group_bbox[0][0], group_bbox[1][0], bbox2[0][0], bbox2[1][0])
                min_y = min(group_bbox[0][1], group_bbox[3][1], bbox2[0][1], bbox2[3][1])
                max_x = max(group_bbox[2][0], group_bbox[1][0], bbox2[2][0], bbox2[1][0])
                max_y = max(group_bbox[2][1], group_bbox[3][1], bbox2[2][1], bbox2[3][1])
                
                group_bbox = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
                group_conf = (group_conf + conf2) / 2  # Promedio de confianza
                used_indices.add(j)
        
        # Limpiar texto agrupado
        group_text = re.sub(r'\s+', ' ', group_text.strip())
        grouped.append((group_bbox, group_text, group_conf))
    
    return grouped

def validate_text_temporally(text, current_frame):
    """Validar texto basándose en apariciones anteriores"""
    texto_historico[text].append(current_frame)
    
    # Mantener solo los últimos 10 frames en el historial
    texto_historico[text] = [f for f in texto_historico[text] if current_frame - f <= 10]
    
    # Para párrafos, ser menos restrictivo - aparecer en 2 frames
    return len(texto_historico[text]) >= 2

def calculate_text_stability(bbox_list):
    """Calcular la estabilidad de la posición del texto"""
    if len(bbox_list) < 2:
        return True
    
    # Calcular la variación en las posiciones
    centers = []
    for bbox in bbox_list:
        if len(bbox) >= 4:
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            centers.append((center_x, center_y))
    
    if len(centers) < 2:
        return True
    
    # Si el texto se mueve demasiado, probablemente sea ruido
    max_distance = 0
    for i in range(1, len(centers)):
        distance = np.sqrt((centers[i][0] - centers[i-1][0])**2 + 
                          (centers[i][1] - centers[i-1][1])**2)
        max_distance = max(max_distance, distance)
    
    return max_distance < 50  # Umbral de movimiento máximo

# Función OCR optimizada para leer párrafos completos
def ocr_worker():
    global frame_to_process, results
    while True:
        if frame_to_process is not None:
            with lock:
                frame_copy = frame_to_process.copy()
            
            # Preprocesamiento
            processed_frame = preprocess_image(frame_copy)
            
            # OCR configurado para agrupar texto en párrafos
            ocr_results = reader.readtext(
                processed_frame, 
                detail=True,
                paragraph=True,  # Activar agrupación en párrafos
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.8,  # Aumentar para mejor agrupación
                width_ths=0.9,  # Permitir más variación en ancho
                height_ths=0.9,  # Permitir más variación en altura
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz áéíóúüñÁÉÍÓÚÜÑ.,!?-()[]{}:;'
            )
            
            # Agrupar texto cercano manualmente si es necesario
            grouped_results = group_nearby_text(ocr_results)
            
            # Filtrar resultados con manejo de errores
            filtered_results = []
            for item in ocr_results:
                if len(item) == 3:
                    bbox, text, confidence = item
                elif len(item) == 2:
                    bbox, text = item
                    confidence = 1.0
                else:
                    continue
                
                # Filtros de calidad más permisivos para párrafos
                if (confidence > 0.7 and 
                    is_valid_text(text) and
                    len(text.strip()) >= 1):
                    filtered_results.append((bbox, text.strip(), confidence))
            
            results = filtered_results
            frame_to_process = None
        else:
            time.sleep(0.01)

threading.Thread(target=ocr_worker, daemon=True).start()

# Función que actualiza cámara y consola
def update_frame():
    global frame_to_process, textos_en_pantalla, frame_count
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return
    
    frame_count += 1

    # Procesar cada 2 frames para mejor rendimiento
    if frame_to_process is None and frame_count % 2 == 0:
        with lock:
            frame_to_process = frame.copy()
    
    textos_actuales = set()
    textos_validados = []

    # Procesar solo textos validados con manejo de errores
    for item in results:
        if len(item) == 3:
            bbox, text, confidence = item
        elif len(item) == 2:
            bbox, text = item
            confidence = 1.0
        else:
            continue
            
        # Validación temporal adicional
        if validate_text_temporally(text, frame_count):
            textos_validados.append((bbox, text, confidence))
            textos_actuales.add(text)

    # Procesar solo textos validados
    for bbox, text, confidence in textos_validados:
        if text not in textos_en_pantalla:
            conteo_textos[text] = conteo_textos.get(text, 0) + 1
            console_text.insert(tk.END, f"📄 {text}\n   (Confianza: {confidence:.2f}) - Aparición #{conteo_textos[text]}\n\n")
            console_text.see(tk.END)  # Scroll automático

        # Dibujar en pantalla con información adicional
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Color basado en confianza
        if confidence > 0.90:
            color = (0, 255, 0)  # Verde
        elif confidence > 0.80:
            color = (0, 255, 255)  # Amarillo
        else:
            color = (0, 165, 255)  # Naranja
        
        cv2.rectangle(frame, top_left, bottom_right, color, 3)
        
        # Mostrar solo los primeros 30 caracteres en pantalla para no saturar
        display_text = text[:30] + "..." if len(text) > 30 else text
        cv2.putText(frame, f"{display_text}", 
                    (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    textos_en_pantalla = textos_actuales

    # Mostrar información adicional en el frame
    cv2.putText(frame, f"Textos detectados: {len(textos_validados)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detector de texto", frame)

    if cv2.waitKey(1) == 27:  # ESC para salir
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        return

    root.after(10, update_frame)

# Interfaz gráfica
root = tk.Tk()
root.title("Resultados de la lectura")
root.geometry("600x400")

# Frame para botones
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Botón para limpiar consola
def clear_console():
    console_text.delete(1.0, tk.END)

clear_button = tk.Button(button_frame, text="Limpiar Consola", command=clear_console)
clear_button.pack(side=tk.LEFT, padx=5)

# Botón para resetear contadores
def reset_counters():
    global conteo_textos, texto_historico
    conteo_textos.clear()
    texto_historico.clear()
    console_text.delete(1.0, tk.END)
    console_text.insert(tk.END, "Contadores reseteados\n")

reset_button = tk.Button(button_frame, text="Resetear Contadores", command=reset_counters)
reset_button.pack(side=tk.LEFT, padx=5)

# Console de texto
console_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
console_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

console_text.insert(tk.END, "Sistema OCR optimizado para párrafos completos.\n")
console_text.insert(tk.END, "✅ Agrupación de texto habilitada\n")
console_text.insert(tk.END, "✅ Lectura de números y texto completo\n")
console_text.insert(tk.END, "✅ Detección de párrafos mejorada\n\n")

update_frame()
root.mainloop()