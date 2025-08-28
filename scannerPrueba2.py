import cv2
import easyocr
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import time
import re

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Mayor resolución para mejor OCR
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Inicializar EasyOCR con configuración optimizada
reader = easyocr.Reader(['es'], gpu=False, verbose=False)

# Texto esperado (según la imagen proporcionada)
TEXTO_ESPERADO = {
    "empresa": "Unilever",
    "sscg": "089504161362863499",
    "material": "13702999",
    "descripcion": "Heriberto y Julio",
    "dun": "83569848665099",
    "lote": "12387744894310501399",
    "gramos": "000009457093/GRAM"
}

# Variables globales
frame_to_process = None
lock = threading.Lock()
conteo_textos = {}
textos_detectados = set()
ultimo_texto_detectado = ""
ultimo_tiempo_deteccion = 0
tiempo_entre_detecciones = 2  # segundos

# Preprocesamiento de imagen para mejorar OCR
def preprocess_image(image):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reducir ruido
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # Binarización adaptativa
    binary = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Operaciones morfológicas para limpiar la imagen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return morph

# Función para verificar si el texto detectado coincide con el esperado
def texto_coincide(texto_detectado):
    # Limpiar y normalizar el texto
    texto_limpio = re.sub(r'[^a-zA-Z0-9]', '', texto_detectado.upper())
    
    # Verificar coincidencia con cada campo esperado
    for campo, valor in TEXTO_ESPERADO.items():
        valor_limpio = re.sub(r'[^a-zA-Z0-9]', '', valor.upper())
        if valor_limpio in texto_limpio or texto_limpio in valor_limpio:
            return campo, valor
    
    return None, None

# Worker de OCR optimizado
def ocr_worker():
    global frame_to_process
    while True:
        if frame_to_process is not None:
            with lock:
                frame_copy = frame_to_process.copy()
                frame_to_process = None
                
            if frame_copy is not None:
                try:
                    # Preprocesar imagen
                    processed_frame = preprocess_image(frame_copy)
                    
                    # Realizar OCR
                    resultados = reader.readtext(
                        processed_frame, 
                        detail=1, 
                        paragraph=False, 
                        text_threshold=0.7,
                        low_text=0.4,
                        link_threshold=0.4,
                        decoder='beamsearch',
                        batch_size=5,
                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/:'
                    )
                    
                    # Procesar resultados
                    textos_validos = []
                    for bbox, texto, confianza in resultados:
                        if confianza > 0.6:  # Mayor umbral de confianza
                            texto_limpio = texto.strip()
                            if len(texto_limpio) > 3:  # Ignorar textos muy cortos
                                textos_validos.append((bbox, texto_limpio, confianza))
                    
                    # Actualizar resultados globales
                    with lock:
                        global resultados_ocr
                        resultados_ocr = textos_validos
                except Exception as e:
                    print(f"Error en OCR: {e}")
        
        time.sleep(0.1)  # Pequeña pausa para no saturar la CPU

# Iniciar el thread de OCR
resultados_ocr = []
threading.Thread(target=ocr_worker, daemon=True).start()

# Función principal de actualización
def update_frame():
    global frame_to_process, ultimo_texto_detectado, ultimo_tiempo_deteccion
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return
    
    # Preparar frame para procesamiento
    if frame_to_process is None:
        with lock:
            frame_to_process = frame.copy()
    
    # Dibujar cuadrícula de ayuda para el usuario
    h, w = frame.shape[:2]
    cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
    cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
    cv2.putText(frame, "Coloque el texto en el centro", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Procesar resultados OCR
    textos_actuales = set()
    with lock:
        resultados_actuales = resultados_ocr.copy()
    
    for bbox, texto, confianza in resultados_actuales:
        # Verificar si es un texto esperado
        campo, valor_esperado = texto_coincide(texto)
        
        if campo:
            color = (0, 255, 0)  # Verde para textos válidos
            texto_mostrar = f"{campo.upper()}: {texto} ({confianza:.2f})"
            
            # Verificar si es una nueva detección
            tiempo_actual = time.time()
            if (texto != ultimo_texto_detectado or 
                tiempo_actual - ultimo_tiempo_deteccion > tiempo_entre_detecciones):
                
                ultimo_texto_detectado = texto
                ultimo_tiempo_deteccion = tiempo_actual
                
                # Actualizar conteo
                if campo not in conteo_textos:
                    conteo_textos[campo] = 0
                conteo_textos[campo] += 1
                
                # Mostrar en consola
                console_text.insert(tk.END, f"{campo.upper()} detectado: {texto} (veces: {conteo_textos[campo]})\n")
                console_text.see(tk.END)
                
                # Actualizar estadísticas
                actualizar_estadisticas()
        else:
            color = (0, 0, 255)  # Rojo para textos no esperados
            texto_mostrar = f"Desconocido: {texto} ({confianza:.2f})"
        
        textos_actuales.add(texto)
        
        # Dibujar en el frame
        top_left = (int(bbox[0][0]), int(bbox[0][1]))
        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.putText(frame, texto_mostrar, (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Mostrar frame
    cv2.imshow("Detector de Texto - Mejorado", frame)
    
    # Salir con ESC
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        return
    
    root.after(10, update_frame)

# Función para actualizar estadísticas
def actualizar_estadisticas():
    # Limpiar frame de estadísticas
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    # Título
    tk.Label(stats_frame, text="ESTADÍSTICAS DE DETECCIÓN", 
             font=("Arial", 12, "bold")).pack(pady=5)
    
    # Mostrar conteo por campo
    for campo, valor_esperado in TEXTO_ESPERADO.items():
        conteo = conteo_textos.get(campo, 0)
        frame_campo = tk.Frame(stats_frame)
        frame_campo.pack(fill="x", padx=5, pady=2)
        
        tk.Label(frame_campo, text=f"{campo.upper()}:", 
                 width=15, anchor="w").pack(side="left")
        tk.Label(frame_campo, text=f"{conteo} detecciones", 
                 fg="green" if conteo > 0 else "red").pack(side="left")
    
    # Total de detecciones
    tk.Label(stats_frame, text=f"TOTAL: {sum(conteo_textos.values())} detecciones", 
             font=("Arial", 10, "bold")).pack(pady=10)

# Configuración de la interfaz
root = tk.Tk()
root.title("Sistema Mejorado de Detección de Texto")
root.geometry("1000x700")

# Marco principal
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Consola de resultados
console_label = tk.Label(main_frame, text="RESULTADOS EN TIEMPO REAL:", 
                         font=("Arial", 10, "bold"))
console_label.pack(anchor="w")

console_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                        width=70, height=15)
console_text.pack(fill="x", pady=5)

# Frame para estadísticas
stats_frame = tk.Frame(main_frame, relief="solid", borderwidth=1)
stats_frame.pack(fill="x", pady=10)

# Información del sistema
info_text = """
INSTRUCCIONES:
1. Coloque el documento frente a la cámara
2. Asegúrese de que el texto esté bien iluminado
3. Mantenga estable la cámara para mejor reconocimiento
4. Los textos reconocidos aparecerán en verde
5. Los textos no reconocidos aparecerán en rojo

TEXTOS ESPERADOS:
"""
for campo, valor in TEXTO_ESPERADO.items():
    info_text += f"- {campo.upper()}: {valor}\n"

info_label = tk.Label(main_frame, text=info_text, justify="left")
info_label.pack(anchor="w", pady=5)

# Iniciar actualización de frame
update_frame()

# Al cerrar la ventana
def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()