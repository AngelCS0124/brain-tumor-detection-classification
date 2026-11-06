import streamlit as st
import pathlib as pl
from PIL import Image
import numpy as np
from monai.networks.nets import DenseNet121 as dn_net
from monai.transforms import Compose, ScaleIntensity, Resize, ToTensor
import torch
import torch.nn.functional as F

analsis_terminado = False

# Clases de tumores cerebrales
clases = ['glioma', 'meningioma', 'pituitario', 'sin tumor']
num_clases = 4

# Inicialización de arquitectura
arquitectura = dn_net(
    spatial_dims=2, 
    in_channels=1, 
    out_channels=num_clases
)

# Carga de los pesos del entrenamiento
pesos_entrenamiento = torch.load('modelo/modelo_45.pth', map_location='cpu')
arquitectura.load_state_dict(pesos_entrenamiento)
arquitectura.eval()
print("Modelo cargado exitosamente")

def cargar_css(ruta_archivo):
    with open(ruta_archivo) as f:
        st.html(f"<style>{f.read()}</style>")

def transformar_img(archivo):
    imagen = Image.open(archivo).convert('L') # Transofrmación a escala de grises
    array_imagen = np.array(imagen)
    array_canal = np.expand_dims(array_imagen, axis=0) # Adición de la dimensión de canal, que es lo que espera DenseNet
    
    transforms = Compose([
        ScaleIntensity(),   # Normalización del array a valores pequeños entre 0 y 1       
        Resize((224, 224)),        
        ToTensor()  # Conversión a tensor               
    ])

    tensor_imagen = transforms(array_canal) # Aplicación de las transformaciones
    return tensor_imagen.unsqueeze(0)

def es_mri(img_array):
    alto, ancho = img_array.shape
    
    relacion = max(alto, ancho) / min(alto, ancho)
    if relacion > 2.5: # Comprobación de las dimensiones de la imagen
        return False
    
    if alto < 100 or ancho < 100: # Tamaño minimo de la imagen de Resonancia Magnetica
        return False
    
    if np.std(img_array) < 10: # Comprobación de que la imagen sea no uniforme
        return False
    
    return True

def obtener_resultado(probabilidades):
    prob_max = torch.max(probabilidades).item() # Obtención del mayor valor (la mayor probabilidad)
    clase_idx = torch.argmax(probabilidades).item() # Obtención del indice donde está el mayor valor
    clase_nombre = clases[clase_idx] # Relación del mayor valor con la clase de tumor (o sin tumor) a la que pertenece
    
    return clase_nombre, prob_max, clase_idx

ruta_css = pl.Path("style.css")
if ruta_css.exists():
    cargar_css(ruta_css)

st.set_page_config(
    page_title="SATD Onco IA",
    layout="wide"
)

# Inicializar estado para el análisis
if 'analisis_completado' not in st.session_state:
    st.session_state.analisis_completado = False

st.html("""
        <div class="main-header">
            <h1>SATD Onco IA</h1>
            <p style="color: #7f8c8d;">Sistema de apoyo en la toma de decisiones en el diagnostico de tumores cerebrales</p>
        </div>
        """)


st.subheader("Carga de imágenes")

def reset_analisis():
    st.session_state.analisis_completado = False

archivos = st.file_uploader(
    "Subir una imagen de resonancia magnética cerebral para su análisis",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    on_change=reset_analisis,
    key="file_uploader"
)

# Botón de análisis
if st.button("Iniciar análisis", use_container_width=True, type="primary"):
    if not archivos:
        st.warning("Favor de subir al menos una imagen")
    else:
        resultados = []
        
        for archivo in archivos:
            with st.spinner(f"Analizando {archivo.name}..."):
                img_temp = Image.open(archivo).convert('L')
                img_array_temp = np.array(img_temp)

                if not es_mri(img_array_temp):
                    st.warning(f"'{archivo.name}' no parece una resonancia magnética cerebral. Resultados no confiables.")
                    continue

                tensor = transformar_img(archivo) # Transformación de la imagen a un tensor

                with torch.no_grad():
                    salida = arquitectura(tensor) # Procesamiento de la imagen mediante el modelo

                probabilidades = F.softmax(salida, dim=1)[0] # Transofrmación de los logits (aquello que regresa el modelo) en probabilidades
                clase, confianza, idx = obtener_resultado(probabilidades)

                # Agregamos el resultado de cada archivo en resultados
                resultados.append({
                    'nombre': archivo.name,
                    'clase': clase,
                    'confianza': confianza,
                    'idx': idx,
                    'probabilidades': probabilidades
                })
        
        # Marcar análisis como completado
        st.session_state.analisis_completado = True
        
        if len(resultados) > 0:
            if len(resultados) != 1:
                st.success(f"Análisis completado para {len(resultados)} imágenes")
            else:
                st.success(f"Análisis completado para {len(resultados)} imagen")
            
            st.subheader("Resultados")
            
            # Solo crear columnas si hay al menos 1 resultado
            num_columnas = min(len(resultados), 3)
            cols = st.columns(num_columnas)
            
            for i, res in enumerate(resultados):
                with cols[i % num_columnas]:
                    img = Image.open(archivos[i])
                    img.thumbnail((300, 250))
                    st.image(img, caption=res['nombre'])
                    
                    # Asignación de un color según la clase
                    if res['clase'] == 'sin tumor':
                        st.success(f"**{res['clase']}**")
                    elif res['clase'] in ['glioma', 'meningioma', 'pituitario']:
                        st.warning(f"**{res['clase']}**")
                    else:
                        st.error(f"**{res['clase']}**")

                    # Muestra del % de confianza
                    st.metric("Confianza", f"{res['confianza']*100:.2f}%")
                    st.progress(res['confianza'])

                    # Muestra de las probabilidades del resto de clases
                    with st.expander("Ver todas las probabilidades"):
                        for j, nombre in enumerate(clases):
                            prob_valor = res['probabilidades'][j].item()
                            st.write(f"{nombre}: {prob_valor*100:.2f}%")
        else:
            st.warning("No se pudo analizar ninguna imagen. Asegúrate de subir imágenes de resonancia magnética cerebral válidas.")

if archivos and not st.session_state.analisis_completado:
    st.subheader("Vista previa")
    cols = st.columns(3)
    for i, archivo in enumerate(archivos):
        with cols[i % 3]:
            # Vista previa de las imagenes antes del análisis
            img = Image.open(archivo)
            img.thumbnail((300, 250))
            st.image(img, caption=archivo.name)