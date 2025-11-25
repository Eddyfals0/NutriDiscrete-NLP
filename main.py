import json
import os
import torch
from transformers import pipeline
from collections import Counter
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm # Barra de carga profesional
from datetime import datetime

# --- 1. ARREGLAR EL ERROR DE NLTK Y CONFIGURAR ---
print("Configurando entorno...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True) # ESTO CORRIGE TU ERROR ANTERIOR
except:
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
palabras_ruido = {'study', 'results', 'data', 'shown', 'using', 'analysis', 'found', 'abstract', 'review', 'however', 'significant', 'associated', 'suggest', 'observed', 'conclusion'}
stop_words.update(palabras_ruido)

# --- 2. ACTIVAR GPU ---
# Verificamos si PyTorch detecta la tarjeta gráfica
usar_gpu = torch.cuda.is_available()
device = 0 if usar_gpu else -1

print(f"\n{'='*40}")
if usar_gpu:
    print(f"🚀 MODO TURBO ACTIVADO: Usando GPU {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NO DETECTADA: Se usará CPU (será más lento).")
    print("   (Asegúrate de haber instalado torch con soporte CUDA)")
print(f"{'='*40}\n")

# Cargamos el modelo. batch_size=16 procesa 16 papers simultáneamente
print("Cargando modelo (BART-Large)...")
clasificador = pipeline("zero-shot-classification", 
                        model="facebook/bart-large-mnli", 
                        device=device, 
                        batch_size=16) # <--- LA CLAVE DE LA VELOCIDAD

mis_temas = ["nutrition", "longevity", "health", "strength", "mobility"]
particiones_contenido = {tema: [] for tema in mis_temas}

# --- 3. RUTA DEL ARCHIVO ---
ruta_entrada = r"C:\Users\Eduar\Documents\Universidad\TAREAS\SEMESTRE_7\IA\NutriDiscrete NLP\Datos\nutricion_1000_fuentes.json"
carpeta_base = os.path.dirname(ruta_entrada)
ruta_salida = os.path.join(carpeta_base, "nutricion_procesada_gpu.json")

# --- 4. PROCESAMIENTO INTELIGENTE ---
try:
    print(f"Leyendo archivo: {ruta_entrada}")
    with open(ruta_entrada, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    # PASO A: PREPARAR DATOS
    # Extraemos solo los textos válidos para enviarlos en masa a la GPU
    textos_para_procesar = []
    items_referencia = [] # Guardamos punteros a los objetos originales

    print("Preparando lotes de datos...")
    for item in datos:
        try:
            txt = item['content']['abstract']
            if txt and len(txt) > 20: # Validar que no esté vacío
                textos_para_procesar.append(txt)
                items_referencia.append(item)
            else:
                # Marcar vacíos
                if 'nlp_processing' not in item: item['nlp_processing'] = {}
                item['nlp_processing']['status'] = "skipped_empty"
        except KeyError:
             if 'nlp_processing' not in item: item['nlp_processing'] = {}
             item['nlp_processing']['status'] = "error_missing_key"

    total = len(textos_para_procesar)
    print(f"Se procesarán {total} documentos válidos en la GPU.")

    # PASO B: INFERENCIA MASIVA (Aquí ocurre la magia)
    print("\nIniciando clasificación masiva...")
    
    # tqdm envuelve el proceso para mostrar la barra de carga
    resultados = []
    for output in tqdm(clasificador(textos_para_procesar, candidate_labels=mis_temas), total=total):
        resultados.append(output)

    # PASO C: UNIR RESULTADOS
    print("Integrando resultados al JSON...")
    
    for i, res in enumerate(resultados):
        item = items_referencia[i] # Recuperamos el objeto original
        texto = textos_para_procesar[i]
        
        tema_ganador = res['labels'][0]
        score = res['scores'][0]

        # Escribimos en el objeto original
        if 'nlp_processing' not in item: item['nlp_processing'] = {}
        
        item['nlp_processing']['target_partition'] = tema_ganador.upper()[:3]
        item['nlp_processing']['ai_detected_topic'] = tema_ganador
        item['nlp_processing']['confidence_score'] = round(score, 4)
        item['nlp_processing']['status'] = "processed"

        # Guardamos para patrones
        particiones_contenido[tema_ganador].append(texto)

    # --- 5. ANÁLISIS DE PATRONES Y ESTADÍSTICAS ---
    def extraer_topicos(lista_textos, topk=5):
        texto_completo = " ".join(lista_textos).lower()
        palabras = nltk.word_tokenize(texto_completo)
        palabras_limpias = [p for p in palabras if p.isalnum() and p not in stop_words and len(p) > 3]
        return Counter(palabras_limpias).most_common(topk)

    # Estadísticas básicas
    total_documentos_en_archivo = len(datos)
    total_validos = len(textos_para_procesar)
    processed_count = sum(1 for d in datos if d.get('nlp_processing', {}).get('status') == 'processed')
    skipped_empty = sum(1 for d in datos if d.get('nlp_processing', {}).get('status') == 'skipped_empty')
    error_missing = sum(1 for d in datos if d.get('nlp_processing', {}).get('status') == 'error_missing_key')

    # Distribución por tema y confianza promedio
    distribucion = {}
    confidencias = {tema: [] for tema in particiones_contenido.keys()}
    for item in datos:
        meta = item.get('nlp_processing', {})
        tema = meta.get('ai_detected_topic')
        conf = meta.get('confidence_score')
        if tema and conf is not None:
            distribucion[tema] = distribucion.get(tema, 0) + 1
            confidencias[tema].append(conf)

    # Top palabras por tema
    top_por_tema = {}
    for tema, textos in particiones_contenido.items():
        if textos:
            top_por_tema[tema] = extraer_topicos(textos, topk=8)

    # Top palabras globales
    textos_procesados = [t for t in textos_para_procesar]
    top_global = extraer_topicos(textos_procesados, topk=15) if textos_procesados else []

    # Construir reporte en Markdown
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    md_lines = []
    md_lines.append(f"# Informe de Procesamiento - NutriDiscrete NLP\n")
    md_lines.append(f"**Fecha:** {now}\n")
    md_lines.append("## Resumen Numérico")
    md_lines.append(f"- Documentos en archivo: **{total_documentos_en_archivo}**")
    md_lines.append(f"- Documentos válidos procesados: **{total_validos}**")
    md_lines.append(f"- Procesados correctamente: **{processed_count}**")
    md_lines.append(f"- Skipped (vacíos): **{skipped_empty}**")
    md_lines.append(f"- Errores (missing key): **{error_missing}**\n")

    md_lines.append("## Distribución por Tema (conteo y % sobre válidos)")
    for tema, cnt in distribucion.items():
        pct = (cnt / total_validos * 100) if total_validos else 0
        avg_conf = sum(confidencias[tema]) / len(confidencias[tema]) if confidencias[tema] else 0
        md_lines.append(f"- **{tema}**: {cnt} documentos — {pct:.2f}% — Confianza media: {avg_conf:.3f}")

    md_lines.append("\n## Top palabras por Tema (top 8)")
    for tema, top in top_por_tema.items():
        md_lines.append(f"### {tema}")
        for word, c in top:
            md_lines.append(f"- {word}: {c}")

    md_lines.append("\n## Top palabras Globales (top 15)")
    for word, c in top_global:
        md_lines.append(f"- {word}: {c}")

    md_lines.append("\n---\nGenerado por NutriDiscrete NLP")

    reporte_md = "\n".join(md_lines)

    # Guardar JSON procesado
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=4, ensure_ascii=False)

    # Guardar el informe Markdown en la misma carpeta
    ruta_md = os.path.join(carpeta_base, 'informe_procesamiento.md')
    with open(ruta_md, 'w', encoding='utf-8') as fmd:
        fmd.write(reporte_md)

    # Mostrar resumen en consola
    print(f"\n✅ ¡PROCESO COMPLETADO! Archivo guardado en: {ruta_salida}")
    print(f"✅ Informe Markdown generado en: {ruta_md}")

except Exception as e:
    print(f"\n❌ ERROR CRÍTICO: {e}")
    import traceback
    traceback.print_exc()