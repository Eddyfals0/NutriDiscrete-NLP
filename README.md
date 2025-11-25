# NutriDiscrete-NLP
Pipeline NLP: Clasificación semántica masiva (Zero-Shot/BART) y análisis de frecuencias con aceleración GPU.
# 📋 NutriDiscrete-NLP - Análisis Completo del Proyecto

## 📌 Resumen Ejecutivo

**NutriDiscrete-NLP** es un pipeline de procesamiento de lenguaje natural (NLP) acelerado por GPU que clasifica automáticamente documentos científicos sobre nutrición en 5 categorías temáticas usando clasificación Zero-Shot con el modelo BART.

**Tecnologías principales:**
- PyTorch con GPU CUDA
- Transformers (facebook/bart-large-mnli)
- NLTK para tokenización
- TensorFlow (para modelos secundarios)

---

## 📁 Estructura del Proyecto

```
NutriDiscrete-NLP/
├── main.py                              # ⭐ Pipeline principal de clasificación NLP
├── borrar.py                            # Modelo Siamese para matching (TensorFlow)
├── requirements.txt                     # Dependencias del proyecto
├── README.md                            # Descripción breve del proyecto
├── LICENSE                              # Licencia del proyecto
├── .git/                                # Repositorio Git
├── .gitattributes                       # Configuración de Git
├── .venv/ & env/                        # Entornos virtuales de Python
└── Datos/                               # Carpeta de datos
    ├── nutricion_1000_fuentes.json      # 📥 Input: 1094 documentos fuentes
    ├── nutricion_procesada_gpu.json     # 📤 Output: Documentos clasificados
    └── informe_procesamiento.md         # 📊 Informe de estadísticas
```

---

## 🎯 Funcionalidad de `main.py`

### **Propósito**
Procesar masivamente ~1000 documentos científicos sobre nutrición clasificándolos en 5 temas usando una red neuronal preentrenada.

### **Flujo de Ejecución**

#### 1. **Configuración Inicial**
```python
- Descarga recursos NLTK (stopwords, punkt)
- Detecta disponibilidad de GPU (CUDA)
- Carga modelo BART-Large (facebook/bart-large-mnli)
- Define 5 temas: nutrition, longevity, health, strength, mobility
```

#### 2. **Preparación de Datos**
```python
- Lee JSON con 1094 documentos
- Extrae abstracts válidos (texto > 20 caracteres)
- Genera lista de referencia de objetos originales
- Total válidos: 1094 documentos
```

#### 3. **Inferencia Masiva (GPU)**
```python
- Batch size: 16 (procesa 16 documentos simultáneamente)
- Modelo: facebook/bart-large-mnli (clasificación Zero-Shot)
- Entrada: Abstracto de documento + 5 labels temáticos
- Salida: Tema clasificado + score de confianza (0-1)
```

#### 4. **Post-Procesamiento**
```python
- Integra resultados en JSON original
- Agrega metadatos:
  * target_partition: Código del tema (NUT, LON, HEA, STR, MOB)
  * ai_detected_topic: Tema detectado
  * confidence_score: Score 0-1
  * status: "processed" / "skipped_empty" / "error"
```

#### 5. **Análisis de Patrones**
```python
- Tokenización con NLTK
- Limpieza: stop-words + ruido específico del dominio
- Top palabras por tema (top 8)
- Top palabras globales (top 15)
- Conteos de documentos por tema
- Confianza promedio por tema
```

#### 6. **Generación de Reportes**
```python
- Archivo JSON: nutricion_procesada_gpu.json
- Reporte Markdown: informe_procesamiento.md
- Métricas: Conteos, distribución, estadísticas
```

### **Salidas Generadas**

#### 📊 Distribución Temática (Resultados Reales)
| Tema | Documentos | % | Confianza Media |
|------|-----------|---|-----------------|
| nutrition | 377 | 34.46% | 0.546 |
| health | 332 | 30.35% | 0.488 |
| longevity | 172 | 15.72% | 0.542 |
| strength | 152 | 13.89% | 0.473 |
| mobility | 61 | 5.58% | 0.462 |

#### 🔤 Top Palabras Globales
1. diet (1575)
2. sarcopenia (1433)
3. muscle (1307)
4. metabolic (960)
5. risk (942)
6. disease (812)
7. aging (800)
8. nutrition (755)

---

## 🤖 Funcionalidad de `borrar.py`

### **Propósito**
Demostración de un modelo Siamese con embeddings para matching usuario-empresa considerando:
- Skills técnicos
- Nivel de idioma (Inglés: Básico, Intermedio, Avanzado)
- Certificaciones
- Edad

### **Características**

#### 1. **Diccionario de Tags Expandido**
```python
Tecnologías: Python, Java, SQL, C++, JavaScript, etc.
Idiomas: 
  - Inglés (Básico, Intermedio, Avanzado)
  - Español, Francés, Alemán, Chino, Portugués

Pesos por certificación:
  - CON certificado: 1.0
  - SIN certificado: 0.5
```

#### 2. **Arquitectura Siamese**
```
Usuario/Empresa
    ↓
[IDs Tags] → Embedding → Ponderación
[Pesos]    
[Geo]      → Concatenación → Dense(16) → Dense(4) → L2 Norm
[Edad]     
```

#### 3. **Datos de Ejemplo**
```
Usuarios (5):
- Sr. Pro (35 años, Inglés Avanzado certificado)
- Jr. Novato (22 años, Inglés Básico sin cert)
- Veterano Manager (50 años)

Empresas (5):
- Lead Position (requiere Inglés Avanzado)
- Becario (acepta Inglés Básico)
- Call Center (solo Inglés)
- Dirección (requiere Liderazgo + edad madura)
```

#### 4. **Salidas**
- Matriz de similitud coseno (usuarios vs empresas)
- Top 2 mejores matches por usuario
- Top 2 mejores candidatos por empresa
- Gráfico PCA 2D con líneas de matches

---

## 📦 Dependencias por Módulo

### **main.py requiere:**
```
✅ torch (GPU support)
✅ transformers (BART model)
✅ nltk (NLP utilities)
✅ tqdm (progress bars)
✅ json (built-in)
✅ os (built-in)
✅ collections (built-in)
✅ datetime (built-in)
```

### **borrar.py requiere:**
```
✅ numpy
✅ tensorflow & keras
✅ matplotlib
✅ scikit-learn (PCA, cosine_similarity)
```

### **Todos requieren:**
```
✅ certifi, charset-normalizer, requests
✅ huggingface-hub (descargar modelos)
✅ safetensors
```

---

## 🚀 Guía de Instalación y Ejecución

### **Paso 1: Crear Entorno Virtual**
```powershell
python -m venv env
env\Scripts\Activate.ps1
```

### **Paso 2: Instalar Dependencias**
```powershell
pip install -r requirements.txt
```

### **Paso 3: GPU Support (Opcional pero recomendado)**
```powershell
# Si tienes NVIDIA CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Paso 4: Ejecutar Pipeline**
```powershell
# Clasificación de documentos
python main.py

# Demostración de matching (opcional)
python borrar.py
```

---

## 📊 Archivos de Datos

### **Entrada: `nutricion_1000_fuentes.json`**
```json
{
  "content": {
    "abstract": "texto del documento..."
  },
  "nlp_processing": {} // Se agrega durante ejecución
}
```

### **Salida: `nutricion_procesada_gpu.json`**
```json
{
  "content": {
    "abstract": "texto del documento..."
  },
  "nlp_processing": {
    "target_partition": "NUT",
    "ai_detected_topic": "nutrition",
    "confidence_score": 0.7234,
    "status": "processed"
  }
}
```

### **Reporte: `informe_procesamiento.md`**
```markdown
# Resumen Numérico
- Total documentos: 1094
- Procesados: 1094
- Skipped: 0
- Errores: 0

# Distribución por Tema + Confianza media
# Top palabras por Tema
# Top palabras Globales
```

---

## 🔧 Configuraciones Importantes

### En `main.py`:

```python
# Temas de clasificación
mis_temas = ["nutrition", "longevity", "health", "strength", "mobility"]

# Batch size para GPU (aumentar si hay memoria disponible)
batch_size=16  # Para procesar más documentos en paralelo

# Detección automática de GPU
device = 0 if torch.cuda.is_available() else -1

# Stop-words personalizados para dominio de nutrición
palabras_ruido = {'study', 'results', 'data', 'abstract', ...}

# Rutas de entrada/salida (ajustar según tu sistema)
ruta_entrada = r"C:\Users\...\nutricion_1000_fuentes.json"
```

### En `borrar.py`:

```python
# Número de tags máximo
NUM_TAGS = 100000008
EMBEDDING_DIM = 8

# Pesos por certificación (Inglés)
sin_certificado = 0.5
con_certificado = 1.0

# Epochs de entrenamiento
epochs=150
```

---

## 📈 Métricas y KPIs

### **Rendimiento**
- **Documentos procesados**: 1094 ✅
- **Tasa de éxito**: 100% (0 errores)
- **Velocidad**: Depende de GPU (~1000 docs en 5-30 minutos según hardware)

### **Calidad**
- **Confianza media global**: ~0.51 (muy aceptable para Zero-Shot)
- **Distribución balanceada**: Sí (nutrition 34%, health 30%, otros 36%)
- **Cobertura temática**: 5 categorías bien distribuidas

### **Datos**
- **Documentos fuentes**: 1094
- **Documentos válidos**: 1094 (100%)
- **Campos procesados**: abstract
- **Metadatos generados**: 4 por documento

---

## 🐛 Manejo de Errores

| Error | Solución |
|-------|----------|
| `GPU no detectada` | Instalar torch con CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `NLTK resources missing` | Script descarga automáticamente en ejecución |
| `File not found` | Verificar ruta en `ruta_entrada` |
| `Memory error` | Reducir `batch_size` de 16 a 8 o 4 |
| `Model download fails` | Verificar conexión a internet y espacio en disco (~6GB) |

---

## 💡 Mejoras Futuras

- [ ] Agregar validación cruzada de clasificaciones
- [ ] Implementar fine-tuning con datos etiquetados
- [ ] Exportar resultados a múltiples formatos (CSV, Parquet)
- [ ] Dashboard interactivo con Streamlit
- [ ] Pruebas unitarias automatizadas
- [ ] CI/CD con GitHub Actions
- [ ] Documentación de API REST

---

## 📝 Notas de Desarrollo

- **Lenguaje**: Python 3.8+
- **Versión del Modelo**: facebook/bart-large-mnli (descargar ~1.6GB)
- **GPU mínima**: 2GB VRAM (recomendado 4GB+)
- **CPU mínima**: Funciona pero muy lento (~1h para 1000 docs)
- **Licencia**: Revisar LICENSE
- **Repositorio**: `.git` configurado

---

## 👤 Contacto y Soporte

Este proyecto es parte de: **NutriDiscrete-NLP**
- GitHub: Eddyfals0
- Rama principal: main

---

*Generado: 2025-11-24 | Versión: 1.0*
