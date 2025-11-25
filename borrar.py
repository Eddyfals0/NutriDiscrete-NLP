import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. DICCIONARIO ACTUALIZADO (Con Inglés a 100 millones)
# ==========================================
tags_dict = {
    # --- TECNOLOGÍAS (1-1000) ---
    1: "Python", 2: "Java", 3: "SQL", 5: "Liderazgo",
    10: "C++", 11: "JavaScript", 12: "TypeScript", 13: "Go", 14: "Rust",
    20: "React", 21: "Angular", 22: "Vue", 23: "Django", 24: "FastAPI",
    30: "Docker", 31: "Kubernetes", 32: "AWS", 33: "Azure", 34: "GCP",
    
    # --- IDIOMAS (100,000,000 para dejar espacio) ---
    100000000: "Inglés Básico",
    100000001: "Inglés Intermedio", 
    100000002: "Inglés Avanzado",
    100000003: "Español Nativo",
    100000004: "Francés Básico",
    100000005: "Alemán Intermedio",
    100000006: "Chino Mandarín Avanzado",
    100000007: "Portugués Básico",
}
NUM_TAGS = 100000008 
EMBEDDING_DIM = 8

# ==========================================
# 2. DATOS SIMULADOS (Con Edad y Certificados)
# ==========================================

# Regla de Pesos para Inglés:
# Certificado SI = 1.0
# Certificado NO = 0.5 (Lo sabe, pero no hay papel)

users_data = [
    # 0. EL PRO: 35 años, Python, Inglés Avanzado Certificado (1.0)
    {"tags": [1, 100000002], "weights": [1.0, 1.0], "age": 35, "geo": [0.5, 0.5], "name": "Sr. Pro (35, Ing.Adv)"},
    
    # 1. EL JUNIOR: 22 años, Python, Inglés Básico sin cert (0.5)
    {"tags": [1, 100000000], "weights": [0.33, 0.5], "age": 22, "geo": [0.5, 0.5], "name": "Jr. Novato (22, Ing.Bas)"},
    
    # 2. EL BILINGÜE SIN EXP: 20 años, Solo Inglés Avanzado Certificado
    {"tags": [100000002], "weights": [1.0], "age": 20, "geo": [0.5, 0.5], "name": "Bilingüe (20)"},
    
    # 3. EL VETERANO: 50 años, Liderazgo, Inglés Intermedio (0.5)
    {"tags": [5, 100000001], "weights": [1.0, 0.5], "age": 50, "geo": [0.5, 0.5], "name": "Manager (50)"},
    
    # 4. FALSO EXPERTO: Dice tener Inglés Avanzado pero sin cert (0.5) y Python (0.33)
    {"tags": [1, 100000002], "weights": [0.33, 0.5], "age": 28, "geo": [0.5, 0.5], "name": "Mentiroso (28)"},
]

companies_data = [
    # 0. BUSCA PRO: Edad ideal 30-40, Requiere Inglés Avanzado (1.0)
    {"tags": [1, 100000002], "weights": [1.0, 1.0], "age": 35, "geo": [0.5, 0.5], "name": "Job: Lead (Req. Ing.Adv)"},
    
    # 1. BUSCA BECARIO: Edad ideal 20-25, Acepta Inglés Básico (1.0 de importancia al nivel basico)
    {"tags": [1, 100000000], "weights": [0.5, 1.0], "age": 22, "geo": [0.5, 0.5], "name": "Job: Becario"},
    
    # 2. CALL CENTER: Solo importa el Inglés Avanzado, edad joven
    {"tags": [100000002], "weights": [1.0], "age": 21, "geo": [0.5, 0.5], "name": "Job: Call Center"},
    
    # 3. DIRECCIÓN: Busca Liderazgo y edad madura
    {"tags": [5, 100000001], "weights": [1.0, 0.5], "age": 55, "geo": [0.5, 0.5], "name": "Job: Director"},
    
    # 4. EMPRESA INGENUA: Busca Avanzado pero acepta sin papeles
    {"tags": [1, 100000002], "weights": [1.0, 0.5], "age": 28, "geo": [0.5, 0.5], "name": "Job: Startup"},
]

# PRE-PROCESAMIENTO
def pad_data(data_list):
    ids = tf.keras.preprocessing.sequence.pad_sequences([d['tags'] for d in data_list], maxlen=3, padding='post')
    weights = tf.keras.preprocessing.sequence.pad_sequences([d['weights'] for d in data_list], maxlen=3, padding='post', dtype='float32')
    geos = np.array([d['geo'] for d in data_list], dtype='float32')
    # NORMALIZAMOS LA EDAD (Dividimos por 100)
    ages = np.array([d['age'] / 100.0 for d in data_list], dtype='float32')
    return ids, weights, geos, ages

u_ids, u_w, u_geo, u_age = pad_data(users_data)
c_ids, c_w, c_geo, c_age = pad_data(companies_data)
targets = np.ones((5, 1), dtype='float32') # Asumimos match 1-1 para el demo

# ==========================================
# 3. CONSTRUCCIÓN DE LA RED (NUEVA ENTRADA DE EDAD)
# ==========================================

def build_tower_with_age(name_suffix):
    # Entradas
    input_ids = layers.Input(shape=(3,), name=f'ids_{name_suffix}')
    input_weights = layers.Input(shape=(3,), name=f'weights_{name_suffix}')
    input_geo = layers.Input(shape=(2,), name=f'geo_{name_suffix}')
    
    # --- NUEVO: Entrada de Edad ---
    input_age = layers.Input(shape=(1,), name=f'age_{name_suffix}')

    # Procesamiento Tags (Igual que antes)
    emb_layer = layers.Embedding(NUM_TAGS, EMBEDDING_DIM, mask_zero=True)
    embeddings = emb_layer(input_ids)
    weights_reshaped = layers.Reshape((3, 1))(input_weights)
    weighted_embeddings = layers.Multiply()([embeddings, weights_reshaped])
    pooled = layers.GlobalAveragePooling1D()(weighted_embeddings)

    # --- FUSIÓN: Tags + Geo + EDAD ---
    concat = layers.Concatenate()([pooled, input_geo, input_age])

    x = layers.Dense(16, activation='relu')(concat)
    output = layers.Dense(4, activation='linear')(x) # Vector de 4 dimensiones
    output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(output)

    return models.Model(inputs=[input_ids, input_weights, input_geo, input_age], outputs=output)

# Instanciar
user_tower = build_tower_with_age("user")
comp_tower = build_tower_with_age("company")

# Modelo de Entrenamiento Siames
u_in = [layers.Input(shape=(3,)), layers.Input(shape=(3,)), layers.Input(shape=(2,)), layers.Input(shape=(1,))]
c_in = [layers.Input(shape=(3,)), layers.Input(shape=(3,)), layers.Input(shape=(2,)), layers.Input(shape=(1,))]

u_vec = user_tower(u_in)
c_vec = comp_tower(c_in)
dot = layers.Dot(axes=1)([u_vec, c_vec])
output = layers.Dense(1, activation='sigmoid')(dot)

model = models.Model(inputs=u_in + c_in, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenar (Overfitting intencional para demo)
print("Entrenando con edades y niveles de inglés...")
model.fit(
    [u_ids, u_w, u_geo, u_age, c_ids, c_w, c_geo, c_age],
    targets,
    epochs=150,
    verbose=0
)

# ==========================================
# 4. CÁLCULO DE COINCIDENCIAS MÁS CERCANAS
# ==========================================
u_vecs = user_tower.predict([u_ids, u_w, u_geo, u_age])
c_vecs = comp_tower.predict([c_ids, c_w, c_geo, c_age])

# Calcular similitud de coseno entre cada usuario y empresa
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(u_vecs, c_vecs)

print("\n" + "="*70)
print("COINCIDENCIAS MÁS CERCANAS (Por Usuario)")
print("="*70)

# Para cada usuario, encontrar las 2 empresas más cercanas
for i, user in enumerate(users_data):
    print(f"\n👤 {user['name']}")
    print(f"   Edad: {user['age']} años")
    
    # Obtener índices ordenados por similitud (descendente)
    top_indices = np.argsort(similarity_matrix[i])[::-1]
    
    for rank, comp_idx in enumerate(top_indices[:2], 1):
        similarity = similarity_matrix[i][comp_idx]
        company = companies_data[comp_idx]
        print(f"   {rank}. {company['name']} - Similitud: {similarity:.3f}")

print("\n" + "="*70)
print("COINCIDENCIAS MÁS CERCANAS (Por Empresa)")
print("="*70)

# Para cada empresa, encontrar los 2 usuarios más cercanos
for j, company in enumerate(companies_data):
    print(f"\n🏢 {company['name']}")
    print(f"   Edad ideal: {company['age']} años")
    
    # Obtener índices ordenados por similitud (descendente)
    top_indices = np.argsort(similarity_matrix[:, j])[::-1]
    
    for rank, user_idx in enumerate(top_indices[:2], 1):
        similarity = similarity_matrix[user_idx][j]
        user = users_data[user_idx]
        print(f"   {rank}. {user['name']} - Similitud: {similarity:.3f}")

# ==========================================
# 5. RESULTADO VISUAL
# ==========================================
# PCA para graficar 2D
pca = PCA(n_components=2)
all_vecs = np.concatenate([u_vecs, c_vecs])
pca_result = pca.fit_transform(all_vecs)
u_xy = pca_result[:5]
c_xy = pca_result[5:]

plt.figure(figsize=(12, 8))
plt.title("Matches influenciados por EDAD y Nivel de INGLÉS\n(Líneas = Mejores coincidencias)", fontsize=14, fontweight='bold')

# Graficar
plt.scatter(u_xy[:, 0], u_xy[:, 1], c='blue', s=200, label='Usuarios', zorder=3)
for i, d in enumerate(users_data):
    plt.annotate(d['name'], (u_xy[i, 0], u_xy[i, 1]), xytext=(0, 12), textcoords='offset points', fontsize=9, ha='center', fontweight='bold')

plt.scatter(c_xy[:, 0], c_xy[:, 1], c='red', s=200, marker='s', label='Empresas', zorder=3)
for i, d in enumerate(companies_data):
    plt.annotate(d['name'], (c_xy[i, 0], c_xy[i, 1]), xytext=(0, -22), textcoords='offset points', fontsize=9, color='darkred', ha='center', fontweight='bold')

# Líneas de mejores matches (solo el top 1)
for i in range(5):
    best_company = np.argmax(similarity_matrix[i])
    plt.plot([u_xy[i, 0], c_xy[best_company, 0]], [u_xy[i, 1], c_xy[best_company, 1]], 'g-', alpha=0.5, linewidth=2, zorder=1)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()