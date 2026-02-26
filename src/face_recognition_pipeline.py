"""
PROGETTO FINALE: Statistical Analysis of Face Recognition using SVD
STUDENTE: Angelo Mastrangelo
CORSO: Statistical and Mathematical Methods for AI

DESCRIZIONE:
Script completo per analisi statistica, Grid Search e generazione di immagini.
Configurato per salvare i risultati nella cartella 'output' della root del progetto.
"""

# --- IMPORTAZIONE LIBRERIE ---
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# --- CONFIGURAZIONE PERCORSI (BLINDATA) ---
# 1. Trova la cartella dove si trova fisicamente questo file .py (ovvero 'src')
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 2. Definisce la cartella di output salendo di un livello (..) per andare nella root
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_PATH, "..", "output"))

# 3. Crea la cartella nella root se non esiste
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[INFO] Cartella di output pronta in: {OUTPUT_DIR}")

def save_current_plot(filename):
    """Salva il grafico corrente nel percorso di output globale."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[SALVATO] {filepath}")
    plt.close()

def print_header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

# =============================================================================
# 1. DATA LOADING & SPLIT
# =============================================================================
print_header("1. DATA LOADING")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preview Dataset
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data.images[i], cmap='gray')
    plt.axis('off')
plt.suptitle("Dataset Olivetti Faces (Campioni)", fontsize=14)
save_current_plot("1_dataset_preview.png")

# =============================================================================
# 2. SVD & SCREE PLOT
# =============================================================================
print_header("2. SVD ANALYSIS")
pca_full = PCA(n_components=200, svd_solver='randomized', whiten=True, random_state=42)
pca_full.fit(X_train)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
k90 = np.argmax(cum_var >= 0.90) + 1 

# Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(cum_var, linewidth=2)
plt.axvline(k90, color='r', linestyle='--', label=f'90% Variance (k={k90})')
plt.title('Scree Plot: Explained Variance Analysis')
plt.legend(); plt.grid(True)
save_current_plot("2_scree_plot.png")

# Eigenfaces
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(pca_full.components_[i].reshape(64, 64), cmap='gray')
    plt.axis('off'); plt.title(f"PC {i+1}")
plt.suptitle(f"Top 10 Eigenfaces")
save_current_plot("3_eigenfaces.png")

# =============================================================================
# 3. ANALISI RESIDUI
# =============================================================================
print_header("3. RESIDUALS")
pca_Res = PCA(n_components=k90, whiten=True).fit(X_train)
sample = X_test[0]
recon = pca_Res.inverse_transform(pca_Res.transform(sample.reshape(1, -1)))
residual = np.abs(sample - recon.flatten())

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(sample.reshape(64,64), cmap='gray'); plt.title("Originale")
plt.subplot(1, 3, 2); plt.imshow(recon.reshape(64,64), cmap='gray'); plt.title(f"Ricostruzione (k={k90})")
plt.subplot(1, 3, 3); plt.imshow(residual.reshape(64,64), cmap='hot'); plt.title("Residuo (Errore)")
plt.colorbar(fraction=0.046, pad=0.04)
save_current_plot("4_residual_analysis.png")

# =============================================================================
# 4. GRID SEARCH
# =============================================================================
print_header("4. GRID SEARCH")
pipe_knn = Pipeline([('pca', PCA(whiten=True, random_state=42)), ('knn', KNeighborsClassifier())])
pipe_svm = Pipeline([('pca', PCA(whiten=True, random_state=42)), ('svm', SVC(class_weight='balanced'))])

param_grid_knn = {'pca__n_components': [40, k90, 100], 'knn__n_neighbors': [1, 3, 5], 'knn__weights': ['uniform', 'distance']}
param_grid_svm = {'pca__n_components': [k90, 150], 'svm__C': [1, 10, 100], 'svm__kernel': ['rbf', 'linear']}

grid_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=5, n_jobs=-1).fit(X_train, y_train)
print(f"Best KNN: {grid_knn.best_params_}")
grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5, n_jobs=-1).fit(X_train, y_train)
print(f"Best SVM: {grid_svm.best_params_}")

# Confronto
models = ['KNN', 'SVM']
scores = [grid_knn.score(X_test, y_test), grid_svm.score(X_test, y_test)]
plt.figure(figsize=(6, 5))
plt.bar(models, scores, color=['skyblue', 'salmon'])
plt.ylim(0.5, 1.0); plt.title('Model Comparison')
for i, v in enumerate(scores): plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
save_current_plot("5_model_comparison.png")

# =============================================================================
# 5. ROBUSTEZZA & DENOISING
# =============================================================================
print_header("5. ROBUSTEZZA")

def add_noise(data, mode='gaussian', amount=0.05):
    noisy = data.copy()
    if mode == 'gaussian':
        noisy += np.random.normal(0, amount, data.shape)
    elif mode == 'sp':
        n_samples, n_feat = data.shape
        for i in range(n_samples):
            n_corr = int(amount * n_feat)
            idx = np.random.choice(n_feat, n_corr, replace=False)
            noisy[i, idx[:n_corr//2]] = 0 
            noisy[i, idx[n_corr//2:]] = 1
    return np.clip(noisy, 0, 1)

# Dataset Rumorosi
X_test_gauss = add_noise(X_test, 'gaussian', 0.08)
X_test_sp = add_noise(X_test, 'sp', 0.10)

best_model = grid_svm.best_estimator_
acc_clean = best_model.score(X_test, y_test)
acc_gauss = best_model.score(X_test_gauss, y_test)
acc_sp = best_model.score(X_test_sp, y_test)

# Grafico Decadimento
plt.figure(figsize=(7, 5))
plt.plot(['Clean', 'Gauss', 'S&P'], [acc_clean, acc_gauss, acc_sp], marker='o', color='purple')
plt.ylim(0, 1.05); plt.title('Decadimento Performance')
save_current_plot("6_noise_robustness_chart.png")

# Generazione immagini per slide
sample_face = X_test[10].reshape(1, -1)
face_img = sample_face.reshape(64, 64)

noisy_gauss_sample = add_noise(sample_face, 'gaussian', 0.08).reshape(64, 64)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(face_img, cmap='gray'); plt.title("Originale"); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(noisy_gauss_sample, cmap='gray'); plt.title("Rumore Gaussiano"); plt.axis('off')
save_current_plot("visual_gaussian.png")

noisy_sp_sample = add_noise(sample_face, 'sp', 0.10).reshape(64, 64)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(face_img, cmap='gray'); plt.title("Originale"); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(noisy_sp_sample, cmap='gray'); plt.title("Rumore Sale & Pepe"); plt.axis('off')
save_current_plot("visual_sp.png")

pca_step = best_model.named_steps['pca']

# Denoising
denoised_sp = pca_step.inverse_transform(pca_step.transform(noisy_sp_sample.reshape(1, -1)))
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.imshow(face_img, cmap='gray'); plt.title("Originale"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(noisy_sp_sample, cmap='gray'); plt.title("Input S&P"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(denoised_sp.reshape(64,64), cmap='gray'); plt.title("SVD Filtering"); plt.axis('off')
save_current_plot("7_denoising_visual_sp.png")

denoised_gauss = pca_step.inverse_transform(pca_step.transform(noisy_gauss_sample.reshape(1, -1)))
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.imshow(face_img, cmap='gray'); plt.title("Originale"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(noisy_gauss_sample, cmap='gray'); plt.title("Input Gaussiano"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(denoised_gauss.reshape(64,64), cmap='gray'); plt.title("SVD Filtering"); plt.axis('off')
save_current_plot("7b_denoising_visual_gauss.png")

print(f"\n[COMPLETATO] Tutte le immagini sono state salvate con successo in: {OUTPUT_DIR}")