import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
#  CINF104 Aprendizaje de Máquinas — Fase 1
#  DRG Prediction at Hospital El Pino
#  Grupo 5 — Universidad Andrés Bello — 2026
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CINF104 — DRG Prediction Hospital El Pino")
    print("  Grupo 5 — UNAB 2026")
    print("=" * 60)

    # ── 1. Cargar dataset ────────────────────────────────────────
    print("\n[1/5] Cargando dataset...")
    # El separador real del CSV es punto y coma
    df = pd.read_csv('dataset_elpino.csv', sep=';')
    print(f"      Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # ── 2. EDA ───────────────────────────────────────────────────
    print("\n[2/5] Generando gráficos EDA...")

    # Nombres reales de columnas
    COL_EDAD  = 'Edad en años'
    COL_SEXO  = 'Sexo (Desc)'
    COL_DIAG  = 'Diag 01 Principal (cod+des)'
    COL_GRD   = 'GRD'

    # Estadísticas descriptivas
    print("\n  --- Estadísticas descriptivas ---")
    print(f"  Registros totales    : {len(df)}")
    print(f"  Edad media           : {df[COL_EDAD].mean():.1f} años")
    print(f"  Desviación estándar  : {df[COL_EDAD].std():.1f} años")
    print(f"  Edad mínima          : {df[COL_EDAD].min():.0f} años")
    print(f"  Edad máxima          : {df[COL_EDAD].max():.0f} años")
    print(f"  Outliers (>110 años) : {(df[COL_EDAD] > 110).sum()} registros")
    print(f"  Distribución sexo    :")
    sexo_counts = df[COL_SEXO].value_counts()
    for sexo, count in sexo_counts.items():
        print(f"    {sexo}: {count} ({count/len(df)*100:.1f}%)")
    print(f"  Clases GRD únicas    : {df[COL_GRD].nunique()}")
    print(f"  Diagnósticos únicos  : {df[COL_DIAG].nunique()}")

    # Gráfico 1: Distribución de Edad
    plt.figure(figsize=(9, 5))
    sns.histplot(df[COL_EDAD], bins=30, kde=True, color='skyblue')
    plt.title('Distribución de Edad de los Pacientes (Hospital El Pino)')
    plt.xlabel('Edad en Años')
    plt.ylabel('Cantidad de Pacientes')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('grafico_edad.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("\n  Gráfico guardado: grafico_edad.png")

    # Gráfico 2: Top 10 GRDs
    plt.figure(figsize=(12, 5))
    top10 = df[COL_GRD].value_counts().head(10)
    top10.plot(kind='bar', color='salmon')
    plt.title('Top 10 GRDs más frecuentes (Desbalance de Clases)')
    plt.xlabel('Código GRD')
    plt.ylabel('Frecuencia (Cantidad de Pacientes)')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('grafico_top10_grds.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  Gráfico guardado: grafico_top10_grds.png")

    # Gráfico 3: Distribución por Sexo
    plt.figure(figsize=(5, 4))
    sexo_counts.plot(kind='bar', color=['#FF9999', '#66B2FF'])
    plt.title('Distribución por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Cantidad de Pacientes')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('grafico_sexo.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  Gráfico guardado: grafico_sexo.png")

    # ── 3. Preprocesamiento ──────────────────────────────────────
    print("\n[3/5] Preprocesando datos...")

    # Filtrar registros con GRD válido (sin guión)
    df_clean = df[df[COL_GRD].astype(str) != '-'].copy()
    df_clean = df_clean.dropna(subset=[COL_GRD])
    print(f"      Registros tras limpieza: {len(df_clean)}")

    # Selección de features del baseline
    df_baseline = df_clean[[COL_EDAD, COL_SEXO, COL_DIAG, COL_GRD]].copy()

    # Imputación de valores faltantes (ninguno en estas columnas, pero por robustez)
    df_baseline[COL_EDAD] = df_baseline[COL_EDAD].fillna(df_baseline[COL_EDAD].median())
    df_baseline[COL_SEXO] = df_baseline[COL_SEXO].fillna('Desconocido')
    df_baseline[COL_DIAG] = df_baseline[COL_DIAG].fillna('-')

    # Label Encoding
    le_sexo = LabelEncoder()
    le_diag = LabelEncoder()
    le_grd  = LabelEncoder()

    df_baseline['Sexo_encoded'] = le_sexo.fit_transform(df_baseline[COL_SEXO].astype(str))
    df_baseline['Diag01_encoded'] = le_diag.fit_transform(df_baseline[COL_DIAG].astype(str))
    df_baseline['GRD_encoded'] = le_grd.fit_transform(df_baseline[COL_GRD].astype(str))

    # Guardar dataset limpio
    df_baseline.to_csv('dataset_elpino_limpio.csv', index=False)
    print("      Dataset limpio guardado: dataset_elpino_limpio.csv")

    # ── 4. Entrenamiento de modelos ──────────────────────────────
    print("\n[4/5] Entrenando modelos...")

    X = df_baseline[['Edad en años', 'Sexo_encoded', 'Diag01_encoded']]
    y = df_baseline['GRD_encoded']

    # Split 80/20 estratificado
    # Filtrar clases con menos de 2 registros (no se puede hacer split)
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    mask = y.isin(valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    removed = len(X) - len(X_filtered)
    print(f"      Clases con <2 registros eliminadas: {removed} registros ({df[COL_GRD].nunique() - len(valid_classes)} clases)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
    print(f"      Train: {len(X_train)} registros | Test: {len(X_test)} registros")

    resultados = []

    # --- Modelo 1: Random Forest ---
    print("\n  Entrenando Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    t_rf = round(time.time() - t0, 1)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf  = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    print(f"  → Accuracy: {acc_rf:.4f} | F1-Score: {f1_rf:.4f} | Tiempo: {t_rf}s")
    resultados.append({'Modelo': 'Random Forest', 'Accuracy': acc_rf, 'F1-Score (W)': f1_rf, 'Tiempo (s)': t_rf})

    # --- Modelo 2: MLP ---
    print("\n  Entrenando MLP (Neural Network)...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu',
                        solver='adam', max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    t_mlp = round(time.time() - t0, 1)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp  = f1_score(y_test, y_pred_mlp, average='weighted', zero_division=0)
    print(f"  → Accuracy: {acc_mlp:.4f} | F1-Score: {f1_mlp:.4f} | Tiempo: {t_mlp}s")
    resultados.append({'Modelo': 'MLP (Neural Network)', 'Accuracy': acc_mlp, 'F1-Score (W)': f1_mlp, 'Tiempo (s)': t_mlp})

    # --- Curva de Loss MLP ---
    plt.figure(figsize=(8, 4))
    plt.plot(mlp.loss_curve_, color='royalblue', linewidth=2, marker='o', markersize=4)
    plt.title('Gráfico Loss vs Epochs (MLP Classifier)')
    plt.xlabel('Epochs (Iteraciones)')
    plt.ylabel('Loss (Pérdida)')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig('grafico_loss_mlp.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  Gráfico guardado: grafico_loss_mlp.png")

    # --- Modelo 3: Decision Tree ---
    print("\n  Entrenando Decision Tree...")
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt.fit(X_train, y_train)
    t_dt = round(time.time() - t0, 1)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt  = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    print(f"  → Accuracy: {acc_dt:.4f} | F1-Score: {f1_dt:.4f} | Tiempo: {t_dt}s")
    resultados.append({'Modelo': 'Decision Tree', 'Accuracy': acc_dt, 'F1-Score (W)': f1_dt, 'Tiempo (s)': t_dt})

    # --- Modelo 4: Logistic Regression ---
    print("\n  Entrenando Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    t_lr = round(time.time() - t0, 1)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr  = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    print(f"  → Accuracy: {acc_lr:.4f} | F1-Score: {f1_lr:.4f} | Tiempo: {t_lr}s")
    resultados.append({'Modelo': 'Logistic Regression', 'Accuracy': acc_lr, 'F1-Score (W)': f1_lr, 'Tiempo (s)': t_lr})

    # ── 5. Resultados finales ────────────────────────────────────
    print("\n[5/5] Generando resultados finales...")

    # Tabla comparativa
    df_resultados = pd.DataFrame(resultados)
    df_resultados['Accuracy'] = df_resultados['Accuracy'].apply(lambda x: f"{x:.4f}")
    df_resultados['F1-Score (W)'] = df_resultados['F1-Score (W)'].apply(lambda x: f"{x:.4f}")
    print("\n  --- Tabla Comparativa de Modelos ---")
    print(df_resultados.to_string(index=False))

    # Matriz de confusión (Top 10 GRDs - Random Forest)
    top10_encoded = [le_grd.transform([g])[0] for g in top10.index
                     if g in le_grd.classes_]
    top10_labels  = [le_grd.inverse_transform([e])[0] for e in top10_encoded]
    # Filtrar test set a solo top 10
    mask = y_test.isin(top10_encoded)
    y_test_top10 = y_test[mask]
    y_pred_top10 = pd.Series(y_pred_rf)[mask.values]

    short_labels = [l.split(' - ')[1][:20] if ' - ' in l else l[:20] for l in top10_labels]

    cm = confusion_matrix(y_test_top10, y_pred_top10, labels=top10_encoded)
    fig, ax = plt.subplots(figsize=(12, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    plt.title('Matriz de Confusión (Top 10 GRDs) — Random Forest')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig('grafico_confusion_matrix.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  Gráfico guardado: grafico_confusion_matrix.png")

    print("\n" + "=" * 60)
    print("  ✅ Script completado exitosamente")
    print("  Archivos generados:")
    print("    - dataset_elpino_limpio.csv")
    print("    - grafico_edad.png")
    print("    - grafico_top10_grds.png")
    print("    - grafico_sexo.png")
    print("    - grafico_loss_mlp.png")
    print("    - grafico_confusion_matrix.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
