from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score



cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_model_evaluate(X, y, model, cv_strategy=cv_strategy, use_cv=True):
    """
    Se use_cv=True: esegue la K-Fold Cross Validation.
    Se use_cv=False: addestra il modello sull'intero set fornito e lo valuta.
    """
    scoring_metrics = {
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_score': make_scorer(f1_score, average='macro', zero_division=0),
        'auc_roc': 'roc_auc_ovr'
    }

    if use_cv:
        print(f"Addestramento in corso (5-Fold CV) per {type(model).__name__}...")
        cv_results = cross_validate(
            estimator=model, X=X, y=y, cv=cv_strategy,
            scoring=scoring_metrics, return_train_score=False
        )
        
        print("\n=== RISULTATI (Media su 5 Folds) ===")
        print(f"Precision (Macro): {np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}")
        print(f"Recall (Macro):    {np.mean(cv_results['test_recall']):.4f} ± {np.std(cv_results['test_recall']):.4f}")
        print(f"F1-Score (Macro):  {np.mean(cv_results['test_f1_score']):.4f} ± {np.std(cv_results['test_f1_score']):.4f}")
        print(f"AUC-ROC (OvR):     {np.mean(cv_results['test_auc_roc']):.4f} ± {np.std(cv_results['test_auc_roc']):.4f}")
        
        # Per calcolare un ECE corretto in CV, usiamo le probabilità Out-Of-Fold
        y_proba = cross_val_predict(estimator=model, X=X, y=y, cv=cv_strategy, method='predict_proba', n_jobs=-1)
        
    else:
        print(f"Addestramento in corso (Fit Singolo) per {type(model).__name__}...")
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        print("\n=== RISULTATI (Addestramento Singolo) ===")
        print(f"Precision (Macro): {precision_score(y, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall (Macro):    {recall_score(y, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1-Score (Macro):  {f1_score(y, y_pred, average='macro', zero_division=0):.4f}")
        print(f"AUC-ROC (OvR):     {roc_auc_score(y, y_proba, multi_class='ovr'):.4f}")

    # Calcolo ECE L2 (Binarizziamo la y per la Classe 1. Modifica == 1 in == 2 per DLB)
    y_true_bin = (y == 2).astype(int)
    print(f"ECE L2 (Classe 3): {calculate_ece_l2(y_true_bin, y_proba[:, 2]):.4f}")

    return y_true_bin, y_proba  # Ritorna le etichette binarie e le probabilità della classe 1


def generate_predictions_and_cm(X_scaled, y, model, cv_strategy=cv_strategy, use_cv=True):
    """
    Genera e stampa la matrice di confusione e il report clinico.
    """
    etichette = ['Sani (0)', 'Alzheimer (1)', 'Lewy Body (2)']
    
    if use_cv:
        print("Generazione previsioni Out-of-Fold (Validazione Interna in corso)...")
        y_pred = cross_val_predict(estimator=model, X=X_scaled, y=y, cv=cv_strategy, n_jobs=-1)
        titolo_grafico = 'Matrice di Confusione (Validazione Interna OOF)'
    else:
        print("Generazione previsioni (Senza CV)...")
        # Attenzione: se use_cv=False, si presuppone che il modello sia GIA' stato fittato 
        # (ad esempio chiamando prima train_model_evaluate con use_cv=False)
        y_pred = model.predict(X_scaled)
        titolo_grafico = 'Matrice di Confusione (Dataset Singolo)'

    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=etichette, yticklabels=etichette,
                linewidths=1, linecolor='black')

    plt.title(titolo_grafico, fontsize=14, pad=15)
    plt.ylabel('Diagnosi Reale (Medico)', fontsize=12, fontweight='bold')
    plt.xlabel(f'Previsione ({type(model).__name__})', fontsize=12, fontweight='bold')
    plt.show()

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y, y_pred, target_names=etichette))


def calculate_ece_l2(y_true_binary, y_proba_1d, n_bins=10):
    """
    Calcola l'Expected Calibration Error (Norma L2) in modo model-agnostic.
    
    Parametri:
    - y_true_binary: Array 1D di etichette reali (0 o 1).
    - y_proba_1d: Array 1D di probabilità predette (tra 0.0 e 1.0).
    - n_bins: Numero di intervalli in cui dividere le probabilità.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece_l2_squared = 0.0

    
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Gestione dell'inclusione matematica: [lower, upper) tranne l'ultimo che è [lower, upper]
        if bin_upper == 1.0:
            in_bin = (y_proba_1d >= bin_lower) & (y_proba_1d <= bin_upper)
        else:
            in_bin = (y_proba_1d >= bin_lower) & (y_proba_1d < bin_upper)
            
        prop_in_bin = np.mean(in_bin) # Peso del bin (|Bm| / N)
        
        if prop_in_bin > 0: # Evitiamo divisioni per zero nei bin vuoti
            accuracy_in_bin = np.mean(y_true_binary[in_bin])
            avg_confidence_in_bin = np.mean(y_proba_1d[in_bin])
            
            # Norma L2 (differenza al quadrato pesata)
            ece_l2_squared += prop_in_bin * (accuracy_in_bin - avg_confidence_in_bin)**2
            
    return np.sqrt(ece_l2_squared)


def plot_reliability_diagram(y_true_binary, y_proba_1d, n_bins=10, title="Reliability Diagram"):
    """
    Genera un diagramma di affidabilità (Calibration Curve) model-agnostic.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_upper == 1.0:
            in_bin = (y_proba_1d >= bin_lower) & (y_proba_1d <= bin_upper)
        else:
            in_bin = (y_proba_1d >= bin_lower) & (y_proba_1d < bin_upper)
            
        if np.any(in_bin):
            bin_accuracies.append(np.mean(y_true_binary[in_bin]))
            bin_confidences.append(np.mean(y_proba_1d[in_bin]))
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            
    # Calcolo ECE L2 per il titolo
    ece = calculate_ece_l2(y_true_binary, y_proba_1d, n_bins)
    
    plt.figure(figsize=(8, 8))
    
    # Linea della calibrazione perfetta (y = x)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibrazione Perfetta')
    
    # Plot delle accuratezze per bin
    plt.plot(bin_confidences, bin_accuracies, marker='o', linewidth=2, color='blue', label='Modello')
    
    # Disegniamo i gap (l'errore di calibrazione)
    for conf, acc in zip(bin_confidences, bin_accuracies):
        if not np.isnan(acc):
            plt.plot([conf, conf], [conf, acc], color='red', alpha=0.5, linestyle=':')
            
    plt.title(f"{title}\nECE L2: {ece:.4f}", fontsize=14, pad=15)
    plt.xlabel('Confidenza Predetta Media', fontsize=12)
    plt.ylabel('Frazione di Positivi Reali (Accuratezza)', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()