from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score



cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _get_final_estimator(estimator):
    """Return the final estimator object for Pipelines (sklearn or imblearn) or the estimator itself.
    This is used to inspect whether the final step supports predict_proba.
    """
    try:
        # sklearn / imblearn pipelines expose named_steps as an ordered dict
        if hasattr(estimator, 'named_steps') and isinstance(getattr(estimator, 'named_steps'), dict) and len(estimator.named_steps):
            return list(estimator.named_steps.values())[-1]
        # Generic fallback for objects exposing 'steps' as a list of (name, est) tuples
        if hasattr(estimator, 'steps') and isinstance(getattr(estimator, 'steps'), (list, tuple)) and len(estimator.steps):
            return estimator.steps[-1][1]
    except Exception:
        pass
    return estimator


def _ensure_predict_proba(model):
    """Raise a clear error if the estimator (or its final step) does not provide predict_proba.

    The message explains how to fix the problem (use a probabilistic estimator or wrap with
    CalibratedClassifierCV)."""
    if hasattr(model, 'predict_proba'):
        return
    final = _get_final_estimator(model)
    final_has = hasattr(final, 'predict_proba')
    msg = (
        f"Estimator '{type(model).__name__}' does not implement 'predict_proba'. "
        f"Final estimator is '{type(final).__name__}' which {'does' if final_has else 'does NOT'} implement 'predict_proba'.\n\n"
        "Possible fixes:\n"
        "- Use a probabilistic final estimator (e.g., RandomForestClassifier, LogisticRegression).\n"
        "- Wrap the final estimator with sklearn.calibration.CalibratedClassifierCV to obtain calibrated probabilities.\n"
        "  Example:\n"
        "    from sklearn.calibration import CalibratedClassifierCV\n"
        "    calibrated = CalibratedClassifierCV(base_estimator=your_clf)\n"
        "    pipeline = make_pipeline(..., calibrated)\n"
    )
    raise AttributeError(msg)

def train_model_evaluate(X, y, model, cv_strategy=cv_strategy, use_cv=True, X_test=None, y_test=None):
    """
    Se use_cv=True: esegue la K-Fold Cross Validation.
    Se use_cv=False: addestra il modello sull'intero set fornito e lo valuta.
    """
    scoring_metrics = {
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_score': make_scorer(f1_score, average='macro', zero_division=0),
        'auc_roc': scorer_auc_alz_dlb,
        'pra_auc': scorer_prauc_alz_dlb
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
        print(f"AUC-ROC (Alz vs Levy):     {np.mean(cv_results['test_auc_roc']):.4f} ± {np.std(cv_results['test_auc_roc']):.4f}")
        print(f"PRAUC Alz vs Levy:      {np.mean(cv_results['test_pra_auc']):.4f} ± {np.std(cv_results['test_pra_auc']):.4f}")
        
        # Per calcolare un ECE corretto in CV, usiamo le probabilità Out-Of-Fold
        # Verifichiamo che l'estimator supporti predict_proba
        
        _ensure_predict_proba(model)
        y_proba = cross_val_predict(estimator=model, X=X, y=y, cv=cv_strategy, method='predict_proba', n_jobs=-1)
         # Calcolo ECE L2 (Binarizziamo la y per la Classe 1. Modifica == 1 in == 2 per DLB)
        y_true_bin = (y == 2).astype(int)
        print(f"ECE L2 (Classe 3): {calculate_ece_l2(y_true_bin, y_proba[:, 2]):.4f}")
    
        return y_true_bin, y_proba  # Ritorna le etichette binarie e le probabilità della classe 1
        
    else:
        print(f"Addestramento in corso (Fit Singolo) per {type(model).__name__}...")
        model.fit(X, y)

        if X_test is  None and y_test is  None:
            raise Exception("nessun test set passato")
        y_pred = model.predict(X_test)
        # Verifichiamo che il modello (o il suo final estimator) supporti predict_proba
        _ensure_predict_proba(model)
        y_proba = model.predict_proba(X_test)
        
        print("\n=== RISULTATI (Addestramento Singolo) ===")
        print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall (Macro):    {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1-Score (Macro):  {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"AUC-ROC (OvO) Alz vs Levy:     {alz_vs_dlb_auc_func(y_test, y_proba):.4f}")
        print(f"PRAUC Alz vs Levy:      {alz_vs_dlb_pr_auc_func(y_test, y_proba):.4f}")
         # Calcolo ECE L2 (Binarizziamo la y per la Classe 1. Modifica == 1 in == 2 per DLB)
        y_true_bin = (y_test == 2).astype(int)
        print(f"ECE L2 (Classe 3): {calculate_ece_l2(y_true_bin, y_proba[:, 2]):.4f}")
    
        return y_true_bin, y_proba, y_pred  # Ritorna le etichette binarie e le probabilità della classe 1
   


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


def alz_vs_dlb_auc_func(y_true, y_proba):
    """
    Calcola l'AUC-ROC isolando esclusivamente i pazienti con Alzheimer (1) e Lewy Body (2).
    Ignora completamente la classe dei Sani (0) per evitare AUC gonfiate.
    """
    # 1. Creiamo una maschera per isolare SOLO i pazienti con demenza
    mask = (y_true == 1) | (y_true == 2)
    
    # 2. Filtriamo i veri target e le probabilità predette
    y_true_filtered = y_true[mask]
    y_proba_filtered = y_proba[mask]
    
    # Controllo di sicurezza nel caso un fold fosse anomalmente vuoto
    if len(np.unique(y_true_filtered)) < 2:
        return 0.0
        
    # 3. Binarizziamo il problema per il calcolo dell'AUC
    # Lewy Body (2) diventa la classe Positiva (1)
    # Alzheimer (1) diventa la classe Negativa (0)
    y_true_binary = (y_true_filtered == 2).astype(int)
    
    # 4. Estraiamo le probabilità che il modello aveva assegnato alla classe Lewy Body (Indice 2)
    # Rinominalizziamo le probabilità in modo che la somma tra prob_ALZ e prob_DLB faccia 1
    prob_alz = y_proba_filtered[:, 1]
    prob_dlb = y_proba_filtered[:, 2]
    
    # Normalizzazione Pairwise: prob_DLB / (prob_ALZ + prob_DLB)
    # Evita divisioni per zero se il modello ha previsto 100% Sano per un malato
    somma_prob = prob_alz + prob_dlb
    somma_prob[somma_prob == 0] = 1e-15 
    prob_dlb_normalized = prob_dlb / somma_prob
    
    # 5. Calcoliamo la ROC-AUC pura su questo confine decisionale
    return roc_auc_score(y_true_binary, prob_dlb_normalized)

def alz_vs_dlb_pr_auc_func(y_true, y_proba):
    mask = (y_true == 1) | (y_true == 2)
    y_true_filtered = y_true[mask]
    y_proba_filtered = y_proba[mask]
    
    if len(np.unique(y_true_filtered)) < 2:
        return 0.0
        
    y_true_binary = (y_true_filtered == 2).astype(int)
    prob_dlb_normalized = y_proba_filtered[:, 2] / (y_proba_filtered[:, 1] + y_proba_filtered[:, 2] + 1e-15)
    
    # Usiamo average_precision_score invece di roc_auc_score
    return average_precision_score(y_true_binary, prob_dlb_normalized)

def clinical_cost_loss(y_true, y_pred):
    """
    Calcola un punteggio di penalità basato sulla gravità dell'errore medico.
    L'obiettivo della GridSearch sarà MINIMIZZARE questo numero.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    
    fn_dlb = cm[2, 0] # DLB previsto come Sano
    confusione_dlb_ad = cm[2, 1] # DLB previsto come Alzheimer
    fp_dlb = cm[0, 2] + cm[1, 2] # Sani o Alzheimer previsti come DLB
    
   
    costo_totale = (10 * fn_dlb) + (5 * confusione_dlb_ad) + (fp_dlb)
    
    return costo_totale

# Trasformiamo la funzione in uno scorer di Scikit-Learn
# Dice alla Grid Search: "Vince la combinazione di parametri che ottiene il numero PIÙ BASSO"
scorer_costo_clinico = make_scorer(clinical_cost_loss, greater_is_better=False)

scorer_prauc_alz_dlb = make_scorer(alz_vs_dlb_pr_auc_func, response_method='predict_proba')

scorer_auc_alz_dlb = make_scorer(alz_vs_dlb_auc_func, response_method='predict_proba')