import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engineering import prepare_data_expert
from feature_selection import define_feature_sets, feature_selection
from models import create_ensemble_model, evaluate_feature_sets, optimize_weights
import time
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis

def print_progress(message, start_time=None):
    """Affiche un message de progression avec le temps écoulé si start_time est fourni"""
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"{message} (Temps écoulé: {elapsed:.2f}s)")
    else:
        print(message)
    return current_time

def main():
    # Chargement des données
    start_time = print_progress("Chargement des données...")
    
    df_train = pd.read_csv("./clinical_train.csv")
    df_test = pd.read_csv("./clinical_test.csv")
    maf_train = pd.read_csv("./molecular_train.csv")
    maf_test = pd.read_csv("./molecular_test.csv")
    target_df = pd.read_csv("./target_train.csv")
    
    print_progress("Données chargées", start_time)
    
    # Préparation des données
    start_time = print_progress("Préparation des données...")
    
    X_train, y_train = prepare_data_expert(df_train, maf_train, target_df, is_training=True)
    X_test = prepare_data_expert(df_test, maf_test, is_training=False)
    
    print_progress("Données préparées", start_time)
    
    # Définition des ensembles de features
    feature_sets = define_feature_sets(X_train)
    
    # Évaluation des ensembles de features
    start_time = print_progress("Évaluation des ensembles de features...")
    
    cv_results, best_features = evaluate_feature_sets(X_train, y_train, feature_sets)
    evaluate_feature_sets
    print_progress("Évaluation terminée", start_time)
    
    # Visualisation des résultats
    plt.figure(figsize=(12, 6))
    
    names = list(cv_results.keys())
    scores = [result['mean_score'] for result in cv_results.values()]
    errors = [result['std_score'] for result in cv_results.values()]
    
    # Tri par score
    sorted_indices = np.argsort(scores)
    names = [names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]
    
    plt.barh(names, scores, xerr=errors, alpha=0.7)
    plt.xlabel('Score de concordance')
    plt.title('Performance des différents ensembles de features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('cv_results.png')
    
    # Optimisation des poids
    start_time = print_progress("Optimisation des poids de l'ensemble...")
    
    best_weights, best_weight_score = optimize_weights(X_train, y_train, best_features)
    
    print_progress("Optimisation terminée", start_time)
    
    # Entraînement du modèle final
    start_time = print_progress("Entraînement du modèle final...")
    
    # Modèle Cox avec régularisation
    cox_model = CoxnetSurvivalAnalysis(
        l1_ratio=0.7,
        alpha_min_ratio=0.01,
        max_iter=1000,
        tol=1e-5
    )
    cox_model.fit(X_train[best_features], y_train)
    
    # Modèle Random Survival Forest
    rsf_model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    rsf_model.fit(X_train[best_features], y_train)
    
    # Modèle Gradient Boosting
    gb_model = GradientBoostingSurvivalAnalysis(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train[best_features], y_train)
    
    # Modèle Componentwise Gradient Boosting
    comp_gb_model = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    comp_gb_model.fit(X_train[best_features], y_train)
    
    print_progress("Modèle final entraîné", start_time)
    
    # Prédictions sur l'ensemble de test
    start_time = print_progress("Génération des prédictions...")
    
    # Prédictions de chaque modèle
    cox_pred = cox_model.predict(X_test[best_features])
    rsf_pred = rsf_model.predict(X_test[best_features])
    gb_pred = gb_model.predict(X_test[best_features])
    comp_gb_pred = comp_gb_model.predict(X_test[best_features])
    
    # Normalisation des prédictions
    cox_pred_norm = (cox_pred - cox_pred.mean()) / (cox_pred.std() + 1e-8)
    rsf_pred_norm = (rsf_pred - rsf_pred.mean()) / (rsf_pred.std() + 1e-8)
    gb_pred_norm = (gb_pred - gb_pred.mean()) / (gb_pred.std() + 1e-8)
    comp_gb_pred_norm = (comp_gb_pred - comp_gb_pred.mean()) / (comp_gb_pred.std() + 1e-8)
    
    # Application des poids optimaux
    w1, w2, w3, w4 = best_weights
    ensemble_pred = (w1 * cox_pred_norm + 
                     w2 * rsf_pred_norm + 
                     w3 * gb_pred_norm + 
                     w4 * comp_gb_pred_norm)
    
    # Création du fichier de soumission
    submission = pd.DataFrame({
        'ID': X_test.index if X_test.index.name == 'ID' else X_test['ID'],
        'risk_score': ensemble_pred
    })
    
    # Assurer que ID est la première colonne
    if 'ID' not in submission.columns:
        submission['ID'] = X_test.index if X_test.index.name == 'ID' else X_test['ID']
    
    submission.to_csv('submission.csv', index=False)
    
    print_progress("Prédictions générées et fichier de soumission créé", start_time)
    
    # Résumé
    print("\nRésumé:")
    print(f"Meilleur ensemble de features: {max(cv_results.items(), key=lambda x: x[1]['mean_score'])[0]}")
    print(f"Score CV: {max(cv_results.values(), key=lambda x: x['mean_score'])['mean_score']:.4f}")
    print(f"Nombre de features: {len(best_features)}")
    print(f"Poids optimaux: {best_weights}")
    print(f"Fichier de soumission créé: submission.csv")

if __name__ == "__main__":
    main()