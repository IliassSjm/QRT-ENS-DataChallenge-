import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from boruta import BorutaPy

def define_feature_sets(X_train):
    """Définit différents ensembles de features à tester"""
    # Exclure les colonnes non-numériques
    exclude_cols = ['ID', 'CENTER', 'CYTOGENETICS']
    potential_features = [col for col in X_train.columns if col not in exclude_cols]
    
    # Définition des ensembles de features
    clinical_features = [
        'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
        'anemia', 'thrombocytopenia', 'leukocytosis', 'leukopenia',
        'NLR', 'PLR', 'blast_index', 'severe_anemia', 'severe_thrombocytopenia',
        'severe_leukocytosis', 'clinical_risk_score'
    ]

    cytogenetic_features = [
        'normal_karyotype', 'favorable_cytogenetics', 'adverse_cytogenetics', 
        'intermediate_cytogenetics', 'complex_karyotype', 'monosomy_karyotype',
        'del_5q', 'del_7q', 'del_17p', 'trisomy_8', 'trisomy_21',
        'total_abnormalities', 'cytogenetic_risk_score'
    ]

    molecular_features = [
        'total_mutations', 'mean_vaf', 'max_vaf', 'std_vaf', 'median_vaf', 'min_vaf',
        'adverse_mutations', 'favorable_mutations', 'intermediate_mutations',
        'molecular_risk_score', 'mutation_burden', 'vaf_heterogeneity'
    ]
    
    interaction_features = [
        'age_blast_interaction', 'tumor_burden', 'cytopenia_index',
        'log_WBC', 'log_PLT'
    ]

    # Ajouter les features de gènes spécifiques
    gene_features = [col for col in potential_features if col.startswith('has_') or 
                    (col.endswith('_vaf') and col not in ['mean_vaf', 'max_vaf', 'std_vaf', 'median_vaf', 'min_vaf'])]

    # Features combinées
    combined_features = ['combined_risk_score']

    # Ensembles de features à tester
    feature_sets = {
        "Clinique": [f for f in clinical_features if f in potential_features],
        "Cytogénétique": [f for f in cytogenetic_features if f in potential_features],
        "Moléculaire": [f for f in molecular_features if f in potential_features],
        "Interactions": [f for f in interaction_features if f in potential_features],
        "Gènes spécifiques": [f for f in gene_features if f in potential_features],
        "Clinique + Cytogénétique": [f for f in clinical_features + cytogenetic_features if f in potential_features],
        "Clinique + Moléculaire": [f for f in clinical_features + molecular_features if f in potential_features],
        "Cytogénétique + Moléculaire": [f for f in cytogenetic_features + molecular_features if f in potential_features],
        "Score combiné": [f for f in clinical_features + cytogenetic_features + molecular_features + combined_features if f in potential_features],
        "Tous les features": potential_features
    }
    
    return feature_sets

def select_optimal_features(X, y, n_features=60):
    """Sélectionne les features optimales en combinant plusieurs méthodes"""
    # Exclure les colonnes non-numériques
    exclude_cols = ['ID', 'CENTER', 'CYTOGENETICS']
    numeric_cols = [col for col in X.columns if col not in exclude_cols]
    
    # Conversion de y pour les méthodes de régression
    y_numeric = np.array([time if status else time * 1.5 for status, time in y])
    
    # 1. Sélection par information mutuelle
    selector_mi = SelectKBest(mutual_info_regression, k=min(n_features, len(numeric_cols)))
    selector_mi.fit(X[numeric_cols], y_numeric)
    mi_scores = selector_mi.scores_
    mi_features = [numeric_cols[i] for i in np.argsort(mi_scores)[-n_features:]]
    
    # 2. Sélection par F-test
    selector_f = SelectKBest(f_regression, k=min(n_features, len(numeric_cols)))
    selector_f.fit(X[numeric_cols], y_numeric)
    f_scores = selector_f.scores_
    f_features = [numeric_cols[i] for i in np.argsort(f_scores)[-n_features:]]
    
    # 3. Sélection par Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X[numeric_cols], y_numeric)
    rf_importances = rf.feature_importances_
    rf_features = [numeric_cols[i] for i in np.argsort(rf_importances)[-n_features:]]
    
    # 4. Sélection par Lasso
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X[numeric_cols], y_numeric)
    lasso_importances = np.abs(lasso.coef_)
    lasso_features = [numeric_cols[i] for i in np.argsort(lasso_importances)[-n_features:]]
    
    # 5. Sélection par Boruta (optionnel, peut être lent)
    try:
        rf_boruta = RandomForestRegressor(n_estimators=100, random_state=42)
        boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=42)
        boruta.fit(X[numeric_cols].values, y_numeric)
        boruta_features = [numeric_cols[i] for i in np.where(boruta.support_)[0]]
    except:
        boruta_features = []
    
    # Combinaison des résultats
    all_features = mi_features + f_features + rf_features + lasso_features + boruta_features
    feature_counts = {}
    
    for feature in all_features:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1
    
    # Sélection des features qui apparaissent dans au moins 2 méthodes
    selected_features = [feature for feature, count in feature_counts.items() if count >= 2]
    
    # Si nous avons moins de n_features, ajoutons les features les plus importantes selon RF
    if len(selected_features) < n_features:
        remaining_features = [f for f in rf_features if f not in selected_features]
        selected_features.extend(remaining_features[:n_features - len(selected_features)])
    
    # Limiter à n_features
    final_features = selected_features[:min(n_features, len(selected_features))]
    
    # Ajouter ID pour le suivi
    if 'ID' in X.columns:
        final_features.append('ID')
    
    return final_features

def feature_selection(X, test_set=False):
    """Applique la sélection de features aux données"""
    # Si c'est l'ensemble de test, on retourne simplement les colonnes
    if test_set:
        return X
    
    # # Sinon, on effectue la sélection de features
    # best_features = [
    #     'combined_risk_score', 'molecular_risk_score', 'cytogenetic_risk_score', 
    #     'clinical_risk_score', 'total_mutations', 'adverse_mutations', 
    #     'has_FLT3_mutation', 'has_NPM1_mutation', 'has_DNMT3A_mutation', 
    #     'has_IDH1_mutation', 'has_IDH2_mutation', 'has_TET2_mutation', 
    #     'has_ASXL1_mutation', 'has_RUNX1_mutation', 'has_TP53_mutation', 
    #     'BM_BLAST', 'WBC', 'HB', 'PLT', 'blast_index', 'tumor_burden',
    #     'cytopenia_index', 'NLR', 'PLR', 'anemia', 'thrombocytopenia',
    #     'leukocytosis', 'complex_karyotype', 'normal_karyotype',
    #     'favorable_cytogenetics', 'adverse_cytogenetics'
    # ]
    best_features = [
        # Scores composites
        'combined_risk_score', 'molecular_risk_score', 'cytogenetic_risk_score', 
        'clinical_risk_score', 
        
        # Charge mutationnelle
        'total_mutations', 'adverse_mutations', 'weighted_mutation_load',
        'high_impact_mutation_count', 'mutation_diversity_index',
        
        # Mutations spécifiques importantes en leucémie myéloïde
        'has_FLT3_mutation', 'has_NPM1_mutation', 'has_DNMT3A_mutation', 
        'has_IDH1_mutation', 'has_IDH2_mutation', 'has_TET2_mutation', 
        'has_ASXL1_mutation', 'has_RUNX1_mutation', 'has_TP53_mutation',
        'has_CEBPA_mutation', 'has_NRAS_mutation', 'has_KRAS_mutation',
        'has_KIT_mutation', 'has_PTPN11_mutation', 'has_U2AF1_mutation',
        'has_SF3B1_mutation', 'has_SRSF2_mutation',
        
        # Co-occurrences de mutations importantes
        'FLT3_NPM1_co', 'DNMT3A_NPM1_co', 'DNMT3A_FLT3_co', 'TP53_complex_co',
        
        # Paramètres cliniques de base
        'BM_BLAST', 'WBC', 'HB', 'PLT', 'ANC', 'MONOCYTES',
        
        # Indices cliniques dérivés
        'blast_index', 'tumor_burden', 'cytopenia_index', 
        'NLR', 'PLR', 'MLR', 'ALC',
        'anemia', 'thrombocytopenia', 'leukocytosis', 'neutropenia',
        'monocytosis', 'WBC_to_PLT', 'ANC_to_WBC', 'HB_PLT_ratio',
        'log_WBC', 'log_PLT', 'sqrt_BM_BLAST',
        
        # Caractéristiques cytogénétiques
        'complex_karyotype', 'normal_karyotype',
        'favorable_cytogenetics', 'adverse_cytogenetics',
        'monosomy_7', 'trisomy_8', 'del_5q', 'del_7q', 'inv_16', 't_8_21', 't_15_17',
        
        # Voies de signalisation
        'RTK_RAS_pathway_active', 'chromatin_modifiers_active', 
        'DNA_methylation_active', 'tumor_suppressors_active',
        'spliceosome_pathway_active', 'cohesin_complex_active',
        
        # Interactions clinico-moléculaires
        'blast_mutation_interaction', 'cytogenetic_molecular_risk',
        'age_mutation_interaction', 'age_cytogenetic_interaction',
        
        # Caractéristiques démographiques (si disponibles)
        'AGE', 'SEX', 'SEX_BM_BLAST', 'SEX_HB'
    ]
    #best_features = [
    #     # Scores composites
    #     'combined_risk_score', 'molecular_risk_score', 'cytogenetic_risk_score', 
    #     'clinical_risk_score', 'composite_risk_score',
        
    #     # Charge mutationnelle
    #     'total_mutations', 'adverse_mutations', 'weighted_mutation_load',
    #     'high_impact_mutation_count', 'mutation_diversity_index',
    #     'mutation_count', 'total_impact_score',
        
    #     # Mutations spécifiques importantes en leucémie myéloïde
    #     'has_FLT3_mutation', 'has_NPM1_mutation', 'has_DNMT3A_mutation', 
    #     'has_IDH1_mutation', 'has_IDH2_mutation', 'has_TET2_mutation', 
    #     'has_ASXL1_mutation', 'has_RUNX1_mutation', 'has_TP53_mutation',
    #     'has_CEBPA_mutation', 'has_NRAS_mutation', 'has_KRAS_mutation',
    #     'has_KIT_mutation', 'has_PTPN11_mutation', 'has_U2AF1_mutation',
    #     'has_SF3B1_mutation', 'has_SRSF2_mutation', 'has_JAK2_mutation',
    #     'has_CALR_mutation', 'has_MPL_mutation', 'has_EZH2_mutation',
    #     'has_BCOR_mutation', 'has_STAG2_mutation', 'has_PHF6_mutation',
    #     'has_WT1_mutation', 'has_GATA2_mutation', 'has_CBL_mutation',
        
    #     # Co-occurrences de mutations importantes
    #     'FLT3_NPM1_co', 'DNMT3A_NPM1_co', 'DNMT3A_FLT3_co', 'TP53_complex_co',
    #     'IDH1_NPM1_co', 'IDH2_NPM1_co', 'TET2_ASXL1_co',
        
    #     # Paramètres cliniques de base
    #     'BM_BLAST', 'WBC', 'HB', 'PLT', 'ANC', 'MONOCYTES',
        
    #     # Indices cliniques dérivés
    #     'blast_index', 'tumor_burden', 'cytopenia_index', 
    #     'NLR', 'PLR', 'MLR', 'ALC',
    #     'anemia', 'thrombocytopenia', 'leukocytosis', 'neutropenia',
    #     'monocytosis', 'WBC_to_PLT', 'ANC_to_WBC', 'HB_PLT_ratio',
    #     'log_WBC', 'log_PLT', 'sqrt_BM_BLAST', 'WBC_squared',
    #     'PLT_cubed', 'HB_squared', 'BM_BLAST_cubed',
    #     'blood_health_score', 'myeloid_index', 'erythroid_index',
        
    #     # Caractéristiques cytogénétiques
    #     'complex_karyotype', 'normal_karyotype', 'Complex',
    #     'favorable_cytogenetics', 'adverse_cytogenetics', 'intermediate_cytogenetics',
    #     'monosomy_5', 'monosomy_7', 'trisomy_8', 'trisomy_21',
    #     'del_5q', 'del_7q', 'del_17p', 'del_20q',
    #     'inv_3', 'inv_16', 't_8_21', 't_9_22', 't_15_17', 't_6_9',
    #     'MLL_rearrangement', 'abn_3q', 'abn_11q',
        
    #     # Voies de signalisation
    #     'RTK_RAS_pathway_active', 'chromatin_modifiers_active', 
    #     'DNA_methylation_active', 'tumor_suppressors_active',
    #     'spliceosome_pathway_active', 'cohesin_complex_active',
    #     'signaling_pathway_score', 'epigenetic_pathway_score',
        
    #     # Interactions clinico-moléculaires
    #     'blast_mutation_interaction', 'cytogenetic_molecular_risk',
    #     'age_mutation_interaction', 'age_cytogenetic_interaction',
    #     'blast_mutation_count', 'WBC_mutation_count', 'PLT_mutation_count',
    #     'HB_mutation_count', 'blast_cytogenetic_interaction',
        
    #     # Caractéristiques démographiques (si disponibles)
    #     'AGE', 'SEX', 'SEX_BM_BLAST', 'SEX_HB', 'AGE_BM_BLAST',
    #     'AGE_WBC', 'AGE_PLT', 'AGE_HB',
        
    #     # Variables de traitement (si disponibles)
    #     'prior_treatment', 'treatment_response', 'time_to_treatment',
        
    #     # Variables spécifiques au centre
    #     'CENTER_1', 'CENTER_2', 'CENTER_3', 'CENTER_4', 'CENTER_5',
        
    #     # Caractéristiques temporelles (si disponibles)
    #     'diagnosis_to_sample_time', 'follow_up_duration',
        
    #     # Caractéristiques VAF (Variant Allele Frequency)
    #     'mean_VAF', 'max_VAF', 'VAF_heterogeneity', 'clonal_dominance',
    #     'subclonal_complexity', 'FLT3_VAF', 'NPM1_VAF', 'TP53_VAF',
        
    #     # Caractéristiques de l'effet des mutations
    #     'missense_count', 'nonsense_count', 'frameshift_count', 'splice_count',
    #     'HIGH_impact_ratio', 'MODERATE_impact_ratio'
    # ]

    
    # Filtrer pour ne garder que les colonnes qui existent
    best_features = [col for col in best_features if col in X.columns]
    
    # Ajouter ID pour le suivi
    if 'ID' in X.columns:
        best_features.append('ID')
    
    return X[best_features]