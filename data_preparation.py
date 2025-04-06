"""Python script for comprehensive data preparation in hematological survival analysis.

This script contains functions to prepare clinical, cytogenetic, and molecular data
for survival analysis in hematological malignancies. It includes data processing,
feature engineering, imputation, standardization, and feature selection steps
based on clinical expertise.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from feature_selector import feature_selection
from load_data import load_data


def process_cytogenetics(df):
    """Traite les données cytogénétiques."""
    # Vérifier si la colonne CYTOGENETICS existe
    if 'CYTOGENETICS' not in df.columns:
        print("Colonne CYTOGENETICS non trouvée. Création de features factices.")
        df['normal_karyotype'] = 0
        df['complex_karyotype'] = 0
        df['del_5q'] = 0
        df['del_7q'] = 0
        df['trisomy_8'] = 0
        return df
    
    # Créer des features basiques à partir des données cytogénétiques
    df['normal_karyotype'] = df['CYTOGENETICS'].str.match('^46,[XY]{2}$', case=False, na=False).astype(int)
    df['complex_karyotype'] = (df['CYTOGENETICS'].str.count('/') > 2).astype(int)
    
    # Détecter les anomalies spécifiques
    df['del_5q'] = df['CYTOGENETICS'].str.contains('del\(5q\)|-5', case=False, regex=True, na=False).astype(int)
    df['del_7q'] = df['CYTOGENETICS'].str.contains('del\(7q\)|-7', case=False, regex=True, na=False).astype(int)
    df['trisomy_8'] = df['CYTOGENETICS'].str.contains(r'\+8', case=False, regex=True, na=False).astype(int)
    
    return df


def process_molecular(clinical_df, molecular_df):
    """Traite les données moléculaires et les fusionne avec les données cliniques."""
    if molecular_df.shape[1] <= 1:  # Seulement la colonne ID
        print("Pas de données moléculaires à traiter")
        return clinical_df
    
    # Créer des features pour chaque gène
    gene_features = {}
    gene_df = pd.DataFrame({'ID': clinical_df['ID'].unique()})
    gene_df.set_index('ID', inplace=True)
    
    # Identifier les colonnes numériques dans molecular_df
    numeric_cols = []
    for col in molecular_df.columns:
        if col != 'ID':
            try:
                # Essayer de convertir en numérique
                molecular_df[col] = pd.to_numeric(molecular_df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                print(f"Colonne {col} non numérique, ignorée")
    
    print(f"Colonnes numériques dans les données moléculaires: {numeric_cols}")
    
    # Traiter chaque colonne numérique
    for col in numeric_cols:
        # Calculer la présence du gène pour chaque ID
        presence = molecular_df.groupby('ID')[col].max().fillna(0)
        gene_df[f'has_{col}'] = (presence > 0).astype(int)
        gene_df[f'{col}_vaf'] = presence
    
    # Réinitialiser l'index pour la fusion
    gene_df.reset_index(inplace=True)
    
    # Fusionner avec les données cliniques
    merged_df = clinical_df.merge(gene_df, on='ID', how='left')
    
    # Remplir les valeurs manquantes
    gene_cols = [col for col in merged_df.columns if col.startswith('has_') or col.endswith('_vaf')]
    merged_df[gene_cols] = merged_df[gene_cols].fillna(0)
    
    return merged_df


def process_cytogenetics_expert(df):
    """Process cytogenetic data with expert knowledge based on ELN 2022 guidelines."""
    # Caryotype normal
    df['normal_karyotype'] = df['CYTOGENETICS'].str.match('^46,[XY]{2}$', case=False, na=False).astype(int)
    
    # Classification pronostique selon ELN 2022 (European LeukemiaNet)
    # Favorable
    df['t_8_21'] = df['CYTOGENETICS'].str.contains('t\(8;21\)', case=False, regex=True, na=False).astype(int)
    df['inv_16'] = df['CYTOGENETICS'].str.contains('inv\(16\)|t\(16;16\)', case=False, regex=True, na=False).astype(int)
    df['t_15_17'] = df['CYTOGENETICS'].str.contains('t\(15;17\)', case=False, regex=True, na=False).astype(int)
    df['favorable_cytogenetics'] = ((df['t_8_21'] + df['inv_16'] + df['t_15_17']) > 0).astype(int)
    
    # Adverse
    df['complex_karyotype'] = (df['CYTOGENETICS'].str.count('/') > 2).astype(int)
    df['monosomy_karyotype'] = df['CYTOGENETICS'].str.contains('-7|-5|del\(5q\)|del\(7q\)|3q|inv\(3\)|t\(3;3\)|t\(6;9\)|t\(9;22\)', 
                                                             case=False, regex=True, na=False).astype(int)
    df['del_5q'] = df['CYTOGENETICS'].str.contains('del\(5q\)|-5', case=False, regex=True, na=False).astype(int)
    df['del_7q'] = df['CYTOGENETICS'].str.contains('del\(7q\)|-7', case=False, regex=True, na=False).astype(int)
    df['del_17p'] = df['CYTOGENETICS'].str.contains('del\(17p\)|-17', case=False, regex=True, na=False).astype(int)
    df['adverse_cytogenetics'] = ((df['complex_karyotype'] + df['monosomy_karyotype'] + 
                                  df['del_5q'] + df['del_7q'] + df['del_17p']) > 0).astype(int)
    
    # Intermediate (ni favorable ni adverse)
    df['trisomy_8'] = df['CYTOGENETICS'].str.contains(r'\+8', case=False, regex=True, na=False).astype(int)
    df['trisomy_21'] = df['CYTOGENETICS'].str.contains(r'\+21', case=False, regex=True, na=False).astype(int)
    df['other_trisomy'] = df['CYTOGENETICS'].str.contains(r'\+\d', case=False, regex=True, na=False).astype(int)
    df['other_trisomy'] = (df['other_trisomy'] - df['trisomy_8'] - df['trisomy_21']).clip(lower=0)
    
    df['intermediate_cytogenetics'] = ((~df['favorable_cytogenetics'].astype(bool)) & 
                                      (~df['adverse_cytogenetics'].astype(bool))).astype(int)
    
    # Nombre total d'anomalies
    df['total_abnormalities'] = df['CYTOGENETICS'].str.count('/')
    df['total_abnormalities'] = df['total_abnormalities'].fillna(0)
    
    # Risque global basé sur la cytogénétique (score pondéré)
    df['cytogenetic_risk_score'] = (df['favorable_cytogenetics'] * (-2) + 
                                   df['intermediate_cytogenetics'] * 0 + 
                                   df['adverse_cytogenetics'] * 2 + 
                                   df['complex_karyotype'] * 1)
    
    return df


def create_clinical_features(data):
    """Create advanced clinical features relevant for hematological prognosis."""
    # Ratio neutrophiles/lymphocytes (NLR) - marqueur d'inflammation
    data['lymphocytes'] = data['WBC'] - data['ANC'] - data['MONOCYTES']
    data['lymphocytes'] = data['lymphocytes'].clip(lower=0.01)  # Éviter les valeurs négatives
    data['NLR'] = data['ANC'] / data['lymphocytes']
    
    # Ratio plaquettes/lymphocytes (PLR) - autre marqueur d'inflammation
    data['PLR'] = data['PLT'] / data['lymphocytes']
    
    # Indice de blast (blast index) - mesure de la charge tumorale
    data['blast_index'] = data['BM_BLAST'] * data['WBC'] / 100
    
    # Indicateurs cliniques binaires
    data['anemia'] = (data['HB'] < 10).astype(int)
    data['thrombocytopenia'] = (data['PLT'] < 100).astype(int)
    data['leukocytosis'] = (data['WBC'] > 25).astype(int)
    data['leukopenia'] = (data['WBC'] < 4).astype(int)
    
    # Score de risque clinique
    data['clinical_risk_score'] = (
        (data['BM_BLAST'] > 20).astype(int) * 2 +
        data['anemia'] * 1 +
        data['thrombocytopenia'] * 1 +
        data['leukocytosis'] * 1
    )
    
    # Ajouter l'âge comme facteur de risque
    data['age_risk'] = (data['AGE'] > 60).astype(int) * 2
    data['clinical_risk_score'] += data['age_risk']
    
    return data


def create_combined_risk_score(data):
    """Create combined risk score integrating clinical, cytogenetic and molecular data."""
    # Score de risque combiné
    if all(col in data.columns for col in ['clinical_risk_score', 'cytogenetic_risk_score']):
        # Version de base
        data['combined_risk_score'] = data['clinical_risk_score'] * 0.4 + data['cytogenetic_risk_score'] * 0.6
        
        # Ajouter le score moléculaire s'il existe
        if 'molecular_risk_score' in data.columns:
            data['combined_risk_score'] = (
                data['clinical_risk_score'] * 0.3 +
                data['cytogenetic_risk_score'] * 0.4 +
                data['molecular_risk_score'] * 0.3
            )
    
    return data


def process_molecular_data_expert(maf_df):
    """Process molecular data with expert knowledge of hematological malignancies."""
    # Agrégation basique par patient
    patient_mutations = maf_df.groupby('ID').agg({
        'GENE': 'count',
        'VAF': ['mean', 'max', 'min', 'std', 'median']
    }).fillna(0)
    
    patient_mutations.columns = ['total_mutations', 'mean_vaf', 'max_vaf', 'min_vaf', 'std_vaf', 'median_vaf']
    
    # Classification des gènes selon leur impact pronostique en hématologie
    # Basé sur les guidelines ELN 2022
    
    # Gènes de mauvais pronostic
    adverse_genes = ['TP53', 'ASXL1', 'RUNX1', 'EZH2', 'SRSF2', 'DNMT3A', 'U2AF1', 'SF3B1', 'ZRSR2']
    
    # Gènes de bon pronostic
    favorable_genes = ['NPM1', 'CEBPA', 'IDH2', 'GATA2']
    
    # Gènes à impact variable/intermédiaire
    intermediate_genes = ['FLT3', 'IDH1', 'TET2', 'NRAS', 'KRAS', 'KIT', 'PTPN11']
    
    # Tous les gènes d'intérêt
    all_important_genes = adverse_genes + favorable_genes + intermediate_genes
    
    # Création de features pour chaque catégorie de gènes
    patient_mutations['adverse_mutations'] = 0
    patient_mutations['favorable_mutations'] = 0
    patient_mutations['intermediate_mutations'] = 0
    
    # Ajouter des colonnes pour tous les gènes importants, même s'ils ne sont pas présents
    for gene in all_important_genes:
        # Initialiser la colonne has_gene à 0 pour tous les patients
        patient_mutations[f'has_{gene}'] = 0
        patient_mutations[f'{gene}_vaf'] = 0
        
        # Si le gène est présent dans les données, mettre à jour les valeurs
        if gene in maf_df['GENE'].values:
            # Présence de mutation dans le gène
            gene_mutations = maf_df[maf_df['GENE'] == gene].groupby('ID').size().reset_index(name=f'has_{gene}')
            gene_mutations[f'has_{gene}'] = 1
            
            # Fusion avec le DataFrame principal
            patient_mutations = patient_mutations.reset_index().merge(
                gene_mutations, on='ID', how='left'
            ).set_index('ID')
            
            # Remplir les valeurs manquantes (patients sans mutation dans ce gène)
            patient_mutations[f'has_{gene}'] = patient_mutations[f'has_{gene}'].fillna(0)
            
            # VAF maximale pour ce gène
            gene_vaf = maf_df[maf_df['GENE'] == gene].groupby('ID')['VAF'].max().reset_index(name=f'{gene}_vaf')
            
            # Fusion avec le DataFrame principal
            patient_mutations = patient_mutations.reset_index().merge(
                gene_vaf, on='ID', how='left'
            ).set_index('ID')
            
            # Remplir les valeurs manquantes
            patient_mutations[f'{gene}_vaf'] = patient_mutations[f'{gene}_vaf'].fillna(0)
            
            # Incrémenter le compteur de la catégorie appropriée
            if gene in adverse_genes:
                patient_mutations['adverse_mutations'] += patient_mutations[f'has_{gene}']
            elif gene in favorable_genes:
                patient_mutations['favorable_mutations'] += patient_mutations[f'has_{gene}']
            elif gene in intermediate_genes:
                patient_mutations['intermediate_mutations'] += patient_mutations[f'has_{gene}']
    
    # Calcul de scores moléculaires
    patient_mutations['molecular_risk_score'] = (
        patient_mutations['adverse_mutations'] * 2 - 
        patient_mutations['favorable_mutations'] * 2 + 
        patient_mutations['intermediate_mutations'] * 0.5
    )
    
    # Hétérogénéité des VAF
    patient_mutations['vaf_heterogeneity'] = patient_mutations['std_vaf'] / (patient_mutations['mean_vaf'] + 0.01)
    
    # Charge mutationnelle
    patient_mutations['mutation_burden'] = patient_mutations['total_mutations'] / 50  # Normalisation
    
    return patient_mutations


def prepare_data_expert(apply_feature_selection=True, kaggle=False):
    """Préparation experte des données avec features cliniquement pertinentes."""
    # Chargement des données brutes avec la fonction load_data
    X_train, y_train, X_test = load_data()
    # Charger les données moléculaires si disponibles
    try:
        if kaggle:
            train_molecular_path = '/kaggle/input/benchmark-hematology/train_molecular_data.csv'
        else:
            train_molecular_path = 'train_molecular_data.csv'
        
        train_molecular = pd.read_csv(train_molecular_path)
        
        # Traitement des données moléculaires
        molecular_features = process_molecular_data_expert(train_molecular)
        
        # Fusion avec les données cliniques
        X_train = X_train.merge(molecular_features, on='ID', how='left')
        
        # Remplir les valeurs manquantes pour les patients sans données moléculaires
        molecular_cols = molecular_features.columns
        X_train[molecular_cols] = X_train[molecular_cols].fillna(0)
    except:
        print("Warning: Molecular data not found or could not be processed.")
    
    # Traitement des données cytogénétiques
    X_train_processed = process_cytogenetics_expert(X_train.copy())
    X_test_processed = process_cytogenetics_expert(X_test.copy())
    
    # Définir les features cliniques de base
    numeric_features = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'AGE']
    
    # Imputation des valeurs manquantes pour les features numériques de base
    imputer = SimpleImputer(strategy='median')
    X_train_processed[numeric_features] = imputer.fit_transform(X_train_processed[numeric_features])
    X_test_processed[numeric_features] = imputer.transform(X_test_processed[numeric_features])
    
    # Création de features cliniques avancées
    X_train_processed = create_clinical_features(X_train_processed)
    X_test_processed = create_clinical_features(X_test_processed)
    
    # Création de score de risque combiné
    X_train_processed = create_combined_risk_score(X_train_processed)
    X_test_processed = create_combined_risk_score(X_test_processed)
    
    # Remplir les valeurs manquantes pour les features moléculaires
    molecular_features = [col for col in X_train_processed.columns 
                         if col not in numeric_features + ['ID', 'CENTER', 'CYTOGENETICS']]
    X_train_processed[molecular_features] = X_train_processed[molecular_features].fillna(0)
    X_test_processed[molecular_features] = X_test_processed[molecular_features].fillna(0)
    
    # Standardisation des features numériques
    scaler = StandardScaler()
    all_numeric_features = [col for col in X_train_processed.columns 
                           if col not in ['ID', 'CENTER', 'CYTOGENETICS', 'OS_STATUS', 'OS_YEARS']]
    X_train_processed[all_numeric_features] = scaler.fit_transform(X_train_processed[all_numeric_features])
    X_test_processed[all_numeric_features] = scaler.transform(X_test_processed[all_numeric_features])
    
    # Application de la sélection de features si demandé
    if apply_feature_selection:
        X_train_processed = feature_selection(X_train_processed, test_set=False)
        X_test_processed = feature_selection(X_test_processed, test_set=True)
    
    return X_train_processed, y_train, X_test_processed


def prepare_data():
    """Prépare les données pour l'analyse de survie."""
    # Charger les données
    train_clinical, y_train, test_clinical, train_molecular, test_molecular = load_data()
    
    # Traiter les données cytogénétiques
    X_train = process_cytogenetics(train_clinical.copy())
    X_test = process_cytogenetics(test_clinical.copy())
    
    # Traiter les données moléculaires
    X_train = process_molecular(X_train, train_molecular)
    X_test = process_molecular(X_test, test_molecular)
    
    # Définir les features numériques
    potential_numeric_features = ['BM_BLAST', 'WBC', 'HB', 'PLT', 'AGE']
    numeric_features = [f for f in potential_numeric_features if f in X_train.columns]
    
    if not numeric_features:
        print("Aucune feature numérique standard trouvée. Recherche d'autres colonnes numériques...")
        # Trouver toutes les colonnes numériques
        numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
        # Exclure les colonnes spéciales
        numeric_features = [col for col in numeric_features if not col.startswith('has_') and 
                           not col.endswith('_vaf') and col != 'ID' and 
                           col != 'status' and col != 'time']
        print(f"Features numériques trouvées: {numeric_features}")
    
    if numeric_features:
        # Imputer les valeurs manquantes
        imputer = SimpleImputer(strategy='median')
        X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = imputer.transform(X_test[numeric_features])
        
        # Standardiser les features numériques
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    return X_train, y_train, X_test
