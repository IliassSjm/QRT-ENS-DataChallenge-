import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

def process_cytogenetics_expert(df):
    """Traitement expert des données cytogénétiques basé sur la connaissance clinique"""
    # Caryotype normal
    df['normal_karyotype'] = df['CYTOGENETICS'].str.match('^Normal$|^46,[XY]{2}$', case=False, na=False).astype(int)
    
    # Classification pronostique selon ELN 2022 (European LeukemiaNet)
    # Favorable
    df['t_8_21'] = df['CYTOGENETICS'].str.contains('t\(8;21\)', case=False, regex=True, na=False).astype(int)
    df['inv_16'] = df['CYTOGENETICS'].str.contains('inv\(16\)|t\(16;16\)', case=False, regex=True, na=False).astype(int)
    df['t_15_17'] = df['CYTOGENETICS'].str.contains('t\(15;17\)', case=False, regex=True, na=False).astype(int)
    df['favorable_cytogenetics'] = ((df['t_8_21'] + df['inv_16'] + df['t_15_17']) > 0).astype(int)
    
    # Adverse
    df['complex_karyotype'] = (df['CYTOGENETICS'].str.count('/') > 2).astype(int) | df['CYTOGENETICS'].str.contains('Complex', case=False, na=False)
    df['monosomy_karyotype'] = df['CYTOGENETICS'].str.contains('--5|del\(5q\)|del\(7q\)|3q|inv\(3\)|t\(3;3\)|t\(6;9\)|t\(9;22\)', 
                                                             case=False, regex=True, na=False).astype(int)
    df['del_5q'] = df['CYTOGENETICS'].str.contains('del\(5q\)|-5', case=False, regex=True, na=False).astype(int)
    df['del_7q'] = df['CYTOGENETICS'].str.contains('del\(7q\)|-7', case=False, regex=True, na=False).astype(int)
    df['del_17p'] = df['CYTOGENETICS'].str.contains('del\(17p\)|-17', case=False, regex=True, na=False).astype(int)
    df['adverse_cytogenetics'] = ((df['complex_karyotype'] + df['monosomy_karyotype'] + 
                                  df['del_5q'] + df['del_7q'] + df['del_17p']) > 0).astype(int)
    
    # Intermediate (ni favorable ni adverse)
    df['trisomy_8'] = df['CYTOGENETICS'].str.contains(r'\+8', case=False, regex=True, na=False).astype(int)
    df['trisomy_21'] = df['CYTOGENETICS'].str.contains(r'\+21', case=False, regex=True, na=False).astype(int)
    df['intermediate_cytogenetics'] = ((df['normal_karyotype'] + df['trisomy_8'] + df['trisomy_21']) > 0 & 
                                      (df['favorable_cytogenetics'] + df['adverse_cytogenetics'] == 0)).astype(int)
    
    # Nombre total d'anomalies
    df['total_abnormalities'] = df['CYTOGENETICS'].str.count(',') + df['CYTOGENETICS'].str.count('/') + df['CYTOGENETICS'].str.count('\+') + df['CYTOGENETICS'].str.count('-')
    df['total_abnormalities'] = df['total_abnormalities'].fillna(0)
    
    # Score de risque cytogénétique (0 = favorable, 1 = intermédiaire, 2 = défavorable)
    df['cytogenetic_risk_score'] = df['adverse_cytogenetics'] * 2 + df['intermediate_cytogenetics'] * 1
    
    # Remplacer les valeurs manquantes par la valeur médiane
    df['cytogenetic_risk_score'] = df['cytogenetic_risk_score'].fillna(1)
    
    return df

def process_molecular_data_expert(molecular_df):
    """Traitement expert des données moléculaires"""
    # Création d'un DataFrame pour stocker les features moléculaires par patient
    patient_features = {}
    
    # Liste des gènes importants en LMA
    important_genes = [
        'FLT3', 'NPM1', 'DNMT3A', 'IDH1', 'IDH2', 'TET2', 'ASXL1', 'RUNX1', 
        'TP53', 'CEBPA', 'NRAS', 'KRAS', 'KIT', 'PTPN11', 'WT1', 'GATA2', 
        'STAG2', 'RAD21', 'SMC1A', 'SMC3', 'EZH2', 'BCOR', 'SRSF2', 'SF3B1', 
        'U2AF1', 'ZRSR2', 'PHF6', 'IKZF1', 'IKZF3', 'CDKN2A', 'RB1'
    ]
    
    # Gènes de bon pronostic
    favorable_genes = ['NPM1', 'CEBPA', 'IDH2']
    
    # Gènes de mauvais pronostic
    adverse_genes = ['FLT3', 'TP53', 'ASXL1', 'RUNX1', 'DNMT3A', 'KRAS', 'NRAS']
    
    # Initialisation du dictionnaire pour chaque patient
    for patient_id in molecular_df['ID'].unique():
        patient_features[patient_id] = {
            'total_mutations': 0,
            'mean_vaf': 0,
            'max_vaf': 0,
            'std_vaf': 0,
            'median_vaf': 0,
            'min_vaf': 0,
            'vaf_heterogeneity': 0,
            'adverse_mutations': 0,
            'favorable_mutations': 0,
            'intermediate_mutations': 0,
            'mutation_burden': 0
        }
        
        # Initialisation des features pour chaque gène important
        for gene in important_genes:
            patient_features[patient_id][f'has_{gene}_mutation'] = 0
            patient_features[patient_id][f'{gene}_vaf'] = 0
    
    # Traitement des mutations pour chaque patient
    for patient_id in molecular_df['ID'].unique():
        # Filtrer les mutations pour ce patient
        patient_mutations = molecular_df[molecular_df['ID'] == patient_id]
        
        if len(patient_mutations) == 0:
            continue
        
        # Nombre total de mutations
        patient_features[patient_id]['total_mutations'] = len(patient_mutations)
        
        # Statistiques VAF
        vafs = patient_mutations['VAF'].values
        if len(vafs) > 0:
            patient_features[patient_id]['mean_vaf'] = np.mean(vafs)
            patient_features[patient_id]['max_vaf'] = np.max(vafs)
            patient_features[patient_id]['min_vaf'] = np.min(vafs)
            patient_features[patient_id]['median_vaf'] = np.median(vafs)
            patient_features[patient_id]['std_vaf'] = np.std(vafs) if len(vafs) > 1 else 0
            patient_features[patient_id]['vaf_heterogeneity'] = np.max(vafs) - np.min(vafs) if len(vafs) > 1 else 0
        
        # Mutations par gène
        for _, mutation in patient_mutations.iterrows():
            gene = mutation['GENE']
            vaf = mutation['VAF']
            
            if gene in important_genes:
                patient_features[patient_id][f'has_{gene}_mutation'] = 1
                patient_features[patient_id][f'{gene}_vaf'] = max(vaf, patient_features[patient_id][f'{gene}_vaf'])
            
            # Classification pronostique
            if gene in favorable_genes:
                patient_features[patient_id]['favorable_mutations'] += 1
            elif gene in adverse_genes:
                patient_features[patient_id]['adverse_mutations'] += 1
            else:
                patient_features[patient_id]['intermediate_mutations'] += 1
        
        # Score de risque moléculaire (0-3)
        # 0 = favorable, 1-2 = intermédiaire, 3 = défavorable
        molecular_risk = 0
        
        # Facteurs de bon pronostic
        if patient_features[patient_id]['has_NPM1_mutation'] == 1 and patient_features[patient_id]['has_FLT3_mutation'] == 0:
            molecular_risk -= 1
        if patient_features[patient_id]['has_CEBPA_mutation'] == 1:
            molecular_risk -= 1
        if patient_features[patient_id]['has_IDH2_mutation'] == 1 and patient_features[patient_id]['has_FLT3_mutation'] == 0:
            molecular_risk -= 1
        
        # Facteurs de mauvais pronostic
        if patient_features[patient_id]['has_TP53_mutation'] == 1:
            molecular_risk += 2
        if patient_features[patient_id]['has_ASXL1_mutation'] == 1:
            molecular_risk += 1
        if patient_features[patient_id]['has_RUNX1_mutation'] == 1:
            molecular_risk += 1
        if patient_features[patient_id]['has_FLT3_mutation'] == 1:
            molecular_risk += 1
        
        # Normalisation du score (0-3)
        molecular_risk = max(0, min(3, molecular_risk + 1))
        patient_features[patient_id]['molecular_risk_score'] = molecular_risk
        
        # Charge mutationnelle (nombre de mutations pondéré par VAF moyenne)
        patient_features[patient_id]['mutation_burden'] = patient_features[patient_id]['total_mutations'] * patient_features[patient_id]['mean_vaf']
    
    # Conversion en DataFrame
    molecular_processed = pd.DataFrame.from_dict(patient_features, orient='index')
    
    # Ajouter l'ID comme colonne (important pour la fusion)
    molecular_processed.reset_index(inplace=True)
    molecular_processed.rename(columns={'index': 'ID'}, inplace=True)
    
    return molecular_processed

def create_clinical_features(df):
    """Création de features cliniques avancées"""
    # Anémie (HB < 10 g/dL)
    if 'HB' in df.columns:
        df['anemia'] = (df['HB'] < 10).astype(int)
        df['severe_anemia'] = (df['HB'] < 8).astype(int)
    
    # Thrombocytopénie (PLT < 100 x 10^9/L)
    if 'PLT' in df.columns:
        df['thrombocytopenia'] = (df['PLT'] < 100).astype(int)
        df['severe_thrombocytopenia'] = (df['PLT'] < 50).astype(int)
    
    # Leucocytose (WBC > 25 x 10^9/L)
    if 'WBC' in df.columns:
        df['leukocytosis'] = (df['WBC'] > 25).astype(int)
        df['severe_leukocytosis'] = (df['WBC'] > 50).astype(int)
        df['leukopenia'] = (df['WBC'] < 4).astype(int)
    
    # Ratio neutrophiles/lymphocytes (NLR)
    if 'ANC' in df.columns and 'WBC' in df.columns:
        # Estimation des lymphocytes
        df['lymphocytes'] = df['WBC'] - df['ANC']
        df['lymphocytes'] = df['lymphocytes'].clip(lower=0.1)  # Éviter division par zéro
        df['NLR'] = df['ANC'] / df['lymphocytes']
    
    # Ratio plaquettes/lymphocytes (PLR)
    if 'PLT' in df.columns and 'WBC' in df.columns and 'ANC' in df.columns:
        df['PLR'] = df['PLT'] / df['lymphocytes']
    
    # Indice de blast (% blast * WBC)
    if 'BM_BLAST' in df.columns and 'WBC' in df.columns:
        df['blast_index'] = df['BM_BLAST'] * df['WBC'] / 100
    
    # Score de risque clinique
    risk_factors = []
    
    if 'leukocytosis' in df.columns:
        risk_factors.append('leukocytosis')
    if 'severe_anemia' in df.columns:
        risk_factors.append('severe_anemia')
    if 'severe_thrombocytopenia' in df.columns:
        risk_factors.append('severe_thrombocytopenia')
    
    if risk_factors:
        df['clinical_risk_score'] = df[risk_factors].sum(axis=1)
        # Normalisation du score entre 0 et 2
        max_score = len(risk_factors)
        df['clinical_risk_score'] = df['clinical_risk_score'] * 2 / max_score
    else:
        df['clinical_risk_score'] = 1  # Score par défaut
    
    return df

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

def prepare_data_expert(df_clinical, df_molecular, target_df=None, is_training=True):
    """Préparation experte des données avec features cliniquement pertinentes"""
    # Traitement des données cliniques
    clinical_processed = process_cytogenetics_expert(df_clinical.copy())
    
    # Traitement des données moléculaires
    molecular_processed = process_molecular_data_expert(df_molecular)
    
    # Fusion des données
    data = clinical_processed.merge(molecular_processed, on='ID', how='left')
    
    # Définir les features cliniques de base
    numeric_features = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT']
    
    # Imputation des valeurs manquantes pour les features numériques de base
    imputer = SimpleImputer(strategy='median')
    data[numeric_features] = imputer.fit_transform(data[numeric_features])
    
    # Création de features cliniques avancées
    data = create_clinical_features(data)
    
    # Création de score de risque combiné
    data = create_combined_risk_score(data)
    
    # Création de features d'interaction
    # Interaction âge et blastes
    if 'AGE' in data.columns and 'BM_BLAST' in data.columns:
        data['age_blast_interaction'] = data['AGE'] * data['BM_BLAST'] / 100
    
    # Interaction leucocytes et blastes (charge tumorale)
    if 'WBC' in data.columns and 'BM_BLAST' in data.columns:
        data['tumor_burden'] = data['WBC'] * data['BM_BLAST'] / 100
    
    # Indice de cytopénie
    if 'HB' in data.columns and 'PLT' in data.columns:
        # Normalisation par rapport aux valeurs normales
        data['cytopenia_index'] = (data['HB'] / 15) * (data['PLT'] / 150)
    
    # Features non-linéaires
    for col in ['WBC', 'PLT']:
        if col in data.columns:
            data[f'log_{col}'] = np.log1p(data[col])
    
    # Remplir les valeurs manquantes pour les features moléculaires
    molecular_features = [col for col in data.columns if col not in numeric_features + ['ID', 'CENTER', 'CYTOGENETICS']]
    data[molecular_features] = data[molecular_features].fillna(0)
    
    # Standardisation des features numériques
    scaler = StandardScaler()
    all_numeric_features = [col for col in data.columns if col not in ['ID', 'CENTER', 'CYTOGENETICS']]
    data[all_numeric_features] = scaler.fit_transform(data[all_numeric_features])
    
    if is_training and target_df is not None:
        # Nettoyage des données target
        target_df = target_df.copy()
        target_df['OS_STATUS'] = target_df['OS_STATUS'].fillna(0).astype(int)
        target_df['OS_YEARS'] = target_df['OS_YEARS'].fillna(target_df['OS_YEARS'].median())
        
        # Alignement des indices
        data = data[data['ID'].isin(target_df['ID'])]
        target_df = target_df[target_df['ID'].isin(data['ID'])]
        
        # Tri des données pour assurer l'alignement
        data = data.sort_values('ID').reset_index(drop=True)
        target_df = target_df.sort_values('ID').reset_index(drop=True)
        
        # Création du format de survie
        y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)
        return data, y
    
    return data