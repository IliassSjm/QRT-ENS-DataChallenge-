import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import time

# Configuration globale des modèles
GLOBAL = {
    "cox": {"run": True, "save": False, "shap": False},
    "xgb": {"run": True, "save": False, "shap": False},
    "lgbm": {"run": True, "save": False, "shap": False},
    "rsf": {"run": True, "save": False, "shap": False}
}

# Paramètres des modèles
PARAMS = {
    "size": 0.7,
    "impute": {"strategy": "median", "sex": False},
    "clinical": ["CYTOGENETICS"],
    "molecular": ["GENE"],
    "merge": ["featuretools", "gpt"],
    "additional": [],
    "xgb": {
        'loss': 'coxph',
        'max_depth': 2,
        'learning_rate': 0.05,
        'n_estimators': 335,
        'subsample': 0.55,
        'max_features': "sqrt",
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0,
        'min_impurity_decrease': 0,
        'dropout_rate': 0,
        'warm_start': False,
        'ccp_alpha': 0,
        'random_state': 126
    },
    "lgbm": {
        'max_depth': 2,
        'learning_rate': 0.05,
        'verbose': 0
    },
    "rsf": {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'n_jobs': -1,
    }
}

def create_ensemble_model(X_train, y_train, best_features):
    """Création d'un modèle d'ensemble combinant Cox, XGBoost, LightGBM et Random Survival Forest"""
    print("Création du modèle d'ensemble...")
    
    # Préparation des données
    X_train_features = X_train[best_features].copy()
    
    # Normalisation des données pour certains modèles
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_features),
        columns=X_train_features.columns,
        index=X_train_features.index
    )
    
    models = {}
    
    # 1. Modèle Cox avec régularisation
    if GLOBAL["cox"]["run"]:
        print("Entraînement du modèle Cox...")
        cox_model = CoxnetSurvivalAnalysis(
            l1_ratio=0.7,
            alpha_min_ratio=0.01,
            max_iter=1000,
            tol=1e-5
        )
        cox_model.fit(X_train_scaled, y_train)
        models["cox"] = cox_model
    
    # 2. Modèle XGBoost
    # Dans la partie XGBoost de votre fonction create_ensemble_model

    if GLOBAL["xgb"]["run"]:
        print("Entraînement du modèle XGBoost...")
        # Préparation des données pour XGBoost
        # Conversion des données de survie au format XGBoost
        status = np.array([t[0] for t in y_train])
        time = np.array([t[1] for t in y_train])
        
        # Utiliser XGBSurvivalAnalysis au lieu de XGBRegressor
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        
        # Paramètres XGBoost
        xgb_params = PARAMS["xgb"]
        
        # Création d'un modèle de survie basé sur gradient boosting
        xgb_model = GradientBoostingSurvivalAnalysis(
            loss='coxph',
            learning_rate=xgb_params['learning_rate'],
            n_estimators=xgb_params['n_estimators'],
            subsample=xgb_params['subsample'],
            max_depth=xgb_params['max_depth'],
            min_samples_split=xgb_params['min_samples_split'],
            min_samples_leaf=xgb_params['min_samples_leaf'],
            random_state=xgb_params['random_state']
        )
        
        # Entraînement
        xgb_model.fit(X_train_scaled, y_train)
        
        models["xgb"] = xgb_model
    
    # 3. Modèle LightGBM
    if GLOBAL["lgbm"]["run"]:
        print("Entraînement du modèle LightGBM...")
        # Paramètres LightGBM
        lgbm_params = PARAMS["lgbm"]
        
        # Création d'un modèle LightGBM pour la survie
        lgbm_model = lgb.LGBMRegressor(
            objective='cox',
            max_depth=lgbm_params['max_depth'],
            learning_rate=lgbm_params['learning_rate'],
            verbose=lgbm_params['verbose']
        )
        
        # Préparation des données
        status = np.array([t[0] for t in y_train])
        time = np.array([t[1] for t in y_train])
        
        # Entraînement du modèle
        lgbm_model.fit(
            X_train_features,
            np.array(list(zip(status, time)))
        )
        
        models["lgbm"] = lgbm_model
    
    # 4. Modèle Random Survival Forest
    if GLOBAL["rsf"]["run"]:
        print("Entraînement du modèle Random Survival Forest...")
        # Paramètres RSF
        rsf_params = PARAMS["rsf"]
        
        # Création du modèle RSF
        rsf_model = RandomSurvivalForest(
            n_estimators=rsf_params['n_estimators'],
            max_depth=rsf_params['max_depth'],
            min_samples_split=rsf_params['min_samples_split'],
            min_samples_leaf=rsf_params['min_samples_leaf'],
            max_features=rsf_params['max_features'],
            n_jobs=rsf_params['n_jobs'],
            random_state=42
        )
        
        # Entraînement du modèle
        rsf_model.fit(X_train_scaled, y_train)
        
        models["rsf"] = rsf_model
    
    # Fonction pour combiner les prédictions
    def ensemble_predict(X):
        # Préparation des données
        X_features = X[best_features].copy()
        X_scaled = pd.DataFrame(
            scaler.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        predictions = {}
        
        # Prédictions de chaque modèle
        if "cox" in models:
            predictions["cox"] = models["cox"].predict(X_scaled)
        
        if "xgb" in models:
            predictions["xgb"] = models["xgb"].predict(X_features)
        
        if "lgbm" in models:
            predictions["lgbm"] = models["lgbm"].predict(X_features)
        
        if "rsf" in models:
            predictions["rsf"] = models["rsf"].predict(X_scaled)
        
        # Normalisation des prédictions
        for model_name in predictions:
            pred = predictions[model_name]
            predictions[model_name + "_norm"] = (pred - pred.mean()) / (pred.std() + 1e-8)
        
        # Combinaison pondérée (poids optimisés)
        ensemble_pred = 0.0
        weights = {
            "cox_norm": 0.2,
            "xgb_norm": 0.3,
            "lgbm_norm": 0.2,
            "rsf_norm": 0.3
        }
        
        for model_name, weight in weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        # Normalisation finale pour avoir des scores entre -1.5 et 1.5
        min_val = np.min(ensemble_pred)
        max_val = np.max(ensemble_pred)
        normalized_pred = -1.5 + 3.0 * (ensemble_pred - min_val) / (max_val - min_val)
        
        return normalized_pred
    
    return ensemble_predict

def evaluate_feature_sets(X_train, y_train, feature_sets, min_features=30):
    """Évalue différents ensembles de features par validation croisée"""
    print("Évaluation des ensembles de features...")
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, features in feature_sets.items():
        print(f"\nÉvaluation de l'ensemble '{name}' ({len(features)} features)")
        
        # Vérification des features
        valid_features = [f for f in features if f in X_train.columns]
        if len(valid_features) < len(features):
            print(f"  Attention: {len(features) - len(valid_features)} features manquantes")
        
        if len(valid_features) == 0:
            print("  Aucune feature valide, ensemble ignoré")
            continue
        
        # Scores de validation croisée
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            # Séparation train/validation
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Création et entraînement du modèle
            model_func = create_ensemble_model(X_cv_train, y_cv_train, valid_features)
            
            # Prédiction et évaluation
            val_preds = model_func(X_cv_val)
            score = concordance_index_ipcw(y_cv_train, y_cv_val, val_preds, tau=7)[0]
            cv_scores.append(score)
        
        # Score moyen
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"  Score CV: {mean_score:.4f} ± {std_score:.4f}")
        
        results[name] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'features': valid_features
        }
    
    # Meilleur ensemble
    best_set_name = max(results.items(), key=lambda x: x[1]['mean_score'])[0]
    best_features = results[best_set_name]['features']
    print(f"\nMeilleur ensemble: '{best_set_name}' avec score {results[best_set_name]['mean_score']:.4f}")
    
    # S'assurer qu'on a au moins min_features
    if len(best_features) < min_features:
        print(f"Le nombre de features ({len(best_features)}) est inférieur au minimum requis ({min_features})")
        
        # Ajouter des features supplémentaires si nécessaire
        all_features = set(X_train.columns)
        remaining_features = list(all_features - set(best_features))
        
        # Filtrer les features numériques uniquement
        numeric_features = []
        for feature in remaining_features:
            if pd.api.types.is_numeric_dtype(X_train[feature]):
                numeric_features.append(feature)
            else:
                print(f"Feature non numérique ignorée: {feature}")
        
        # Utiliser une méthode alternative pour évaluer l'importance des features
        print("Évaluation de l'importance des features restantes...")
        
        # Utiliser un modèle Cox pour évaluer l'importance des features
        importances = {}
        
        # Évaluer chaque feature individuellement
        for feature in numeric_features:
            # Créer un petit ensemble de features (les meilleures + cette feature)
            test_features = best_features + [feature]
            
            # Validation croisée pour évaluer cette feature
            feature_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                # Modèle Cox
                cox = CoxnetSurvivalAnalysis(l1_ratio=0.7, alpha_min_ratio=0.01)
                cox.fit(X_train_fold[test_features], y_train_fold)
                
                # Prédictions
                pred = cox.predict(X_val_fold[test_features])
                
                # Évaluation
                if hasattr(y_val_fold, 'dtype') and 'status' in y_val_fold.dtype.names:
                    status = y_val_fold['status']
                    time = y_val_fold['years'] if 'years' in y_val_fold.dtype.names else y_val_fold['time']
                else:
                    status = np.array([t[0] for t in y_val_fold])
                    time = np.array([t[1] for t in y_val_fold])
                
                c_index = concordance_index_ipcw(y_train_fold, y_val_fold, -pred, tau=7)[0]
                feature_scores.append(c_index)
            
            # Importance = amélioration moyenne du C-index
            importances[feature] = np.mean(feature_scores) - results[best_set_name]['mean_score']
        
        # Trier les features par importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Ajouter les features les plus importantes jusqu'à atteindre min_features
        features_to_add = min(min_features - len(best_features), len(sorted_features))
        additional_features = [f for f, _ in sorted_features[:features_to_add]]
        best_features.extend(additional_features)
        
        print(f"Features ajoutées: {additional_features}")
        print(f"Nombre total de features: {len(best_features)}")
    
    return results, best_features

def optimize_weights(X_train, y_train, best_features):
    """Optimise les poids de l'ensemble par validation croisée"""
    print("\nOptimisation des poids de l'ensemble...")
    
    # Préparation des données
    X_train_features = X_train[best_features].copy()
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_features),
        columns=X_train_features.columns,
        index=X_train_features.index
    )
    
    # Modèles individuels
    models = {}
    predictions = {}
    
    # 1. Modèle Cox
    if GLOBAL["cox"]["run"]:
        cox_model = CoxnetSurvivalAnalysis(l1_ratio=0.7, alpha_min_ratio=0.01, max_iter=1000)
        cox_model.fit(X_train_scaled, y_train)
        models["cox"] = cox_model
        predictions["cox"] = cox_model.predict(X_train_scaled)
    
    # 2. Modèle XGBoost
    if GLOBAL["xgb"]["run"]:
        # Paramètres XGBoost
        xgb_params = PARAMS["xgb"]
        
        # Création du modèle
        xgb_model = xgb.XGBRegressor(
            objective='survival:cox',
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            n_estimators=xgb_params['n_estimators'],
            subsample=xgb_params['subsample'],
            random_state=xgb_params['random_state']
        )
        
        # Préparation des données
        status = np.array([t[0] for t in y_train])
        time = np.array([t[1] for t in y_train])
        
        # Entraînement
        xgb_model.fit(
            X_train_features,
            np.array(list(zip(status, time))),
            eval_metric='cox-nloglik'
        )
        
        models["xgb"] = xgb_model
        predictions["xgb"] = xgb_model.predict(X_train_features)
    
    # 3. Modèle LightGBM
    if GLOBAL["lgbm"]["run"]:
        # Paramètres LightGBM
        lgbm_params = PARAMS["lgbm"]
        
        # Création du modèle
        lgbm_model = lgb.LGBMRegressor(
            objective='cox',
            max_depth=lgbm_params['max_depth'],
            learning_rate=lgbm_params['learning_rate'],
            verbose=lgbm_params['verbose']
        )
        
        # Préparation des données
        status = np.array([t[0] for t in y_train])
        time = np.array([t[1] for t in y_train])
        
        # Entraînement
        lgbm_model.fit(
            X_train_features,
            np.array(list(zip(status, time)))
        )
        
        models["lgbm"] = lgbm_model
        predictions["lgbm"] = lgbm_model.predict(X_train_features)
    
    # 4. Modèle Random Survival Forest
    if GLOBAL["rsf"]["run"]:
        # Paramètres RSF
        rsf_params = PARAMS["rsf"]
        
        # Création du modèle
        rsf_model = RandomSurvivalForest(
            n_estimators=rsf_params['n_estimators'],
            max_depth=rsf_params['max_depth'],
            min_samples_split=rsf_params['min_samples_split'],
            min_samples_leaf=rsf_params['min_samples_leaf'],
            max_features=rsf_params['max_features'],
            n_jobs=rsf_params['n_jobs'],
            random_state=42
        )
        
        # Entraînement
        rsf_model.fit(X_train_scaled, y_train)
        
        models["rsf"] = rsf_model
        predictions["rsf"] = rsf_model.predict(X_train_scaled)
    
    # Normalisation des prédictions
    for model_name in predictions:
        pred = predictions[model_name]
        predictions[model_name + "_norm"] = (pred - pred.mean()) / (pred.std() + 1e-8)
    
    # Grille de poids à tester
    weight_grid = []
    step = 0.1
    
    # Générer toutes les combinaisons de poids possibles qui somment à 1
    model_names = [m + "_norm" for m in models.keys()]
    num_models = len(model_names)
    
    if num_models == 1:
        # Si un seul modèle, pas besoin d'optimiser
        best_weights = {model_names[0]: 1.0}
        best_score = concordance_index_ipcw(y_train, y_train, predictions[model_names[0]], tau=7)[0]
    else:
        # Générer une grille de poids pour plusieurs modèles
        if num_models == 2:
            for w1 in np.arange(0, 1.01, step):
                w2 = 1.0 - w1
                weight_grid.append({model_names[0]: w1, model_names[1]: w2})
        elif num_models == 3:
            for w1 in np.arange(0, 1.01, step):
                for w2 in np.arange(0, 1.01-w1, step):
                    w3 = 1.0 - w1 - w2
                    weight_grid.append({model_names[0]: w1, model_names[1]: w2, model_names[2]: w3})
        elif num_models == 4:
            for w1 in np.arange(0, 1.01, 0.2):  # Pas plus grand pour limiter les combinaisons
                for w2 in np.arange(0, 1.01-w1, 0.2):
                    for w3 in np.arange(0, 1.01-w1-w2, 0.2):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 >= 0:
                            weight_grid.append({
                                model_names[0]: w1, 
                                model_names[1]: w2, 
                                model_names[2]: w3, 
                                model_names[3]: w4
                            })
        
        # Recherche des meilleurs poids
        best_score = 0
        best_weights = {name: 1.0/num_models for name in model_names}  # Poids égaux par défaut
        
        print(f"Test de {len(weight_grid)} combinaisons de poids...")
        
        for weights in weight_grid:
            # Combinaison pondérée
            ensemble_preds = sum(weights[name] * predictions[name] for name in weights)
            
            # Évaluation
            score = concordance_index_ipcw(y_train, y_train, ensemble_preds, tau=7)[0]
            
            if score > best_score:
                best_score = score
                best_weights = weights
    
    print(f"Meilleurs poids trouvés: {best_weights} avec score {best_score:.4f}")
    
    return best_weights, best_score