import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import logging
from typing import Dict, Any


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/models', exist_ok=True)


def validate_data(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    errors = []
    warnings = []
    
    if df.shape[0] < 10:
        errors.append("Слишком мало строк в датасете (минимум 10)")
    
    if target_column not in df.columns:
        errors.append(f"Целевая колонка '{target_column}' не найдена в файле")
    else:
        
        missing_target = df[target_column].isnull().sum()
        if missing_target > df.shape[0] * 0.5:
            errors.append(f"слишком много пропущенных значений в целевой переменной: {missing_target}/{df.shape[0]}")
        elif missing_target > 0:
            warnings.append(f"Есть пропущенные значения в целевой переменной: {missing_target}")
    
    if df.shape[1] < 2:
        errors.append(" Мало признаков в датасете")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Есть дубликаты строк: {duplicates}")
    
    return {"errors": errors, "warnings": warnings}


def process_and_train(file_path: str, target_column: str, task_id: str) -> Dict[str, Any]:
    logger.info(f"Начало обработки задачи {task_id}")
    
    try:
        logger.info("Загрузка датасета")
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            return {"error": f"Ошибка загрузки датасета: {str(e)}"}

        validation_result = validate_data(df, target_column)
        if validation_result["errors"]:
            logger.error(f"Ошибки валидирования: {validation_result['errors']}")
            return {"error": "; ".join(validation_result["errors"])}
        
        if validation_result["warnings"]:
            logger.warning(f"Предупреждения: {validation_result['warnings']}")
        
        if df.duplicated().sum() > 0:
            df = df.drop_duplicates()
            logger.info("Удалены дубликаты строк")
            
        data_summary = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "features": list(df.columns),
            "missing_values": {col: int(count) for col, count in df.isnull().sum().items()}
        }
        logger.info(f"Данные были загружены {data_summary['rows']} строк, {data_summary['columns']} колонок")

        initial_rows = df.shape[0]
        df = df.dropna(subset=[target_column])
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            logger.info(f"Удалено {dropped_rows} строк с пропущенными значениями в целевой переменной")

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        unique_targets = y.nunique()
        is_classification = (not pd.api.types.is_numeric_dtype(y)) or unique_targets <= 20
        task_type = "classification" if is_classification else "regression"

        logger.info(f"Определён тип задачи: {task_type} (уникальных значений целевой переменной: {unique_targets})")

        logger.info("Начало предобработки данных")
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        
        logger.info(f"Числовые признаки: {len(numeric_features)}, Категориальные: {len(categorical_features)}")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        logger.info("Разделение данных на обучающую и тестовые выборки и начало обучения")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1) if is_classification else RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            "CatBoost": CatBoostClassifier(iterations=200, random_state=42, verbose=0) if is_classification else CatBoostRegressor(iterations=200, random_state=42, verbose=0)
        }
        
        best_model_name = None
        best_model = None
        best_r2 = -np.inf
        results = {}

        for name, model in models.items():
            logger.info(f"Обучение модели {name}")
            try:
                pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', model)])
                
                pipeline.fit(X_train, y_train)
                
                y_pred = pipeline.predict(X_test)
                
                if is_classification:
                    acc = float(accuracy_score(y_test, y_pred))
                    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                    results[name] = {"Accuracy": acc, "F1": f1, "pipeline": pipeline, "y_test": y_test, "y_pred": y_pred}
                    logger.info(f"Модель {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

                    current_score = f1
                else:
                    mae = float(mean_absolute_error(y_test, y_pred))
                    r2 = float(r2_score(y_test, y_pred))
                    results[name] = {"MAE": mae, "R2": r2, "pipeline": pipeline, "y_test": y_test, "y_pred": y_pred}
                    logger.info(f"Модель {name}: MAE={mae:.4f}, R2={r2:.4f}")

                    current_score = r2  

                if current_score > best_r2:
                    best_r2 = current_score
                    best_model_name = name
                    best_model = pipeline
                    
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {name}: {e}")
                continue

        if best_model is None:
            return {"error": "Не удалось обучить ни одну модель"}

        logger.info(f"Лучшая модель: {best_model_name} (R2={best_r2:.4f})")

        model_filename = f"results/models/model_{task_id}.joblib"
        joblib.dump(best_model, model_filename)
        logger.info(f"Модель сохранена: {model_filename}")

        plot_importance_path = None
        try:
            logger.info("Создание графика важности признаков")
            if categorical_features:
                ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                all_feature_names = numeric_features + list(ohe_feature_names)
            else:
                all_feature_names = numeric_features
            
            if best_model_name == 'RandomForest':
                importances = best_model.named_steps['regressor'].feature_importances_
            else: 
                importances = best_model.named_steps['regressor'].get_feature_importance()
            
            feature_importance = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
            plt.title('Важность признаков', fontsize=16, fontweight='bold')
            plt.xlabel('Важность', fontsize=12)
            plt.ylabel('Признаки', fontsize=12)
            plt.tight_layout()
            plot_importance_path = f"results/plots/importance_{task_id}.png"
            plt.savefig(plot_importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка при создании графика важности признаков: {e}")

        plot_confusion_path = None
        try:
            if is_classification:
                logger.info("Создание матрицы ошибок")
                cm = confusion_matrix(results[best_model_name]['y_test'], results[best_model_name]['y_pred'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel('Предсказано')
                plt.ylabel('Факт')
                plt.title('Матрица ошибок')
                plot_confusion_path = f"results/plots/confusion_{task_id}.png"
                plt.savefig(plot_confusion_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                logger.info("Создание графика предсказаний")
                plt.figure(figsize=(8, 8))
                sns.scatterplot(x=results[best_model_name]['y_test'], y=results[best_model_name]['y_pred'], alpha=0.6)
                plt.xlabel('Реальные значения', fontsize=12)
                plt.ylabel('Предсказанные значения', fontsize=12)
                plt.title('Предсказания vs Реальность', fontsize=16, fontweight='bold')
                min_val = min(y.min(), results[best_model_name]['y_pred'].min())
                max_val = max(y.max(), results[best_model_name]['y_pred'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='Идеальные предсказания')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_predictions_path = f"results/plots/predictions_{task_id}.png"
                plt.savefig(plot_predictions_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.error(f"Ошибка при создании графика предсказаний: {e}")
            plot_predictions_path = None

        final_result = {
            "task_id": task_id,
            "best_model": best_model_name,
            "task_type": task_type,
            "metrics": ({
                "Accuracy": results[best_model_name]["Accuracy"],
                "F1": results[best_model_name]["F1"]
            } if task_type == "classification" else {
                "MAE": results[best_model_name]["MAE"],
                "R2": results[best_model_name]["R2"]
            }),
            "data_summary": data_summary,
            "plots": {
                "feature_importance": plot_importance_path,
                "confusion_matrix": plot_confusion_path if task_type == "classification" else None,
                "predictions_vs_actual": plot_predictions_path if task_type == "regression" else None
            },
            "model_path": model_filename
        }
        
        logger.info(f"Задача {task_id} завершена успешно")
        return final_result

    except Exception as e:
        logger.error(f"Неожиданная ошибка в задаче {task_id}: {e}")
        return {"error": f"Неожиданная ошибка: {str(e)}"}