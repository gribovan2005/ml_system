from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import logging
from typing import Dict, Any
from .ml_service import process_and_train
from .models import TaskResponse, UploadResponse, ErrorResponse, TaskStatus
from .database import db
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML system", 
    description=")",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



os.makedirs("uploads", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

app.mount("/plots", StaticFiles(directory="results/plots"), name="plots")

def run_training(task_id: str, file_path: str, target_column: str):
    logger.info(f"Запуск фонового обучения для задачи {task_id}")
    
    db.create_task(task_id, "processing")
    
    try:
        result = process_and_train(file_path, target_column, task_id)
        
        if "error" in result:
            db.update_task_status(task_id, "failed", error=result["error"])
            logger.error(f"Задача {task_id} завершилась с ошибкой: {result['error']}")
        else:
            db.update_task_status(task_id, "completed", result=result)
            logger.info(f"Задача {task_id} завершена успешно - результат сохранен в БД")
            
    except Exception as e:
        error_msg = f"Неожиданная ошибка при выполнении задачи: {str(e)}"
        db.update_task_status(task_id, "failed", error=error_msg)
        logger.error(f"Критическая ошибка в задаче - {task_id}: {e}")
    
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Удален временный файл: {file_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")
        
        final_task = db.get_task(task_id)
        logger.info(f"Финальный статус задачи {task_id}: {final_task}")

@app.post("/upload-and-train/", response_model=UploadResponse)
async def upload_and_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV или XLSX"),
    target_column: str = Form(..., description="Название целевой переменной")
):
    logger.info(f"Получен файл: {file.filename}, целевая переменная: {target_column}")
    
    if not file.filename:
        raise HTTPException(
            status_code=400, 
            detail="Файл не был загружен"
        )
    
    if not (file.filename.lower().endswith('.csv') or file.filename.lower().endswith('.xlsx')):
        raise HTTPException(
            status_code=400, 
            detail="Неверный формат файла. Поддерживаются только CSV и XLSX файлы."
        )
    
    file.file.seek(0, 2)  
    file_size = file.file.tell()
    file.file.seek(0)  
    
    if file_size > 50 * 1024 * 1024:  
        raise HTTPException(
            status_code=400, 
            detail="Файл слишком большой. Максимальный размер: 50MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400, 
            detail="Файл пустой"
        )
    
    if not target_column.strip():
        raise HTTPException(
            status_code=400, 
            detail="Название целевой переменной не может быть пустым"
        )
    
    target_column = target_column.strip()

    task_id = str(uuid.uuid4())
    upload_folder = "uploads"
    
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    file_path = os.path.join(upload_folder, f"{task_id}_{safe_filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Файл сохранен: {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Ошибка при сохранении файла"
        )

    background_tasks.add_task(run_training, task_id, file_path, target_column)

    return UploadResponse(
        message="Файл успешно загружен, обучение модели запущено в фоновом режиме", 
        task_id=task_id
    )

@app.get("/results/{task_id}")
def get_results(task_id: str):
    logger.info(f" from SQLITE: Запрос результатов для задачи {task_id}")
    
    if not task_id:
        logger.error("Пустой task_id")
        raise HTTPException(
            status_code=400, 
            detail="ID задачи не может быть пустым"
        )
    
    task = db.get_task(task_id)
    if not task:
        logger.error(f"Задача {task_id} не найдена в БД")
        raise HTTPException(
            status_code=404, 
            detail="Задача не найдена"
        )
    
    logger.info(f"from SQLITE: Найдена задача {task_id} в БД, статус: {task['status']}")
    logger.info(f"from SQLITE: Есть результат: {task.get('result') is not None}")
    
    status = task["status"]
    if status == "processing":
        return {"status": "processing"}
    elif status == "failed":
        return {"status": "failed", "error": task.get("error", "Неизвестная ошибка")}
    elif status == "completed":
        result = task["result"].copy()
        if "plots" in result and result["plots"]:
            if result["plots"]["feature_importance"]:
                result["plots"]["feature_importance"] = result["plots"]["feature_importance"].replace("results/plots/", "/plots/").replace("\\", "/")
            if result["plots"]["predictions_vs_actual"]:
                result["plots"]["predictions_vs_actual"] = result["plots"]["predictions_vs_actual"].replace("results/plots/", "/plots/").replace("\\", "/")
            if result["plots"].get("confusion_matrix"):
                result["plots"]["confusion_matrix"] = result["plots"]["confusion_matrix"].replace("results/plots/", "/plots/").replace("\\", "/")
        
        logger.info(f"from SQLITE: Возвращаем результаты для задачи {task_id} из БД")
        return {"status": "completed", "result": result}
    
    else:
        logger.error(f"Неизвестный статус: {status}")
        raise HTTPException(
            status_code=500, 
            detail=f"Неизвестный статус: {status}"
        )
    

@app.get("/download-model/{task_id}")
def download_model(task_id: str):
    if not task_id:
        raise HTTPException(
            status_code=400, 
            detail="ID задачи не может быть пустым"
        )
    
    task = db.get_task(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(
            status_code=404, 
            detail="Модель не готова или не найдена"
        )
    
    model_path = task["result"]["model_path"]
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail="файл модели не найден"
        )
    
    logger.info(f"Скачивание модели для задачи {task_id}")
    return FileResponse(
        path=model_path, 
        filename=f"model_{task_id}.joblib", 
        media_type='application/octet-stream'
    )

@app.get("/health")
def health_check():
    all_tasks = db.get_all_tasks()
    return {"status": "healthy", "tasks_count": len(all_tasks)}

@app.get("/tasks")
def list_tasks():
    all_tasks = db.get_all_tasks()
    return {
        task_id: {"status": task["status"], "has_result": task["result"] is not None}
        for task_id, task in all_tasks.items()
    }

@app.post("/upload-external-model/")
async def upload_external_model(file: UploadFile = File(..., description="Файл модели .joblib или .pkl")):
    if not (file.filename.lower().endswith('.joblib') or file.filename.lower().endswith('.pkl')):
        raise HTTPException(
            status_code=400, 
            detail="Поддерживаются только файлы .joblib или .pkl"
        )

    model_id = str(uuid.uuid4())
    model_filename = f"external_{model_id}.joblib"
    model_path = os.path.join("results", "models", model_filename)
    try:
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Ошибка при сохранении внешней модели: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Не удалось сохранить файл модели"
        )

    db.create_task(model_id, "completed")
    db.update_task_status(model_id, "completed", result={"model_path": model_path})

    return {"message": "Модель успешно загружена", "model_id": model_id}


@app.post("/predict/{model_id}")
async def predict_with_model(model_id: str, file: UploadFile = File(..., description="CSV или XLSX файл с данными для предсказания")):
    task = db.get_task(model_id)
    if not task or task.get("result") is None:
        raise HTTPException(
            status_code=404, 
            detail="Модель не найдена"
        )

    model_path = task["result"].get("model_path") or task["result"].get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail="Файл модели отсутствует"
        )

    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        logger.error(f"Ошибка чтения входного файла для предсказаний: {e}")
        raise HTTPException(
            status_code=400, 
            detail="Не удалось прочитать входной файл"
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Не удалось загрузить модель"
        )

    try:
        preds = model.predict(df)
    except Exception as e:
        logger.error(f"Ошибка обобщения модели: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Ошибка при обобщении модели на данных"
        )

    preds_df = pd.DataFrame(preds, columns=["prediction"])
    predictions_path = os.path.join("results", "predictions", f"preds_{model_id}.csv")
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    preds_df.to_csv(predictions_path, index=False)

    return {
        "model_id": model_id,
        "predictions_path": predictions_path.replace("results/", "files/").replace("\\", "/"),
        "n_predictions": len(preds)
    }

