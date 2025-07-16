import sqlite3
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class TaskDatabase:
    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        result TEXT,
                        error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                logger.info(f"База данных инициализирована: {self.db_path}")
        except Exception as e:
            logger.error(f"ошибка инициализированиябазы данных: {e}")
            raise
    
    def create_task(self, task_id: str, status: str = "processing") -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO tasks (task_id, status) VALUES (?, ?)',
                    (task_id, status)
                )
                conn.commit()
                logger.info(f"Создана задача {task_id} со статусом {status}")
                return True
        except Exception as e:
            logger.error(f"Ошибка при создании задачи {task_id}: {e}")
            return False
    
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                result_json = json.dumps(result, ensure_ascii=False) if result else None
                
                cursor.execute('''
                    UPDATE tasks 
                    SET status = ?, result = ?, error = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE task_id = ?
                ''', (status, result_json, error, task_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Обновлена задача {task_id}: статус={status}, есть_результат={result is not None}")
                    return True
                else:
                    logger.warning(f"Задача {task_id} не найдена для обновления")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка при обновлении задачи {task_id}: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT task_id, status, result, error, created_at, updated_at FROM tasks WHERE task_id = ?',
                    (task_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    result_data = None
                    if row[2]: 
                        try:
                            result_data = json.loads(row[2])
                        except json.JSONDecodeError as e:
                            logger.error(f"Ошибка парсинга JSON результата для задачи {task_id}: {e}")
                    
                    task = {
                        'task_id': row[0],
                        'status': row[1],
                        'result': result_data,
                        'error': row[3],
                        'created_at': row[4],
                        'updated_at': row[5]
                    }
                    logger.info(f"Найдена задача {task_id} в БД, статус: {task['status']}")
                    return task
                else:
                    logger.warning(f"Задача {task_id} не найдена в БД")
                    return None
                    
        except Exception as e:
            logger.error(f"Ошибка при получении задачи {task_id}: {e}")
            return None
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT task_id, status, result, error, created_at, updated_at FROM tasks ORDER BY created_at DESC'
                )
                rows = cursor.fetchall()
                
                tasks = {}
                for row in rows:
                    result_data = None
                    if row[2]: 
                        try:
                            result_data = json.loads(row[2])
                        except json.JSONDecodeError:
                            pass
                    
                    tasks[row[0]] = {
                        'task_id': row[0],
                        'status': row[1],
                        'result': result_data,
                        'error': row[3],
                        'created_at': row[4],
                        'updated_at': row[5]
                    }
                
                logger.info(f"Получено {len(tasks)} задач из БД")
                return tasks
                
        except Exception as e:
            logger.error(f"Ошибка получения всех задач: {e}")
            return {}
    
    def delete_task(self, task_id: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Удалена задача {task_id}")
                    return True
                else:
                    logger.warning(f"Задача {task_id} не найдена для удаления")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка при удалении задачи {task_id}: {e}")
            return False

db = TaskDatabase() 