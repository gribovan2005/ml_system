import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import styles from './styles/App.module.css'; 
import Header from './components/Header';
import MessageDisplay from './components/MessageDisplay';
import UploadForm from './components/UploadForm';
import LoadingSpinner from './components/LoadingSpinner';
import ResultsDisplay from './components/ResultsDisplay';

function App() {
    const [file, setFile] = useState(null);
    const [targetColumn, setTargetColumn] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [taskId, setTaskId] = useState(null);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);

    const resetAppState = useCallback(() => {
        setResults(null);
        setFile(null);
        setTargetColumn('');
        setError(null);
        setSuccess(null);
    }, []);

    const checkResults = useCallback(async (id) => {
        try {
            console.log(`from react: Проверяем результаты для задачи ${id}`);
            const response = await fetch(`/results/${id}`);
            console.log(`from react: Ответ сервера:`, response.status, response.statusText);
            
            if (!response.ok) {
                if (response.status === 404) {
                    console.log(`from react: Задача ${id} не найдена (404)`);
                    return null;
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`from react: Получены данные:`, data);
            
            if (data.status === 'completed') {
                console.log(`from react: Задача ${id} завершена успешно`);
                setResults(data.result);
                setIsLoading(false);
                setTaskId(null);
                setSuccess('анализ завершен успешно');
                return data;
            } else if (data.status === 'failed') {
                console.log(`from react: Задача ${id} завершена с ошибкой:`, data.error);
                setError(`Ошибка обработки: ${data.error}`);
                setIsLoading(false);
                setTaskId(null);
                return data;
            } else if (data.status === 'processing') {
                console.log(`from react: Задача ${id} все еще обрабатывается`);
                return null; 
            }
        } catch (err) {
            console.error(`from react: Ошибка при проверке результатов:`, err);
            setError(`Ошибка получения результатов: ${err.message}`);
            setIsLoading(false);
            setTaskId(null);
        }
        return null;
    }, []);

    useEffect(() => {
        let interval;
        if (taskId && isLoading) {
            console.log(`from react: Запускаем поллинг для задачи ${taskId}`);
            interval = setInterval(async () => {
                const result = await checkResults(taskId);
                if (result && (result.status === 'completed' || result.status === 'failed')) {
                    clearInterval(interval);
                }
            }, 2000); 
        }
        
        return () => {
            if (interval) {
                console.log(`from react: Останавливаем поллинг для задачи ${taskId}`);
                clearInterval(interval);
            }
        };
    }, [taskId, isLoading, checkResults]);

    const handleFileChange = (selectedFile) => {
        console.log(`from react: Выбран файл:`, selectedFile?.name);
        setFile(selectedFile);
        setError(null);
        setSuccess(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!file) {
            setError('нужно выбрать файл');
            return;
        }
        
        if (!targetColumn.trim()) {
            setError('нужно укажите целевую переменную');
            return;
        }

        console.log(`from react: запуск анализа файла ${file.name}, целевая переменная: ${targetColumn}`);
        
        setIsLoading(true);
        setError(null);
        setSuccess(null);
        setResults(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_column', targetColumn.trim());

        try {
            const response = await fetch('/upload-and-train/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log(`from react: Файл загружен, task_id: ${data.task_id}`);
            
            setTaskId(data.task_id);
            setSuccess(data.message);
            
        } catch (err) {
            console.error(`from react: Ошибка загрузки:`, err);
            setError(`Ошибка загрузки: ${err.message}`);
            setIsLoading(false);
        }
    };

    const downloadModel = async () => {
        if (!results?.task_id) return;
        
        try {
            const response = await fetch(`/download-model/${results.task_id}`);
            if (!response.ok) {
                throw new Error('Ошибка скачвания модели');
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `model_${results.task_id}.joblib`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setError(`Ошибка скачивания: ${err.message}`);
        }
    };

    return (
        <div className={styles.container}>
            <Header />

            <MessageDisplay message={error} type="error" />
            <MessageDisplay message={success} type="success" />

            {!isLoading && !results && (
                <UploadForm 
                    onFileSelect={handleFileChange}
                    onTargetColumnChange={setTargetColumn}
                    onSubmit={handleSubmit}
                    file={file}
                    targetColumn={targetColumn}
                />
            )}

            {isLoading && (
                <LoadingSpinner taskId={taskId} />
            )}

            {results && (
                <ResultsDisplay 
                    results={results}
                    downloadModel={downloadModel}
                    resetApp={resetAppState}
                />
            )}
        </div>
    );
}

export default App;
