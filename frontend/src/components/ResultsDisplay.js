import React from 'react';
import styles from '../styles/ResultsDisplay.module.css';

function ResultsDisplay({
    results,
    downloadModel,
    resetApp
}) {
    if (!results) return null;

    const isClassification = results.task_type === 'classification';

    return (
        <section className={styles.resultsSection}>
            <h2>Результаты анализа</h2>
            
            <div className={styles.resultsGrid}>
                <div className={styles.resultCard}>
                    <h3>Лучшая модель</h3>
                    <div className={styles.metric}>
                        <span>Модель:</span>
                        <strong>{results.best_model}</strong>
                    </div>
                    {isClassification ? (
                        <>
                            <div className={styles.metric}>
                                <span>Accuracy:</span>
                                <strong>{(results.metrics.Accuracy * 100).toFixed(2)}%</strong>
                            </div>
                            <div className={styles.metric}>
                                <span>F1:</span>
                                <strong>{results.metrics.F1.toFixed(4)}</strong>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className={styles.metric}>
                                <span>MAE:</span>
                                <strong>{results.metrics.MAE.toFixed(4)}</strong>
                            </div>
                            <div className={styles.metric}>
                                <span>R²:</span>
                                <strong>{results.metrics.R2.toFixed(4)}</strong>
                            </div>
                        </>
                    )}
                </div>

                <div className={styles.resultCard}>
                    <h3>Данные</h3>
                    <div className={styles.metric}>
                        <span>Строк:</span>
                        <strong>{results.data_summary.rows}</strong>
                    </div>
                    <div className={styles.metric}>
                        <span>Колонок:</span>
                        <strong>{results.data_summary.columns}</strong>
                    </div>
                    <div className={styles.metric}>
                        <span>Признаков:</span>
                        <strong>{results.data_summary.features.length}</strong>
                    </div>
                </div>
            </div>

            {results.plots && (
                <div className={styles.resultsGrid}>
                    <div className={styles.plotContainer}>
                        <h3>Важность признаков</h3>
                        <img src={results.plots.feature_importance} alt="Feature Importance" />
                    </div>
                    {results.task_type === 'regression' && results.plots.predictions_vs_actual && (
                        <div className={styles.plotContainer}>
                            <h3>Предсказания vs Реальные значения</h3>
                            <img src={results.plots.predictions_vs_actual} alt="Predictions vs Actual" />
                        </div>
                    )}
                    {results.task_type === 'classification' && results.plots.confusion_matrix && (
                        <div className={styles.plotContainer}>
                            <h3>Матрица ошибок</h3>
                            <img src={results.plots.confusion_matrix} alt="Confusion Matrix" />
                        </div>
                    )}
                </div>
            )}

            <div style={{textAlign: 'center'}}>
                <button onClick={downloadModel} className={styles.downloadBtn}>
                    Скачать обкченную модель
                </button>
                <button 
                    onClick={resetApp} 
                    className={styles.newAnalysisBtn} 
                >
                    Новый анализ
                </button>
            </div>
        </section>
    );
}

export default ResultsDisplay; 