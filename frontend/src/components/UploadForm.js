import React from 'react';
import styles from '../styles/UploadForm.module.css';

function UploadForm({
    onFileSelect,
    onTargetColumnChange,
    onSubmit,
    file,
    targetColumn
}) {
    return (
        <section className={styles.uploadSection}>
            <form onSubmit={onSubmit}>
                <div className={styles.formGroup}>
                    <label htmlFor="file">Загрузите файл данных (CSV или XLSX):</label>
                    <div className={styles.fileInputWrapper}>
                        <input
                            type="file"
                            id="file"
                            className={styles.fileInput}
                            accept=".csv,.xlsx"
                            onChange={(e) => onFileSelect(e.target.files[0])}
                        />
                        <div
                            className={styles.fileInputDisplay}
                        >
                            {file ? (
                                <div>
                                    <p><strong>Выбран файл:</strong></p>
                                    <p>{file.name}</p>
                                    <p>Размер: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                </div>
                            ) : (
                                <div>
                                    <p><strong>Нажмите для выбора файла</strong></p>
                                    <p>Поддерживаемые форматы: CSV, XLSX</p>
                                    <p>Максимальный размер: 50мб</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className={styles.formGroup}>
                    <label htmlFor="target_column">Целевая переменная:</label>
                    <input
                        type="text"
                        id="target_column"
                        className={styles.textInput}
                        value={targetColumn}
                        onChange={(e) => onTargetColumnChange(e.target.value)}
                        required
                    />
                </div>

                <button type="submit" className={styles.submitBtn}>
                    Запустить анализ
                </button>
            </form>
        </section>
    );
}

export default UploadForm; 