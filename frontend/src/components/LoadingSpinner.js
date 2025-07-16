import React from 'react';
import styles from '../styles/LoadingSpinner.module.css';

function LoadingSpinner({ taskId }) {
    return (
        <section className={styles.loading}>
            <div className={styles.spinner}></div>
            <h2>Обработка данных</h2>
            <p>Выполняется загрузка, предобработка и обучение моделей</p>
            <p>Task ID: <code>{taskId}</code></p>
            <p><em>Это может занять время</em></p>
        </section>
    );
}

export default LoadingSpinner; 