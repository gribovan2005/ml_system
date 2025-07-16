import React from 'react';
import styles from '../styles/Header.module.css';

function Header() {
    return (
        <header className={styles.header}>
            <h1>ML Analytics System</h1>
            <p>Система анализа данных с машинным обучением</p>
        </header>
    );
}

export default Header; 