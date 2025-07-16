import React from 'react';
import styles from '../styles/MessageDisplay.module.css';

function MessageDisplay({ message, type }) {
    if (!message) return null;

    return (
        <div className={styles[type]}>
            {message}
        </div>
    );
}

export default MessageDisplay; 