-- Schéma de la base de données SocialMetrics AI
-- Table `tweets` : dataset annoté servant à l'entraînement du modèle.

CREATE DATABASE IF NOT EXISTS socialmetrics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE socialmetrics;

CREATE TABLE IF NOT EXISTS tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    positive TINYINT(1) NOT NULL DEFAULT 0,
    negative TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
