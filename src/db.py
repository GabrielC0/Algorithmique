"""Connexion à la base de données MySQL et accès à la table `tweets`."""

import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "socialmetrics"),
    "password": os.getenv("DB_PASSWORD", "socialmetrics"),
    "database": os.getenv("DB_NAME", "socialmetrics"),
}


def get_connection(with_database: bool = True):
    """Ouvre une connexion MySQL à partir des variables d'environnement.

    with_database=False permet de se connecter avant que la base n'existe
    (utile pour l'initialisation).
    """
    config = dict(DB_CONFIG)
    if not with_database:
        config.pop("database")
    return mysql.connector.connect(**config)


def check_connection() -> bool:
    """Vérifie que la base de données est joignable."""
    try:
        conn = get_connection()
        conn.ping(reconnect=False)
        conn.close()
        return True
    except mysql.connector.Error:
        return False


def fetch_all_tweets():
    """Retourne tous les tweets annotés : liste de tuples (text, positive, negative)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT text, positive, negative FROM tweets ORDER BY id")
        return cursor.fetchall()
    finally:
        conn.close()


def count_tweets() -> int:
    """Retourne le nombre de tweets annotés en base."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tweets")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def insert_tweets(rows) -> int:
    """Insère une liste de tweets annotés [(text, positive, negative), ...]."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            rows,
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
