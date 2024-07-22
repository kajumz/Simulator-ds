import json
import re
import sqlite3

import fire
import numpy as np
import pandas as pd


def valid_email(email: str) -> bool:
    """Checks if an email address is valid.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email address is valid, False otherwise.
    """
    valid_email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.fullmatch(valid_email_regex, email))


def filter_db(db_path: str) -> str:
    """Filters a SQLite database and returns the user IDs with invalid email addresses.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        str: json string with the user IDs with invalid email addresses.
        e.g: {"user_ids": ["1", "2", "3", "4", "5", ...]}
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM users", conn)
    df["email_valid"] = np.vectorize(valid_email)(df["email"])
    invalid_emails = df[~df["email_valid"]]["user_id"]
    json_str = json.dumps({"user_ids": invalid_emails.tolist()})
    return json_str


if __name__ == "__main__":
    fire.Fire(filter_db)
