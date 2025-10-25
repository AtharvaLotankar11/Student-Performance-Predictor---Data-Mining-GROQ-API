#!/usr/bin/env python3
"""
Generate a secure secret key for Flask application
"""
import secrets

def generate_secret_key():
    """Generate a secure random secret key"""
    return secrets.token_hex(32)

if __name__ == "__main__":
    secret_key = generate_secret_key()
    print("Generated Secret Key:")
    print(secret_key)
    print("\nAdd this to your .env file:")
    print(f"SECRET_KEY={secret_key}")