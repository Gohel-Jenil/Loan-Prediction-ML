"""
Vercel serverless entry point.
Imports the Flask app from backend/app.py.
Vercel's @vercel/python runtime detects the 'app' variable as a WSGI application.
"""
from backend.app import app
