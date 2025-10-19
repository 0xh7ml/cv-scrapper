"""
Configuration settings for CV Scraper
"""

from pathlib import Path
from typing import Set

# File processing settings
SUPPORTED_EXTENSIONS: Set[str] = {'.pdf', '.docx'}
DEFAULT_INPUT_DIR: Path = Path('input')
DEFAULT_OUTPUT_FILE: Path = Path('extracted_cvs.csv')
DEFAULT_CONCURRENCY: int = 10

# API settings
GEMINI_MODEL_NAME: str = 'gemini-2.0-flash-exp'
GEMINI_TEMPERATURE: float = 0.1
GEMINI_MAX_OUTPUT_TOKENS: int = 500
GEMINI_TOP_P: float = 0.8
GEMINI_TOP_K: int = 40

# Processing settings
FILE_TIMEOUT_SECONDS: int = 60
MAX_FIELD_LENGTH: int = 200

# Logging settings
LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

# CSV field names
CSV_FIELDNAMES = ['id', 'name', 'phone', 'email', 'education', 'skills', 'experience']

# Gemini extraction prompt template
EXTRACTION_PROMPT_TEMPLATE = """You are an expert CV/Resume data extraction system. Extract the following information from the CV text and return it as a JSON object.

Required fields:
- name: Full name of the person
- phone: Phone number (normalize format, include country code if present)
- email: Email address
- education: Educational background, degrees, institutions (summarize key points)
- skills: Technical skills, competencies, programming languages (comma-separated list)
- experience: Work experience, job titles, companies, duration (summarize key roles)

Rules:
1. If a field is not found, return empty string ""
2. For education: Include degree, institution, year if available
3. For skills: Extract technical skills, software, programming languages
4. For experience: Include job titles, companies, and brief descriptions
5. Keep each field concise but informative (max {max_length} chars per field)
6. Return only valid JSON, no additional text

CV Text:
{text}"""