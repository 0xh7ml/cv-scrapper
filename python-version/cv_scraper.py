#!/usr/bin/env python3
"""
CV Scraper - Extract data from PDF and DOCX CV files using Google Gemini AI
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import zipfile
import xml.etree.ElementTree as ET

# Import with graceful error handling for optional dependencies
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CVData:
    """Data class to hold extracted CV information"""
    id: str
    name: str = ""
    phone: str = ""
    email: str = ""
    education: str = ""
    skills: str = ""
    experience: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for CSV writing"""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if CV has at least some useful data"""
        return bool(self.name or self.phone or self.email)


@dataclass
class ProcessingResult:
    """Result of processing a single CV file"""
    cv_data: Optional[CVData]
    filename: str
    success: bool
    error: Optional[str] = None


class TextExtractor:
    """Handles text extraction from different file formats"""
    
    @staticmethod
    def extract_from_pdf(file_path: Path) -> str:
        """Extract text from PDF using multiple extraction methods"""
        extractors = [
            ("PyMuPDF", TextExtractor._extract_with_pymupdf),
            ("pdfplumber", TextExtractor._extract_with_pdfplumber),
            ("PyPDF2", TextExtractor._extract_with_pypdf2),
        ]
        
        for extractor_name, extractor_func in extractors:
            try:
                text = extractor_func(file_path)
                if text.strip():  # Only return if we got meaningful text
                    logger.debug(f"Successfully extracted text using {extractor_name}")
                    return text
            except Exception as e:
                logger.debug(f"{extractor_name} failed for {file_path}: {e}")
                continue
        
        raise Exception("All PDF extractors failed to extract meaningful text")
    
    @staticmethod
    def _extract_with_pymupdf(file_path: Path) -> str:
        """Extract text using PyMuPDF (fitz)"""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed")
        
        doc = fitz.open(file_path)
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        return '\n'.join(text_parts)
    
    @staticmethod
    def _extract_with_pdfplumber(file_path: Path) -> str:
        """Extract text using pdfplumber (good for complex layouts)"""
        if pdfplumber is None:
            raise ImportError("pdfplumber is not installed")
        
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        return '\n'.join(text_parts)
    
    @staticmethod
    def _extract_with_pypdf2(file_path: Path) -> str:
        """Extract text using PyPDF2 (basic extraction)"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is not installed")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            
            return '\n'.join(text_parts)
    
    @staticmethod
    def extract_from_docx(file_path: Path) -> str:
        """Extract text from DOCX using python-docx (fallback to zip method)"""
        try:
            # Try python-docx first if available
            if Document is not None:
                doc = Document(file_path)
                text_parts = []
                
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_parts.append(paragraph.text)
                
                return '\n'.join(text_parts)
            else:
                logger.info(f"python-docx not available, using zip method for {file_path}")
                raise ImportError("python-docx not installed")
            
        except Exception as e:
            logger.warning(f"python-docx failed for {file_path}, trying zip method: {e}")
            
            # Fallback to zip extraction method
            try:
                return TextExtractor._extract_docx_fallback(file_path)
            except Exception as e2:
                raise Exception(f"Both DOCX extractors failed: python-docx({e}), zip({e2})")
    
    @staticmethod
    def _extract_docx_fallback(file_path: Path) -> str:
        """Fallback DOCX extraction using zip and XML parsing"""
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            try:
                # Read the main document XML
                doc_xml = zip_file.read('word/document.xml')
                root = ET.fromstring(doc_xml)
                
                # Extract text from all text nodes
                text_parts = []
                for elem in root.iter():
                    if elem.tag.endswith('}t'):  # Text elements
                        if elem.text:
                            text_parts.append(elem.text)
                
                return ' '.join(text_parts)
                
            except KeyError:
                raise Exception("word/document.xml not found in DOCX file")


class GeminiCVExtractor:
    """Handles CV data extraction using Google Gemini AI"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client"""
        if genai is None:
            raise ImportError("google-generativeai is not installed. Install it with: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configure model settings
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent extraction
            max_output_tokens=500,
            top_p=0.8,
            top_k=40
        )
    
    def extract_cv_data(self, text: str, cv_id: str) -> CVData:
        """Extract CV data from text using Gemini AI"""
        prompt = self._create_extraction_prompt(text)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if not response.text:
                raise Exception("Empty response from Gemini")
            
            # Clean and parse JSON response
            json_text = self._clean_json_response(response.text)
            cv_dict = json.loads(json_text)
            
            return CVData(
                id=cv_id,
                name=cv_dict.get('name', ''),
                phone=cv_dict.get('phone', ''),
                email=cv_dict.get('email', ''),
                education=cv_dict.get('education', ''),
                skills=cv_dict.get('skills', ''),
                experience=cv_dict.get('experience', '')
            )
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Gemini response as JSON: {e}")
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    @staticmethod
    def _create_extraction_prompt(text: str) -> str:
        """Create the prompt for CV data extraction"""
        return f"""You are an expert CV/Resume data extraction system. Extract the following information from the CV text and return it as a JSON object.

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
5. Keep each field concise but informative (max 200 chars per field)
6. Return only valid JSON, no additional text

CV Text:
{text}"""
    
    @staticmethod
    def _clean_json_response(response_text: str) -> str:
        """Clean Gemini response to extract valid JSON"""
        text = response_text.strip()
        
        # Remove markdown code fences if present
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        
        if text.endswith('```'):
            text = text[:-3]
        
        return text.strip()


class CVScraper:
    """Main CV scraper application"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx'}
    
    def __init__(self, api_key: str, concurrency: int = 10):
        """Initialize CV scraper"""
        self.gemini_extractor = GeminiCVExtractor(api_key)
        self.text_extractor = TextExtractor()
        self.concurrency = concurrency
    
    def find_cv_files(self, input_dir: Path) -> List[Path]:
        """Find all supported CV files in the input directory"""
        cv_files = []
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                cv_files.append(file_path)
            elif file_path.is_file():
                logger.info(f"Skipping unsupported file: {file_path.name}")
        
        return cv_files
    
    def process_single_cv(self, file_path: Path, cv_id: str) -> ProcessingResult:
        """Process a single CV file"""
        logger.info(f"Processing: {file_path.name}")
        
        try:
            # Extract text based on file extension
            if file_path.suffix.lower() == '.pdf':
                text = self.text_extractor.extract_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                text = self.text_extractor.extract_from_docx(file_path)
            else:
                raise Exception(f"Unsupported file type: {file_path.suffix}")
            
            if not text.strip():
                raise Exception("No text extracted from file")
            
            # Extract CV data using Gemini
            cv_data = self.gemini_extractor.extract_cv_data(text, cv_id)
            
            # Check if we got useful data
            if not cv_data.is_valid():
                return ProcessingResult(
                    cv_data=None,
                    filename=file_path.name,
                    success=False,
                    error="No useful CV data found"
                )
            
            return ProcessingResult(
                cv_data=cv_data,
                filename=file_path.name,
                success=True
            )
            
        except Exception as e:
            return ProcessingResult(
                cv_data=None,
                filename=file_path.name,
                success=False,
                error=str(e)
            )
    
    def process_cvs_concurrently(self, cv_files: List[Path]) -> List[CVData]:
        """Process multiple CV files concurrently"""
        results = []
        successful_cvs = []
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.process_single_cv, file_path, str(i + 1)): file_path
                for i, file_path in enumerate(cv_files)
            }
            
            # Collect results
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per file
                    results.append(result)
                    
                    if result.success and result.cv_data:
                        successful_cvs.append(result.cv_data)
                        self._rename_file(file_path, result.cv_data.id)
                    else:
                        logger.warning(f"Failed to process {result.filename}: {result.error}")
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path.name}: {e}")
        
        logger.info(f"Processing complete: {len(successful_cvs)}/{len(cv_files)} files processed successfully")
        return successful_cvs
    
    def _rename_file(self, original_path: Path, cv_id: str):
        """Rename processed file with CV ID"""
        new_name = f"{cv_id}{original_path.suffix}"
        new_path = original_path.parent / new_name
        
        try:
            shutil.move(str(original_path), str(new_path))
            logger.info(f"Renamed {original_path.name} to {new_name}")
        except Exception as e:
            logger.warning(f"Could not rename {original_path.name} to {new_name}: {e}")
    
    @staticmethod
    def save_to_csv(cv_data: List[CVData], output_file: Path):
        """Save CV data to CSV file"""
        if not cv_data:
            logger.warning("No CV data to save")
            return
        
        fieldnames = ['id', 'name', 'phone', 'email', 'education', 'skills', 'experience']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for cv in cv_data:
                writer.writerow(cv.to_dict())
        
        logger.info(f"Successfully saved {len(cv_data)} CV records to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Extract CV data from PDF and DOCX files using Google Gemini AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path('input'),
        help='Input directory containing CV files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('extracted_cvs.csv'),
        help='Output CSV file'
    )
    
    parser.add_argument(
        '-c', '--concurrency',
        type=int,
        default=10,
        help='Number of concurrent workers (recommended: 5-20 for API)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get Gemini API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("Gemini API key is required. Use --api-key flag or set GEMINI_API_KEY environment variable")
        return 1
    
    # Create input directory if it doesn't exist
    if not args.input.exists():
        args.input.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created '{args.input}' directory. Please place your CV files there.")
        return 0
    
    # Initialize scraper
    scraper = CVScraper(api_key=api_key, concurrency=args.concurrency)
    
    # Find CV files
    cv_files = scraper.find_cv_files(args.input)
    if not cv_files:
        logger.info("No valid CV files found in the input directory.")
        return 0
    
    logger.info(f"Found {len(cv_files)} CV files to process")
    
    # Process CVs
    cv_data = scraper.process_cvs_concurrently(cv_files)
    
    # Save results
    if cv_data:
        scraper.save_to_csv(cv_data, args.output)
        logger.info(f"Successfully processed {len(cv_data)} CVs and saved to {args.output}")
    else:
        logger.warning("No CVs were processed successfully.")
    
    return 0


if __name__ == "__main__":
    exit(main())