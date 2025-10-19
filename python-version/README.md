# CV Scraper - Python Version

A Python-based CV/Resume data extraction tool that uses Google Gemini AI to extract structured information from PDF and DOCX files.

## Features

- **Multi-format Support**: Extracts text from PDF and DOCX files
- **AI-Powered Extraction**: Uses Google Gemini AI for intelligent data extraction
- **Concurrent Processing**: Processes multiple files simultaneously for better performance
- **Robust Text Extraction**: Multiple fallback methods for reliable text extraction
- **Clean Output**: Exports extracted data to CSV format
- **File Management**: Automatically renames processed files with extracted CV IDs

## Installation

1. **Navigate to the Python version directory**:
```bash
cd python-version
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Google Gemini API**:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set the environment variable:
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```
   - Or pass it directly via command line (see usage below)

## Usage

### Basic Usage

```bash
python cv_scraper.py
```

This will:
- Look for CV files in the `input/` directory
- Process all PDF and DOCX files found
- Save extracted data to `extracted_cvs.csv`

### Advanced Usage

```bash
python cv_scraper.py -i /path/to/cv/files -o output.csv -c 15 --api-key YOUR_API_KEY -v
```

### Command Line Options

- `-i, --input`: Input directory containing CV files (default: `input/`)
- `-o, --output`: Output CSV file (default: `extracted_cvs.csv`)
- `-c, --concurrency`: Number of concurrent workers (default: 10, recommended: 5-20)
- `--api-key`: Gemini API key (or set `GEMINI_API_KEY` environment variable)
- `-v, --verbose`: Enable verbose logging
- `-h, --help`: Show help message

### Example Workflow

1. **Prepare your files**:
```bash
mkdir input
# Copy your CV files (PDF/DOCX) to the input directory
cp /path/to/cvs/*.pdf input/
cp /path/to/cvs/*.docx input/
```

2. **Run the scraper**:
```bash
export GEMINI_API_KEY="your-api-key-here"
python cv_scraper.py -v
```

3. **Check results**:
   - View the generated `extracted_cvs.csv` file
   - Original files will be renamed with their extracted CV IDs

## Output Format

The tool extracts the following information and saves it to CSV:

| Field | Description |
|-------|-------------|
| `id` | Auto-generated CV ID |
| `name` | Full name of the person |
| `phone` | Phone number |
| `email` | Email address |
| `education` | Educational background (degrees, institutions) |
| `skills` | Technical skills and competencies |
| `experience` | Work experience and job history |

## Supported File Formats

- **PDF**: Uses PyMuPDF (primary) with PyPDF2 fallback
- **DOCX**: Uses python-docx (primary) with ZIP/XML fallback

## Configuration

You can modify `config.py` to adjust:
- Gemini model settings (temperature, tokens, etc.)
- Processing timeouts and concurrency limits
- File naming patterns and supported extensions
- Extraction prompt templates

## Error Handling

The tool includes comprehensive error handling:
- **File Processing Errors**: Logged with specific error messages
- **API Errors**: Retry logic and detailed error reporting
- **Text Extraction Failures**: Multiple fallback methods
- **Invalid Data**: Skips files with insufficient extractable information

## Performance Tips

1. **Optimal Concurrency**: Start with 10 workers, adjust based on your system and API limits
2. **File Organization**: Keep CV files in a dedicated directory for better organization
3. **API Key Management**: Use environment variables for better security
4. **Batch Processing**: Process files in batches if you have hundreds of CVs

## Troubleshooting

### Common Issues

1. **"No text extracted from file"**
   - File might be corrupted or password-protected
   - Try opening the file manually to verify it's readable

2. **"Gemini API error"**
   - Check your API key is valid and has sufficient quota
   - Verify internet connection
   - Check if the Gemini service is available

3. **"Failed to parse Gemini response as JSON"**
   - The AI response wasn't in proper JSON format
   - This is usually temporary; try running again

4. **Permission errors when renaming files**
   - Ensure the script has write permissions to the input directory
   - Close any applications that might have the files open

### Debugging

Enable verbose logging to see detailed processing information:
```bash
python cv_scraper.py -v
```

## Dependencies

- `PyMuPDF` (fitz): Advanced PDF text extraction
- `PyPDF2`: Fallback PDF processing
- `python-docx`: DOCX file processing
- `google-generativeai`: Google Gemini AI client

## Differences from Go Version

The Python version includes several improvements:

1. **Better Structure**: Object-oriented design with clear separation of concerns
2. **Enhanced Error Handling**: More robust error handling and logging
3. **Improved Text Extraction**: Better PDF extraction with PyMuPDF
4. **Configuration Management**: Separate config file for easy customization
5. **Type Hints**: Full type annotations for better code maintainability
6. **Dataclasses**: Clean data structures for CV information
7. **Better CLI**: More intuitive command-line interface with help text

## License

This project is open source. Please check the license file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.