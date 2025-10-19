# CV Scraper with Google Gemini AI

A Go application that extracts information from CV files (PDF and DOCX) using Google's Gemini 1.5 Flash model for high accuracy and saves the data to a CSV file.

## Features

- **🆓 FREE with Generous Limits**: Uses Gemini 1.5 Flash with 15 requests/minute, 1M tokens/day
- **High Accuracy**: Intelligent CV data extraction with context understanding
- Processes PDF and DOCX files
- Automatically renames files incrementally (1.pdf, 2.docx, etc.)
- Extracts: Name, Phone, Email, Education, Skills, Experience
- Concurrent processing for speed (with API rate limiting)
- Saves data to CSV with CV-ID
- Skips files with no useful data

## Setup

1. **Get Gemini API Key** (FREE): 
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Click "Get API key" and create a new API key
   - No credit card required for free tier!

2. **Install Go dependencies**:
```bash
go mod tidy
```

3. **Set up API key** (choose one method):
   - Environment variable: `export GEMINI_API_KEY="your-api-key"`
   - Command line flag: `-key "your-api-key"`

4. **Create input directory and add CV files**:
```bash
mkdir input
# Copy your PDF/DOCX CV files to the input directory
```

## Usage

### Basic Usage with Environment Variable:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
go run main.go -i ./cvs -c 10 -o data.csv
```

### Using API Key Flag:
```bash
go run main.go -i ./cvs -c 10 -o data.csv -key "your-gemini-api-key"
```

### Build and Run:
```bash
go build -o cv-scrapper-gemini
./cv-scrapper-gemini -i ./cvs -c 10 -o data.csv
```

## Command Line Options:
- `-i`: Input directory containing CV files (default: "input")
- `-c`: Number of concurrent goroutines (default: 10, recommended: 5-15 for API)
- `-o`: Output CSV file name (default: "extracted_cvs.csv")
- `-key`: Gemini API key (or set GEMINI_API_KEY environment variable)

## 🎯 Gemini Free Tier Benefits

**FREE Usage Limits**:
- ✅ **15 requests per minute**
- ✅ **1 million tokens per day**
- ✅ **1,500 requests per day**

**Cost Comparison**:
- **Gemini 1.5 Flash**: 100% FREE for moderate usage
- **OpenAI GPT-4o-mini**: ~$0.003 per CV
- **OpenAI GPT-4**: ~$0.05 per CV

**Daily Processing Capacity (FREE)**:
- ~**500-700 CVs per day** (depending on CV length)
- Perfect for small to medium businesses
- No credit card required

## Performance vs Accuracy

**Gemini Integration Benefits**:
- ✅ **Near 100% accuracy** with intelligent context understanding
- ✅ **Handles various CV formats** and layouts
- ✅ **Extracts complex information** like work experience summaries
- ✅ **Normalizes data** (phone numbers, education formats)
- ✅ **Multi-language support**
- ✅ **Completely FREE** for most use cases

**Recommended Settings**:
- Concurrency: 10-15 (respects API rate limits)
- Model: gemini-1.5-flash (free and fast)

## Output

The CSV file contains:
- CV-ID: Incremental ID (1, 2, 3, etc.)
- Name: Full name extracted intelligently
- Phone: Normalized phone number with country code
- Email: Email address
- Education: Summarized educational background
- Skills: Technical skills, programming languages, tools
- Experience: Work experience with companies and roles

## Requirements

- Go 1.19 or higher
- Gemini API key (FREE from Google AI Studio)
- Internet connection

## Dependencies

- `github.com/ledongthuc/pdf` - PDF text extraction
- `github.com/unidoc/unioffice` - DOCX text extraction  
- `github.com/google/generative-ai-go` - Google Gemini API client

## Error Handling

The application includes robust error handling for:
- API rate limiting and timeouts
- Invalid CV formats
- Network connectivity issues
- JSON parsing errors
- File access problems

## 💡 Pro Tips

1. **Stay within rate limits**: Use concurrency of 10-15 to respect the 15 requests/minute limit
2. **Monitor usage**: Track your daily token usage at [Google AI Studio](https://aistudio.google.com/)
3. **Optimize for free tier**: Process in batches to maximize your 1M tokens/day limit