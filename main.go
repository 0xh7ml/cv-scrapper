package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/generative-ai-go/genai"
	"github.com/ledongthuc/pdf"
	"github.com/unidoc/unioffice/document"
	"google.golang.org/api/option"
)

type CVData struct {
	ID         string
	Name       string
	Phone      string
	Email      string
	Education  string
	Skills     string
	Experience string
}
type FileJob struct {
	FileName string
	FilePath string
	ID       int
}
type ProcessedCV struct {
	CV       CVData
	FileName string
	Success  bool
	Error    error
}
type GeminiResponse struct {
	Name       string `json:"name"`
	Phone      string `json:"phone"`
	Email      string `json:"email"`
	Education  string `json:"education"`
	Skills     string `json:"skills"`
	Experience string `json:"experience"`
}

func main() {
	// Command line flags
	var inputDir = flag.String("i", "input", "Input directory containing CV files")
	var concurrency = flag.Int("c", 10, "Number of concurrent goroutines (recommended: 10-20 for API)")
	var outputFile = flag.String("o", "extracted_cvs.csv", "Output CSV file")
	var apiKey = flag.String("key", "", "Gemini API key (or set GEMINI_API_KEY env var)")
	flag.Parse()

	// Get Gemini API key
	geminiKey := *apiKey
	if geminiKey == "" {
		geminiKey = os.Getenv("GEMINI_API_KEY")
	}
	if geminiKey == "" {
		log.Fatal("Gemini API key is required. Use -key flag or set GEMINI_API_KEY environment variable")
	}

	// Create input directory if it doesn't exist
	if _, err := os.Stat(*inputDir); os.IsNotExist(err) {
		os.Mkdir(*inputDir, 0755)
		fmt.Printf("Created '%s' directory. Please place your CV files there.\n", *inputDir)
		return
	}

	// Process files in input directory
	files, err := os.ReadDir(*inputDir)
	if err != nil {
		log.Fatal("Error reading input directory:", err)
	}

	// Filter valid CV files
	var validFiles []string
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		fileName := file.Name()
		ext := strings.ToLower(filepath.Ext(fileName))

		if ext == ".pdf" || ext == ".docx" {
			validFiles = append(validFiles, fileName)
		} else {
			fmt.Printf("Skipping unsupported file: %s\n", fileName)
		}
	}

	if len(validFiles) == 0 {
		fmt.Println("No valid CV files found in the input directory.")
		return
	}

	fmt.Printf("Found %d CV files to process\n", len(validFiles))

	// Create Gemini client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		log.Fatal("Failed to create Gemini client:", err)
	}
	defer client.Close()

	// Process files concurrently
	cvData := processCVsConcurrently(validFiles, *inputDir, *concurrency, client)

	// Save to CSV
	if len(cvData) > 0 {
		err = saveToCSV(cvData, *outputFile)
		if err != nil {
			log.Fatal("Error saving to CSV:", err)
		}
		fmt.Printf("Successfully processed %d CVs and saved to %s\n", len(cvData), *outputFile)
	} else {
		fmt.Println("No CVs were processed successfully.")
	}
}

func processCVsConcurrently(validFiles []string, inputDir string, concurrency int, client *genai.Client) []CVData {
	jobs := make(chan FileJob, len(validFiles))
	results := make(chan ProcessedCV, len(validFiles))

	// Create worker pool
	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go worker(jobs, results, inputDir, client, &wg)
	}

	// Send jobs
	go func() {
		for i, fileName := range validFiles {
			filePath := filepath.Join(inputDir, fileName)
			jobs <- FileJob{
				FileName: fileName,
				FilePath: filePath,
				ID:       i + 1,
			}
		}
		close(jobs)
	}()

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	var cvData []CVData
	successCount := 0

	for result := range results {
		if result.Success {
			successCount++

			// Rename file
			ext := strings.ToLower(filepath.Ext(result.FileName))
			newFileName := fmt.Sprintf("%s%s", result.CV.ID, ext)
			oldPath := filepath.Join(inputDir, result.FileName)
			newPath := filepath.Join(inputDir, newFileName)

			err := os.Rename(oldPath, newPath)
			if err != nil {
				fmt.Printf("Warning: Could not rename %s to %s: %v\n", result.FileName, newFileName, err)
			} else {
				fmt.Printf("Renamed %s to %s\n", result.FileName, newFileName)
			}

			cvData = append(cvData, result.CV)
		} else if result.Error != nil {
			fmt.Printf("Error processing %s: %v\n", result.FileName, result.Error)
		} else {
			fmt.Printf("No useful data found in %s, skipping...\n", result.FileName)
		}
	}

	fmt.Printf("Processing complete: %d/%d files processed successfully\n", successCount, len(validFiles))
	return cvData
}

func worker(jobs <-chan FileJob, results chan<- ProcessedCV, inputDir string, client *genai.Client, wg *sync.WaitGroup) {
	defer wg.Done()
	for job := range jobs {
		fmt.Printf("Processing: %s\n", job.FileName)
		ext := strings.ToLower(filepath.Ext(job.FileName))
		// Extract text from file
		var text string
		var err error
		if ext == ".pdf" {
			text, err = extractTextFromPDF(job.FilePath)
		} else if ext == ".docx" {
			// Try the unioffice extractor first (may require a license). If it fails, fall back to a zip/xml-based reader.
			text, err = extractTextFromDOCX(job.FilePath)
			if err != nil {
				// Attempt fallback extractor
				log.Printf("unioffice extractor failed for %s: %v. Trying fallback extractor...", job.FileName, err)
				fbText, fbErr := extractTextFromDOCXFallback(job.FilePath)
				if fbErr == nil {
					text = fbText
					err = nil
				} else {
					// keep original error if fallback also fails
					log.Printf("fallback extractor also failed for %s: %v", job.FileName, fbErr)
				}
			}
		}
		if err != nil {
			results <- ProcessedCV{
				FileName: job.FileName,
				Success:  false,
				Error:    err,
			}
			continue
		}
		// Extract CV data using Gemini
		cv, err := extractCVDataWithGemini(text, fmt.Sprintf("%d", job.ID), client)
		if err != nil {
			results <- ProcessedCV{
				FileName: job.FileName,
				Success:  false,
				Error:    err,
			}
			continue
		}

		// Skip if no useful data found
		if cv.Name == "" && cv.Phone == "" && cv.Email == "" {
			results <- ProcessedCV{
				FileName: job.FileName,
				Success:  false,
				Error:    nil,
			}
			continue
		}

		results <- ProcessedCV{
			CV:       cv,
			FileName: job.FileName,
			Success:  true,
			Error:    nil,
		}
	}
}
func extractCVDataWithGemini(text, id string, client *genai.Client) (CVData, error) {
	prompt := `You are an expert CV/Resume data extraction system. Extract the following information from the CV text and return it as a JSON object.

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
` + text

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	model := client.GenerativeModel("gemini-2.0-flash")
	model.SetTemperature(0.1) // Low temperature for consistent extraction
	model.SetMaxOutputTokens(500)

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return CVData{}, fmt.Errorf("gemini API error: %v", err)
	}

	if len(resp.Candidates) == 0 {
		return CVData{}, fmt.Errorf("no candidates in Gemini response")
	}

	if len(resp.Candidates[0].Content.Parts) == 0 {
		return CVData{}, fmt.Errorf("no content parts in Gemini response")
	}

	// Extract content as string, handling the Text type properly
	part := resp.Candidates[0].Content.Parts[0]
	content := ""
	if textPart, ok := part.(genai.Text); ok {
		content = string(textPart)
	} else {
		content = fmt.Sprintf("%v", part)
	}

	// Clean the content - remove markdown code fences if present
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var aiResp GeminiResponse
	err = json.Unmarshal([]byte(content), &aiResp)
	if err != nil {
		return CVData{}, fmt.Errorf("failed to parse gemini response: %v (content: %s)", err, content)
	}

	return CVData{
		ID:         id,
		Name:       aiResp.Name,
		Phone:      aiResp.Phone,
		Email:      aiResp.Email,
		Education:  aiResp.Education,
		Skills:     aiResp.Skills,
		Experience: aiResp.Experience,
	}, nil
}

func extractTextFromPDF(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return "", err
	}

	pdfReader, err := pdf.NewReader(file, fileInfo.Size())
	if err != nil {
		return "", err
	}

	var text strings.Builder
	for i := 1; i <= pdfReader.NumPage(); i++ {
		page := pdfReader.Page(i)
		if page.V.IsNull() {
			continue
		}

		pageText, err := page.GetPlainText(nil)
		if err != nil {
			continue
		}
		text.WriteString(pageText)
		text.WriteString("\n")
	}

	return text.String(), nil
}

func extractTextFromDOCX(filePath string) (string, error) {
	doc, err := document.Open(filePath)
	if err != nil {
		return "", err
	}
	defer doc.Close()

	var text strings.Builder
	for _, para := range doc.Paragraphs() {
		for _, run := range para.Runs() {
			text.WriteString(run.Text())
		}
		text.WriteString("\n")
	}

	return text.String(), nil
}

// extractTextFromDOCXFallback reads the .docx file as a zip archive, finds word/document.xml
// and extracts text from <w:t> elements. This avoids the need for a unioffice license.
func extractTextFromDOCXFallback(filePath string) (string, error) {
	zr, err := zip.OpenReader(filePath)
	if err != nil {
		return "", err
	}
	defer zr.Close()

	var docFile *zip.File
	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			docFile = f
			break
		}
	}
	if docFile == nil {
		return "", fmt.Errorf("word/document.xml not found in docx")
	}

	rc, err := docFile.Open()
	if err != nil {
		return "", err
	}
	defer rc.Close()

	data, err := io.ReadAll(rc)
	if err != nil {
		return "", err
	}

	decoder := xml.NewDecoder(bytes.NewReader(data))
	var sb strings.Builder
	for {
		tok, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch se := tok.(type) {
		case xml.StartElement:
			// capture text nodes; local name may be "t" (w:t)
			if se.Name.Local == "t" {
				var content string
				if err := decoder.DecodeElement(&content, &se); err == nil {
					sb.WriteString(content)
					sb.WriteString(" ")
				}
			}
		}
	}

	return strings.TrimSpace(sb.String()), nil
}

func saveToCSV(data []CVData, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"CV-ID", "Name", "Phone", "Email", "Education", "Skills", "Experience"}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write data
	for _, cv := range data {
		record := []string{
			cv.ID,
			cv.Name,
			cv.Phone,
			cv.Email,
			cv.Education,
			cv.Skills,
			cv.Experience,
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
