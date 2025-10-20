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

// Config holds application configuration
type Config struct {
	InputDir    string
	Concurrency int
	OutputFile  string
	APIKey      string
	RetryCount  int
	RetryDelay  time.Duration
}

func main() {
	// Command line flags
	config := Config{}
	flag.StringVar(&config.InputDir, "i", "input", "Input directory containing CV files")
	flag.IntVar(&config.Concurrency, "c", 10, "Number of concurrent goroutines (recommended: 10-20 for API)")
	flag.StringVar(&config.OutputFile, "o", "extracted_cvs.csv", "Output CSV file")
	flag.StringVar(&config.APIKey, "key", "", "Gemini API key (or set GEMINI_API_KEY env var)")
	flag.IntVar(&config.RetryCount, "retry", 3, "Number of retries for failed API calls")
	flag.Parse()

	config.RetryDelay = 2 * time.Second

	// Get Gemini API key
	if config.APIKey == "" {
		config.APIKey = os.Getenv("GEMINI_API_KEY")
	}
	if config.APIKey == "" {
		log.Fatal("Gemini API key is required. Use -key flag or set GEMINI_API_KEY environment variable")
	}

	// Create input directory if it doesn't exist
	if err := ensureInputDirectory(config.InputDir); err != nil {
		log.Fatal(err)
	}

	// Get valid CV files
	validFiles, err := getValidCVFiles(config.InputDir)
	if err != nil {
		log.Fatal("Error reading input directory:", err)
	}

	if len(validFiles) == 0 {
		fmt.Println("No valid CV files found in the input directory.")
		return
	}

	fmt.Printf("Found %d CV files to process\n", len(validFiles))

	// Create Gemini client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(config.APIKey))
	if err != nil {
		log.Fatal("Failed to create Gemini client:", err)
	}
	defer client.Close()

	// Process files concurrently
	cvData := processCVsConcurrently(validFiles, config, client)

	// Save to CSV
	if len(cvData) > 0 {
		if err := saveToCSV(cvData, config.OutputFile); err != nil {
			log.Fatal("Error saving to CSV:", err)
		}
		fmt.Printf("\n✓ Successfully processed %d CVs and saved to %s\n", len(cvData), config.OutputFile)
	} else {
		fmt.Println("\n✗ No CVs were processed successfully.")
	}
}

func ensureInputDirectory(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.Mkdir(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory: %v", err)
		}
		fmt.Printf("Created '%s' directory. Please place your CV files there.\n", dir)
		os.Exit(0)
	}
	return nil
}

func getValidCVFiles(inputDir string) ([]string, error) {
	files, err := os.ReadDir(inputDir)
	if err != nil {
		return nil, err
	}

	var validFiles []string
	supportedExts := map[string]bool{".pdf": true, ".docx": true, ".doc": true}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		fileName := file.Name()
		ext := strings.ToLower(filepath.Ext(fileName))

		if supportedExts[ext] {
			validFiles = append(validFiles, fileName)
		} else {
			fmt.Printf("⊘ Skipping unsupported file: %s\n", fileName)
		}
	}

	return validFiles, nil
}

func processCVsConcurrently(validFiles []string, config Config, client *genai.Client) []CVData {
	jobs := make(chan FileJob, len(validFiles))
	results := make(chan ProcessedCV, len(validFiles))

	// Create worker pool
	var wg sync.WaitGroup
	for i := 0; i < config.Concurrency; i++ {
		wg.Add(1)
		go worker(jobs, results, config, client, &wg)
	}

	// Send jobs
	go func() {
		for i, fileName := range validFiles {
			filePath := filepath.Join(config.InputDir, fileName)
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

	// Collect results with timeout protection
	var cvData []CVData
	successCount := 0
	failureCount := 0

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Processing CVs...")
	fmt.Println(strings.Repeat("=", 60))

	timeout := time.After(30 * time.Minute) // Generous timeout

	for {
		select {
		case result, ok := <-results:
			if !ok {
				// Channel closed, we're done
				goto finish
			}
			if result.Success {
				successCount++
				fmt.Printf("[%d] %s - Successfully processed\n", successCount+failureCount, result.FileName)

				// Rename file
				if err := renameProcessedFile(config.InputDir, result.FileName, result.CV.ID); err != nil {
					fmt.Printf("Warning: Could not rename file: %v\n", err)
				}

				cvData = append(cvData, result.CV)
			} else {
				failureCount++
				if result.Error != nil {
					fmt.Printf("[%d] %s - Error: %v\n", successCount+failureCount, result.FileName, result.Error)
				} else {
					fmt.Printf("[%d] %s - No useful data found\n", successCount+failureCount, result.FileName)
				}
			}
		case <-timeout:
			fmt.Printf("\n⚠ Processing timed out after 30 minutes\n")
			goto finish
		}
	}

finish:
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Processing complete: %d succeeded, %d failed\n", successCount, failureCount)
	fmt.Println(strings.Repeat("=", 60))

	return cvData
}

func worker(jobs <-chan FileJob, results chan<- ProcessedCV, config Config, client *genai.Client, wg *sync.WaitGroup) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Worker panic recovered: %v", r)
		}
		wg.Done()
	}()

	for job := range jobs {
		// Extract text from file
		text, err := extractTextFromFile(job.FilePath)
		if err != nil {
			results <- ProcessedCV{
				FileName: job.FileName,
				Success:  false,
				Error:    fmt.Errorf("text extraction failed: %v", err),
			}
			continue
		}

		// Validate extracted text
		if strings.TrimSpace(text) == "" {
			results <- ProcessedCV{
				FileName: job.FileName,
				Success:  false,
				Error:    fmt.Errorf("extracted text is empty"),
			}
			continue
		}

		// Extract CV data using Gemini with retry logic
		cv, err := extractCVDataWithRetry(text, fmt.Sprintf("%03d", job.ID), client, config.RetryCount, config.RetryDelay)
		if err != nil {
			results <- ProcessedCV{
				FileName: job.FileName,
				Success:  false,
				Error:    err,
			}
			continue
		}

		// Skip if no useful data found
		if isEmptyCV(cv) {
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

func extractTextFromFile(filePath string) (string, error) {
	ext := strings.ToLower(filepath.Ext(filePath))

	switch ext {
	case ".pdf":
		return extractTextFromPDF(filePath)
	case ".docx", ".doc":
		return extractTextFromDOCX(filePath)
	default:
		return "", fmt.Errorf("unsupported file format: %s", ext)
	}
}

func extractCVDataWithRetry(text, id string, client *genai.Client, retryCount int, retryDelay time.Duration) (CVData, error) {
	var lastErr error

	for attempt := 0; attempt <= retryCount; attempt++ {
		if attempt > 0 {
			time.Sleep(retryDelay * time.Duration(attempt))
		}

		cv, err := extractCVDataWithGemini(text, id, client)
		if err == nil {
			return cv, nil
		}

		lastErr = err
		if attempt < retryCount {
			log.Printf("Retry %d/%d for CV %s: %v", attempt+1, retryCount, id, err)
		}
	}

	return CVData{}, fmt.Errorf("failed after %d retries: %v", retryCount, lastErr)
}

func extractCVDataWithGemini(text, id string, client *genai.Client) (CVData, error) {
	// Truncate text if too long (Gemini has token limits)
	if len(text) > 10000 {
		text = text[:10000] + "... [truncated]"
	}

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
6. Return ONLY valid JSON, no additional text or markdown formatting

CV Text:
` + text

	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	model := client.GenerativeModel("gemini-2.0-flash")
	model.SetTemperature(0.1)
	model.SetMaxOutputTokens(800)

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return CVData{}, fmt.Errorf("gemini API error: %v", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return CVData{}, fmt.Errorf("empty response from Gemini")
	}

	// Extract and clean content
	content := extractContentFromResponse(resp.Candidates[0].Content.Parts[0])
	content = cleanJSONResponse(content)

	var aiResp GeminiResponse
	if err := json.Unmarshal([]byte(content), &aiResp); err != nil {
		return CVData{}, fmt.Errorf("failed to parse JSON: %v (content: %s)", err, content)
	}

	return CVData{
		ID:         id,
		Name:       strings.TrimSpace(aiResp.Name),
		Phone:      strings.TrimSpace(aiResp.Phone),
		Email:      strings.TrimSpace(aiResp.Email),
		Education:  strings.TrimSpace(aiResp.Education),
		Skills:     strings.TrimSpace(aiResp.Skills),
		Experience: strings.TrimSpace(aiResp.Experience),
	}, nil
}

func extractContentFromResponse(part interface{}) string {
	if textPart, ok := part.(genai.Text); ok {
		return string(textPart)
	}
	return fmt.Sprintf("%v", part)
}

func cleanJSONResponse(content string) string {
	content = strings.TrimSpace(content)

	// Remove markdown code fences
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```JSON")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	// Find JSON object bounds
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")

	if start != -1 && end != -1 && end > start {
		content = content[start : end+1]
	}

	return content
}

func isEmptyCV(cv CVData) bool {
	return cv.Name == "" && cv.Phone == "" && cv.Email == ""
}

func renameProcessedFile(inputDir, oldFileName, id string) error {
	ext := strings.ToLower(filepath.Ext(oldFileName))
	newFileName := fmt.Sprintf("%s%s", id, ext)
	oldPath := filepath.Join(inputDir, oldFileName)
	newPath := filepath.Join(inputDir, newFileName)

	// Don't rename if already has the target name
	if oldFileName == newFileName {
		return nil
	}

	// Check if target file already exists
	if _, err := os.Stat(newPath); err == nil {
		return fmt.Errorf("file %s already exists", newFileName)
	}

	return os.Rename(oldPath, newPath)
}

func extractTextFromPDF(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF: %v", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to stat PDF: %v", err)
	}

	pdfReader, err := pdf.NewReader(file, fileInfo.Size())
	if err != nil {
		return "", fmt.Errorf("failed to create PDF reader: %v", err)
	}

	var text strings.Builder
	pageCount := pdfReader.NumPage()

	for i := 1; i <= pageCount; i++ {
		page := pdfReader.Page(i)
		if page.V.IsNull() {
			continue
		}

		pageText, err := page.GetPlainText(nil)
		if err != nil {
			log.Printf("Warning: Could not extract text from page %d: %v", i, err)
			continue
		}

		text.WriteString(pageText)
		text.WriteString("\n")
	}

	result := text.String()
	if strings.TrimSpace(result) == "" {
		return "", fmt.Errorf("no text content extracted from PDF")
	}

	return result, nil
}

// extractTextFromDOCX reads .docx/.doc files as zip archives and extracts text from word/document.xml
// This works for both .docx and many .doc files saved in the Office Open XML format
func extractTextFromDOCX(filePath string) (string, error) {
	zr, err := zip.OpenReader(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open as zip archive: %v", err)
	}
	defer zr.Close()

	// Find document.xml
	var docFile *zip.File
	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			docFile = f
			break
		}
	}

	if docFile == nil {
		return "", fmt.Errorf("word/document.xml not found in archive")
	}

	rc, err := docFile.Open()
	if err != nil {
		return "", fmt.Errorf("failed to open document.xml: %v", err)
	}
	defer rc.Close()

	data, err := io.ReadAll(rc)
	if err != nil {
		return "", fmt.Errorf("failed to read document.xml: %v", err)
	}

	return parseWordXML(data)
}

// parseWordXML extracts text from Word XML content
func parseWordXML(data []byte) (string, error) {
	decoder := xml.NewDecoder(bytes.NewReader(data))
	var sb strings.Builder
	var inParagraph bool

	for {
		tok, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("XML parsing error: %v", err)
		}

		switch se := tok.(type) {
		case xml.StartElement:
			// Track paragraphs for better text formatting
			if se.Name.Local == "p" {
				inParagraph = true
			}
			// Extract text nodes (w:t elements)
			if se.Name.Local == "t" {
				var content string
				if err := decoder.DecodeElement(&content, &se); err == nil {
					sb.WriteString(content)
				}
			}
		case xml.EndElement:
			// Add newline after paragraphs
			if se.Name.Local == "p" && inParagraph {
				sb.WriteString("\n")
				inParagraph = false
			}
		}
	}

	result := strings.TrimSpace(sb.String())
	if result == "" {
		return "", fmt.Errorf("no text content extracted from document")
	}

	return result, nil
}

func saveToCSV(data []CVData, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"CV-ID", "Name", "Phone", "Email", "Education", "Skills", "Experience"}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %v", err)
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
			return fmt.Errorf("failed to write CSV record: %v", err)
		}
	}

	return nil
}
