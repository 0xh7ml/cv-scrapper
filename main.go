package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
)

// Config holds application configuration
type Config struct {
	BaseURL     string
	AuthToken   string
	StartID     int
	EndID       int
	Concurrency int
	OutputFile  string
	Timeout     time.Duration
	S3Bucket    string
}

// APIResponse represents the API response structure
type APIResponse struct {
	Success  bool `json:"success"`
	Response struct {
		Data *CandidateData `json:"data"`
	} `json:"response"`
}

// CandidateData represents the candidate information
type CandidateData struct {
	ID         int    `json:"id"`
	FirstName  string `json:"first_name"`
	MiddleName string `json:"middle_name"`
	LastName   string `json:"last_name"`
	User       struct {
		Email         string `json:"email"`
		ContactNumber string `json:"contact_number"`
		DialCode      string `json:"dial_code"`
	} `json:"user"`
	Skills []struct {
		SkillName string `json:"skill_name"`
	} `json:"skills"`
	CandidateExperience []struct {
		OrganizationName string `json:"organization_name"`
		OrganizationDesc string `json:"organization_desc"`
		StartDate        string `json:"start_date"`
		EndDate          string `json:"end_date"`
		CurrentlyWorking int    `json:"currently_working"`
	} `json:"candidate_experience"`
	CandidateResume []struct {
		ResumeFullURL string `json:"resume_full_url"`
	} `json:"candidate_resume"`
}

// ProcessedCandidate represents the final CSV row data
type ProcessedCandidate struct {
	ID         string
	Name       string
	Phone      string
	Email      string
	Experience string
	Skills     string
	ResumeURLs string
}

func main() {
	// Load configuration
	config, err := loadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "   Configuration Error:\n%v\n\n", err)
		fmt.Fprintf(os.Stderr, "   Please check your .env file and ensure all required values are set.\n")
		fmt.Fprintf(os.Stderr, "   Example .env file:\n")
		os.Exit(1)
	}

	fmt.Printf("🚀 Starting CV scraper...\n")
	fmt.Printf("📊 Fetching candidates from ID %d to %d\n", config.StartID, config.EndID)
	fmt.Printf("⚡ Using %d concurrent workers\n", config.Concurrency)
	fmt.Printf("📁 Output file: %s\n", config.OutputFile)
	fmt.Printf("⏱️  Request timeout: %v\n", config.Timeout)
	fmt.Println()

	// Create output directory if it doesn't exist
	outputDir := filepath.Dir(config.OutputFile)
	if outputDir != "." && outputDir != "" {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Failed to create output directory '%s': %v\n", outputDir, err)
			fmt.Fprintf(os.Stderr, "💡 Please check directory permissions and try again\n")
			os.Exit(1)
		}
	}

	// Create CSV file
	csvFile, err := os.Create(config.OutputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to create output CSV file '%s': %v\n", config.OutputFile, err)
		fmt.Fprintf(os.Stderr, "💡 Please check:\n")
		fmt.Fprintf(os.Stderr, "   - The output directory exists and is writable\n")
		fmt.Fprintf(os.Stderr, "   - You have permission to create files in this location\n")
		fmt.Fprintf(os.Stderr, "   - The file is not currently open in another program\n")
		os.Exit(1)
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// Write CSV header
	header := []string{"ID", "Name", "Phone", "Email", "Experience", "Skills", "Resume Files"}
	if err := csvWriter.Write(header); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to write CSV header: %v\n", err)
		os.Exit(1)
	}

	// Start processing
	candidates := processCandidatesConcurrently(config, csvWriter)

	if candidates > 0 {
		fmt.Printf("\nSuccessfully processed %d candidates and saved to %s\n", candidates, config.OutputFile)
	} else {
		fmt.Printf("\nNo candidates were processed successfully.\n")
		fmt.Printf("   This might be due to:\n")
		fmt.Printf("   - Invalid ID range (no candidates exist in this range)\n")
		fmt.Printf("   - Authentication issues (check your AUTH_TOKEN)\n")
		fmt.Printf("   - API connectivity problems\n")
		fmt.Printf("   - Use VERBOSE=true in .env for detailed error logging\n")
	}
}

func loadConfig() (*Config, error) {
	// Load .env file
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found, using environment variables")
	}

	config := &Config{
		BaseURL:     getEnv("API_BASE_URL", ""),
		AuthToken:   getEnv("AUTH_TOKEN", ""),
		StartID:     getEnvInt("START_ID", 1),
		EndID:       getEnvInt("END_ID", 317376),
		Concurrency: getEnvInt("CONCURRENCY", 10),
		OutputFile:  getEnv("OUTPUT_FILE", "candidates.csv"),
		Timeout:     time.Duration(getEnvInt("TIMEOUT_SECONDS", 30)) * time.Second,
		S3Bucket:    getEnv("S3_BUCKET_URL", ""),
	}

	// Validate required configuration
	var errors []string

	if config.BaseURL == "" {
		errors = append(errors, "API_BASE_URL is required")
	}

	if config.AuthToken == "" {
		errors = append(errors, "AUTH_TOKEN is required")
	}

	if config.StartID < 1 {
		errors = append(errors, "START_ID must be greater than 0")
	}

	if config.EndID < config.StartID {
		errors = append(errors, "END_ID must be greater than or equal to START_ID")
	}

	if config.Concurrency < 1 {
		errors = append(errors, "CONCURRENCY must be greater than 0")
	}

	if config.Concurrency > 50 {
		errors = append(errors, "CONCURRENCY should not exceed 50 to avoid overwhelming the API")
	}

	if config.OutputFile == "" {
		errors = append(errors, "OUTPUT_FILE cannot be empty")
	}

	if config.Timeout < time.Second {
		errors = append(errors, "TIMEOUT_SECONDS must be at least 1 second")
	}

	if config.S3Bucket == "" {
		errors = append(errors, "S3_BUCKET_URL cannot be empty")
	}

	// If there are validation errors, return them
	if len(errors) > 0 {
		return nil, fmt.Errorf("configuration validation failed:\n  - %s", strings.Join(errors, "\n  - "))
	}

	return config, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func processCandidatesConcurrently(config *Config, csvWriter *csv.Writer) int {
	jobs := make(chan int, config.EndID-config.StartID+1)
	results := make(chan *ProcessedCandidate, 100)

	var wg sync.WaitGroup
	var mu sync.Mutex // Mutex for CSV writing
	processedCount := 0

	// Start workers
	for i := 0; i < config.Concurrency; i++ {
		wg.Add(1)
		go worker(jobs, results, config, &wg)
	}

	// Send jobs
	go func() {
		for id := config.StartID; id <= config.EndID; id++ {
			jobs <- id
		}
		close(jobs)
	}()

	// Close results when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Process results
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("Processing candidates...")
	fmt.Println(strings.Repeat("=", 60))

	for candidate := range results {
		if candidate != nil {
			mu.Lock()
			record := []string{
				candidate.ID,
				candidate.Name,
				candidate.Phone,
				candidate.Email,
				candidate.Experience,
				candidate.Skills,
				candidate.ResumeURLs,
			}
			if err := csvWriter.Write(record); err != nil {
				log.Printf("Error writing CSV record for ID %s: %v", candidate.ID, err)
			} else {
				processedCount++
				if processedCount%100 == 0 {
					fmt.Printf("Processed %d candidates...\n", processedCount)
				}
			}
			mu.Unlock()
		}
	}

	fmt.Println(strings.Repeat("=", 60))
	return processedCount
}

func worker(jobs <-chan int, results chan<- *ProcessedCandidate, config *Config, wg *sync.WaitGroup) {
	defer wg.Done()

	client := &http.Client{
		Timeout: config.Timeout,
	}

	for id := range jobs {
		candidate, err := fetchCandidate(client, id, config)
		if err != nil {
			// Only log errors in verbose mode to avoid spam
			if os.Getenv("VERBOSE") == "true" {
				log.Printf("Error fetching candidate ID %d: %v", id, err)
			}
			continue
		}

		if candidate != nil {
			results <- candidate
		}
	}
}

func fetchCandidate(client *http.Client, id int, config *Config) (*ProcessedCandidate, error) {
	url := fmt.Sprintf("%s/%d", config.BaseURL, id)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request for ID %d: %v", id, err)
	}

	req.Header.Set("Authorization", "Bearer "+config.AuthToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed for ID %d: %v", id, err)
	}
	defer resp.Body.Close()

	// Handle different HTTP status codes with specific messages
	switch resp.StatusCode {
	case http.StatusOK:
		// Continue processing
	case http.StatusUnauthorized:
		return nil, fmt.Errorf("authentication failed for ID %d - please check your AUTH_TOKEN", id)
	case http.StatusForbidden:
		return nil, fmt.Errorf("access forbidden for ID %d - insufficient permissions", id)
	case http.StatusNotFound:
		return nil, fmt.Errorf("candidate ID %d not found", id)
	case http.StatusTooManyRequests:
		return nil, fmt.Errorf("rate limit exceeded for ID %d - consider reducing CONCURRENCY", id)
	case http.StatusInternalServerError:
		return nil, fmt.Errorf("server error for ID %d - API may be experiencing issues", id)
	default:
		return nil, fmt.Errorf("unexpected HTTP status %d for ID %d", resp.StatusCode, id)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body for ID %d: %v", id, err)
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("empty response body for ID %d", id)
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response for ID %d: %v", id, err)
	}

	// Check if the response indicates success and has data
	if !apiResp.Success {
		return nil, nil // Skip this record - API indicates no success
	}

	if apiResp.Response.Data == nil {
		return nil, nil // Skip this record - no data available
	}

	return processCandidate(apiResp.Response.Data, config), nil
}

func processCandidate(data *CandidateData, config *Config) *ProcessedCandidate {
	// Build full name
	var nameParts []string
	if data.FirstName != "" {
		nameParts = append(nameParts, data.FirstName)
	}
	if data.MiddleName != "" {
		nameParts = append(nameParts, data.MiddleName)
	}
	if data.LastName != "" {
		nameParts = append(nameParts, data.LastName)
	}
	fullName := strings.Join(nameParts, " ")

	// Build phone number
	phone := ""
	if data.User.ContactNumber != "" {
		if data.User.DialCode != "" {
			phone = data.User.DialCode + data.User.ContactNumber
		} else {
			phone = data.User.ContactNumber
		}
	}

	// Extract skills
	var skills []string
	for _, skill := range data.Skills {
		if skill.SkillName != "" {
			skills = append(skills, skill.SkillName)
		}
	}
	skillsStr := strings.Join(skills, ", ")

	// Extract experience
	var experiences []string
	for _, exp := range data.CandidateExperience {
		if exp.OrganizationName != "" {
			expStr := exp.OrganizationName
			if exp.OrganizationDesc != "" {
				// Limit description length
				desc := exp.OrganizationDesc
				if len(desc) > 200 {
					desc = desc[:200] + "..."
				}
				expStr += " - " + desc
			}
			experiences = append(experiences, expStr)
		}
	}
	experienceStr := strings.Join(experiences, " | ")

	// Extract resume URLs - extract filename only
	var resumeFilenames []string
	for _, resume := range data.CandidateResume {
		if resume.ResumeFullURL != "" {
			// Extract filename from URL by removing the base path
			filename := strings.TrimPrefix(resume.ResumeFullURL, config.S3Bucket)

			// If the URL doesn't match the expected pattern, extract filename from the last part
			if filename == resume.ResumeFullURL {
				// Fallback: get the last part after the last "/"
				parts := strings.Split(resume.ResumeFullURL, "/")
				if len(parts) > 0 {
					filename = parts[len(parts)-1]
				}
			}

			if filename != "" {
				resumeFilenames = append(resumeFilenames, filename)
			}
		}
	}
	resumeFilenamesStr := strings.Join(resumeFilenames, ", ")

	return &ProcessedCandidate{
		ID:         strconv.Itoa(data.ID),
		Name:       fullName,
		Phone:      phone,
		Email:      data.User.Email,
		Experience: experienceStr,
		Skills:     skillsStr,
		ResumeURLs: resumeFilenamesStr,
	}
}
