package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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
		log.Fatal("Failed to load configuration:", err)
	}

	fmt.Printf("Starting CV scraper...\n")
	fmt.Printf("Fetching candidates from ID %d to %d\n", config.StartID, config.EndID)
	fmt.Printf("Using %d concurrent workers\n", config.Concurrency)

	// Create CSV file
	csvFile, err := os.Create(config.OutputFile)
	if err != nil {
		log.Fatal("Failed to create CSV file:", err)
	}
	defer csvFile.Close()

	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// Write CSV header
	header := []string{"ID", "Name", "Phone", "Email", "Experience", "Skills", "Resume Files"}
	if err := csvWriter.Write(header); err != nil {
		log.Fatal("Failed to write CSV header:", err)
	}

	// Start processing
	candidates := processCandidatesConcurrently(config, csvWriter)

	fmt.Printf("\n✓ Successfully processed %d candidates and saved to %s\n", candidates, config.OutputFile)
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
		S3Bucket:    getEnv("S3_BUCKET_URL", "https://atbjobs.s3.ap-southeast-1.amazonaws.com/candidate/resume/"),
	}

	if config.AuthToken == "" {
		return nil, fmt.Errorf("AUTH_TOKEN is required in .env file")
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
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Authorization", "Bearer "+config.AuthToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %v", err)
	}

	// Check if the response indicates success and has data
	if !apiResp.Success || apiResp.Response.Data == nil {
		return nil, nil // Skip this record
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
