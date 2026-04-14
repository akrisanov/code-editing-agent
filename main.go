package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: OPENROUTER_API_KEY is not set")
		os.Exit(1)
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithHeader("X-OpenRouter-Title", "Code Editing Agent"),
	)

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{
		ReadFileDefinition,
		ListFilesDefinition,
		EditFileDefinition,
	}
	agent := NewAgent(&client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessageParamUnion{}

	fmt.Println("Chat with LLM (use 'ctrl-c' to quit)")

	readUserInput := true
	for {
		if readUserInput {
			fmt.Print("\u001b[94mYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}
			conversation = append(conversation, openai.UserMessage(userInput))
		}

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		if len(message.Choices) == 0 {
			return fmt.Errorf("model returned no choices")
		}

		msg := message.Choices[0].Message

		conversation = append(conversation, msg.ToParam())

		if msg.Content != "" {
			fmt.Printf("\u001b[93mAssistant\u001b[0m: %s\n", msg.Content)
		}

		toolResults := []openai.ChatCompletionMessageParamUnion{}

		for _, toolCall := range msg.ToolCalls {
			if toolCall.Function.Name == "" {
				continue
			}

			result := a.executeTool(
				toolCall.ID,
				toolCall.Function.Name,
				json.RawMessage(toolCall.Function.Arguments),
			)
			toolResults = append(toolResults, result)
		}

		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}

		readUserInput = false
		conversation = append(conversation, toolResults...)
	}

	return nil
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) openai.ChatCompletionMessageParamUnion {
	var toolDef ToolDefinition
	var found bool
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}

	if !found {
		return openai.ToolMessage("tool not found", id)
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%s)\n", name, input)

	resp, err := toolDef.Function(input)
	if err != nil {
		return openai.ToolMessage("tool error "+err.Error(), id)
	}

	return openai.ToolMessage(resp, id)
}

func (a *Agent) runInference(
	ctx context.Context,
	conversation []openai.ChatCompletionMessageParamUnion,
) (*openai.ChatCompletion, error) {
	tools := []openai.ChatCompletionToolUnionParam{}

	for _, tool := range a.tools {
		tools = append(tools, openai.ChatCompletionFunctionTool(
			openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  openai.FunctionParameters(tool.Parameters),
			},
		))
	}

	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:     "qwen/qwen3.6-plus",
		Messages:  conversation,
		Tools:     tools,
		MaxTokens: openai.Int(1024),
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func NewAgent(
	client *openai.Client,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, list files in the current directory.",
	Parameters:  GenerateSchema[ListFilesInput](),
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to the current directory if not provided."`
}

func ListFiles(input json.RawMessage) (string, error) {
	var args ListFilesInput
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}

	dir := "."
	if args.Path != "" {
		if filepath.IsAbs(args.Path) {
			return "", fmt.Errorf("absolute paths are not allowed")
		}

		clean := filepath.Clean(args.Path)
		if clean == ".." || strings.HasPrefix(clean, "../") {
			return "", fmt.Errorf("path escapes working directory")
		}
		dir = clean
	}

	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath == "." {
			return nil
		}

		if info.IsDir() {
			files = append(files, relPath+"/")
		} else {
			files = append(files, relPath)
		}

		return nil
	})
	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	Parameters:  GenerateSchema[ReadFileInput](),
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

func ReadFile(input json.RawMessage) (string, error) {
	var readFileInput ReadFileInput
	if err := json.Unmarshal(input, &readFileInput); err != nil {
		return "", err
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}

	return string(content), nil
}

var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	Parameters: GenerateSchema[EditFileInput](),
	Function:   EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema_description:"Text to search for - must match exactly and must only have one exact match"`
	NewStr string `json:"new_str" jsonschema_description:"Text to replace old_str with"`
}

func EditFile(input json.RawMessage) (string, error) {
	var editFileInput EditFileInput
	if err := json.Unmarshal(input, &editFileInput); err != nil {
		return "", err
	}

	if editFileInput.Path == "" {
		return "", fmt.Errorf("path is required")
	}
	if editFileInput.OldStr == editFileInput.NewStr {
		return "", fmt.Errorf("old_str and new_str must be different")
	}

	content, err := os.ReadFile(editFileInput.Path)
	if err != nil {
		if os.IsNotExist(err) && editFileInput.OldStr == "" {
			return createNewFile(editFileInput.Path, editFileInput.NewStr)
		}
		return "", err
	}

	oldContent := string(content)

	if editFileInput.OldStr == "" {
		newContent := editFileInput.NewStr
		if err := os.WriteFile(editFileInput.Path, []byte(newContent), 0644); err != nil {
			return "", err
		}
		return "OK", nil
	}

	matchCount := strings.Count(oldContent, editFileInput.OldStr)
	if matchCount == 0 {
		return "", fmt.Errorf("old_str not found in file")
	}
	if matchCount > 1 {
		return "", fmt.Errorf("old_str matches more than once in file")
	}

	newContent := strings.Replace(oldContent, editFileInput.OldStr, editFileInput.NewStr, 1)

	if err := os.WriteFile(editFileInput.Path, []byte(newContent), 0644); err != nil {
		return "", err
	}

	return "OK", nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := filepath.Dir(filePath)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("successfully created file %s", filePath), nil
}

func GenerateSchema[T any]() JSONSchema {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var v T
	schema := reflector.Reflect(v)

	result := JSONSchema{
		"type":                 "object",
		"additionalProperties": false,
	}

	if schema.Properties != nil {
		result["properties"] = schema.Properties
	}

	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	if schema.Description != "" {
		result["description"] = schema.Description
	}

	return result
}

type JSONSchema map[string]any

type ToolDefinition struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  JSONSchema `json:"parameters"`
	Function    func(input json.RawMessage) (string, error)
}
