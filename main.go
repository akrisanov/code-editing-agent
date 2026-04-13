package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

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

	agent := NewAgent(&client, getUserMessage)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessageParamUnion{}

	fmt.Println("Chat with LLM (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		conversation = append(conversation, openai.UserMessage(userInput))

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		if len(message.Choices) == 0 {
			return fmt.Errorf("model returned no choices")
		}

		assistantText := message.Choices[0].Message.Content
		conversation = append(conversation, openai.AssistantMessage(assistantText))

		fmt.Printf("\u001b[93mAssistant\u001b[0m: %s\n", assistantText)
	}

	return nil
}

func (a *Agent) runInference(
	ctx context.Context,
	conversation []openai.ChatCompletionMessageParamUnion,
) (*openai.ChatCompletion, error) {
	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:     "qwen/qwen3.6-plus",
		Messages:  conversation,
		MaxTokens: openai.Int(1024),
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func NewAgent(client *openai.Client, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
}
