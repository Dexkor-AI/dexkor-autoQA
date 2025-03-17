# dexkor-autoQA

This repository contains a lambda function to evaluate chat agent performance based on customer transcripts. The evaluation is performed using a large language model (LLM) via the Groq API.

## Lambda Function

The lambda function processes a customer-agent conversation transcript and evaluates the agent's performance across various parameters. The evaluation includes sentiment analysis and scoring based on predefined criteria.

### Key Components

- **ChatAgentEvaluator Class**: This class handles the evaluation process.
  - `call_groq_inference(input_text: str)`: Calls the Groq API for LLM inference.
  - `analyze_customer_sentiment_and_responses(transcript)`: Analyzes the transcript to extract sentiment and response relevance.
  - `parse_llm_output(output)`: Parses the LLM output to extract key values.
  - `calculate_score(llm_response)`: Calculates the score based on the LLM response.
  - `evaluate_conversation(transcript)`: Evaluates the conversation using the LLM.

### Lambda Handler

The `lambda_handler` function is the entry point for the AWS Lambda function. It processes the incoming event, extracts the transcript, and uses the `ChatAgentEvaluator` class to evaluate the conversation.

### Environment Variables

- `GROQ_API`: API key for accessing the Groq API.

## API Testing

### Endpoint

- **POST /evaluate**

### Request

- **Headers**:
  - `Content-Type: application/json`

- **Body**:
  ```json
  {
    "transcript": [
      {
        "timestamp": "2023-10-01T12:00:00Z",
        "user": "customer",
        "message": "Hello, I need help with my order."
      },
      {
        "timestamp": "2023-10-01T12:01:00Z",
        "user": "agent",
        "message": "Sure, I can help you with that. Can you please provide your order number?"
      }
      // ...additional transcript entries...
    ]
  }
  ```

### Response

- **Success (200)**:
  ```json
  {
    "Opening Score": 15,
    "Communication Skills Score": 20,
    "Chat Handling Score": 25,
    "Product Knowledge Score": 40,
    "Fatal Error": "No",
    "Total Score": 100,
    "Summary": "The agent was very helpful and provided all the necessary information.",
    "Sentiment": "Positive",
    "llm_response": {
      // ...detailed LLM response...
    }
  }
  ```

- **Client Error (400)**:
  ```json
  {
    "error": "Missing key: 'transcript'"
  }
  ```

- **Server Error (500)**:
  ```json
  {
    "error": "Unexpected error: <error_message>"
  }
  ```