from flask import Flask, request, jsonify
import json
from lambda_function import lambda_handler
import logging

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock context for local testing
class MockContext:
    def __init__(self):
        self.function_name = "test_lambda_function"
        self.memory_limit_in_mb = 128
        self.aws_request_id = "local-test-request-id"
        self.log_group_name = "/aws/lambda/test_lambda_function"
        self.log_stream_name = "test-stream"
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test_lambda_function"
    
    def get_remaining_time_in_millis(self):
        return 10000  # example value in milliseconds

@app.route('/test-lambda', methods=['POST'])
def test_lambda():
    try:
        # Construct the event from the Flask request
        event = {
            "body": json.dumps(request.json)
        }

        # Create a mock context
        context = MockContext()

        # Call the lambda_handler function
        response = lambda_handler(event, context)

        # Log the response for debugging
        logger.info(f"Lambda response: {response}")

        # Return the response from the Lambda handler
        return jsonify(json.loads(response['body'])), response['statusCode']
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500


if __name__ == '__main__':
    app.run(debug=True)
