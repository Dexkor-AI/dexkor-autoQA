import logging
from groq import Groq
import time
from datetime import datetime
import json
import pandas as pd
import time
import os
from dotenv import load_dotenv

# load the env file
load_dotenv()

class ChatAgentEvaluator:
    def __init__(self):

        self.groq_client = Groq(api_key=os.getenv("GROQ_API"))
        self.model_id = 'llama3-70b-8192'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def call_groq_inference(self, input_text: str):
        """Function to call Groq for LLM inference."""
        if not self.model_id or not input_text:
            raise ValueError("Both model_id and input_text must be provided")

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "you are a experienced helpful QA assistant."
                    },
                    {
                        "role": "user",
                        "content": input_text,
                    }
                ],
                model=self.model_id,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

    def analyze_customer_sentiment_and_responses(self, transcript):
        """Analyzes the transcript to extract sentiment and response relevance."""
        # Format the prompt to extract the required parameters (Positive Sentiment, Total Customer Messages, etc.)
        prompt = f"""
            You are tasked with evaluating a conversation between a customer and an agent. Your job is to assess the agent's performance across 
            various categories and provide a score for each sub-parameter based on how well the agent followed best practices, responded to the 
            customer, and handled the issue at hand. Please use the scoring system outlined below and ensure that your evaluation is fair, 
            consistent, and objective. After completing the evaluation, return the results as a structured JSON object containing scores for each 
            of the parameters. You have to provide score out of the maximum given for each parameter.

            Evaluation Parameters:
            1. Opening (max 15 points)
                - First response given within defined timeframe (10 points): Did the agent respond within 1 minute after being assigned to the customer?
                - Opening statement (Pre-defined) (5 points): Did the agent use the predefined opening statement template correctly?
            
            2. Communication Skills (20 points)
                - Apology / Empathy when required (10 points): Did the agent demonstrate empathy or apologize when necessary (e.g., when the customer faced an issue)?
                - Timely response (5 points): Did the agent respond within 5-7 minutes to avoid dead air during the conversation?
                - Correct sentence formation (5 points): Was the agent’s language clear, free from spelling errors, and professionally structured? Did they use simple words and follow language guidelines?
            
            3. Chat Handling (25 points)
                - Asked Probing questions (5 points): Did the agent ask relevant probing questions (e.g., "What happened exactly?") to understand the issue better?
                - Chat Disposition (5 points): Did the agent use the correct disposition for the scenario, or was it misused?
                - Internal Notes (5 points): Were the agent’s internal notes clear and reflective of the conversation (e.g., actions taken, customer concerns)?
                - Notes for inter-department assignment (5 points): Did the agent provide appropriate notes if the case was handed over to another department (e.g., L2, L3)?
                - Logs URL to be added for reference (5 points): Did the agent add the necessary URL to logs, if required, for future reference?
            
            4. Product Knowledge (40 points)
                - Proactively sharing product knowledge / education for future reference (10 points): Did the agent proactively share helpful product knowledge or resources to prevent future issues?
                - Proper information to Tech / TL / Other internal departments while assigning the case (10 points): Did the agent correctly convey information to other departments (Tech, TL) when needed?
                - Possible Resolution in case of no response based on available information (10 points): Did the agent provide a potential resolution when the customer did not respond, using the available information?
                - Screenshot / knowledgebase / steps / reference (10 points): Did the agent include helpful screenshots, knowledge base links, or step-by-step instructions to assist in resolving the issue?
            
            5. Fatal (Zero Tolerance)
                - Provided incorrect / incomplete information (Zero tolerance): Was any incorrect or incomplete information provided to the customer? If so, the score is zero unless the agent rectified it within the same conversation.
                - No Changes / action to be done on customer panel without permission and confirmation (Zero tolerance): Did the agent take any action on the customer’s panel without explicit permission? If so, the score is zero.
                - Issue avoidance (Zero tolerance): Did the agent proactively close the case without resolving all the customer’s queries? If so, the score is zero.
                - Use of unprofessional language (Zero tolerance): Did the agent use any unprofessional language, derogatory words, or engage in inappropriate conversations? If so, the score is zero.
            
            6. Sentiment:
                - Overall Sentiment: Positive, Negative, Neutral
            
            7. Summary:
                - Provide a brief summary of conversation.
                - Include all the key points discussed during the conversation.
                - Include any action items or next steps.
                - Include any additional information that may be relevant.
                
            Task Steps:
            1. Evaluate the Transcript:
                - Review the entire conversation between the customer and the agent. For each parameter, assign a score based on the details of the interaction. Ensure the score reflects the agent's adherence to the best practices outlined in the above categories.

            2. Return a JSON Object:
                - Once the evaluation is complete, please return the results in the following JSON format:

            # Return only JSON object, do not return any additional text in the response, like (here is the json response..., Here is the evaluation in JSON format, etc)
            {{
                "Opening": {{
                    "First response given within defined timeframe": <score>,
                    "Opening statement (Pre-defined)": <score>,
                    "Reasoning": <reasoning_for_score>
                }},
                "Communication skills": {{
                    "Apology / Empathy when required": <score>, 
                    "Timely response": <score>,
                    "Correct sentence formation": <score>,
                    "Reasoning": <reasoning_for_score>
                }},
                "Chat Handling": {{
                    "Asked Probing questions (WH Questions)": <score>,
                    "Chat Disposition": <score>,
                    "Internal Notes": <score>,
                    "Notes for inter department assignment": <score>,
                    "Logs URL to be added for reference": <score>,
                    "Reasoning": <reasoning_for_score>

                }},
                "Product Knowledge": {{
                    "Proactively sharing product knowledge/education for future reference": <score>,
                    "Proper information to Tech / TL / Other internal departments while assigning the case": <score>,
                    "Possible Resolution in case of no response basis available information": <score>,
                    "Screenshot / knowledgebase / steps / reference": <score>,
                    "Reasoning": <reasoning_for_score>
                }},
                "Fatal": <yes or no>,
                "Sentiment": <overall_sentiment>,
                "Summary": <summary_of_transcript>
            }}

            Transcript: 
            {transcript}
        """
        try:
            with open("llm_input.txt", "w") as file:  # Change mode to 'a' for append
                logging.debug(f"Appending prompt to file for LLM inference. {prompt}")
                file.write(prompt)
            result = self.call_groq_inference(prompt)  

            if result:
                with open("llm_output.txt", "a") as file:
                    file.write(result + "\n\n"  + "*" * 100 + "\n\n")
                return self.parse_llm_output(result)
            else:
                logging.warning("No result returned from LLM.")
                return None
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            return None

    def parse_llm_output(self, output):
        """Extract key values from the LLM response."""

        if "```" in output:
            output = output.split("```")[1]
        
        if output[-1] != '}':
            if output[-1] == '"':
                output += "}"
            else:
                output += '"}' 
        try:
            response_data = json.loads(output)  # Use json.loads for safer parsing
            return response_data
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing LLM output: {str(e)}")
            return None

    def calculate_score(self, llm_response):
        """Calculate the score based on the LLM response."""
        try:
            # Extract the scores from the LLM response
            # 1. Opening (max score: 15 points)
            opening_score = llm_response.get('Opening', {})
            opening_score_value = sum({key: value for key, value in opening_score.items() if key != "Reasoning"}.values())

            # 2. Communication Skills (max score: 20 points)
            communication_score = llm_response.get('Communication skills', {})
            communication_score_value = sum({key: value for key, value in communication_score.items() if key != "Reasoning"}.values())

            # 3. Chat Handling (max score: 25 points)
            chat_handling_score = llm_response.get('Chat Handling', {})
            chat_handling_score_value = sum({key: value for key, value in chat_handling_score.items() if key != "Reasoning"}.values())

            # 4. Product Knowledge (max score: 40 points)
            product_knowledge_score = llm_response.get('Product Knowledge', {})
            product_knowledge_score_value = sum({key: value for key, value in product_knowledge_score.items() if key != "Reasoning"}.values())

            # 5. Fatal (Zero Tolerance)
            fatal_error_score = llm_response.get('Fatal', "no")
            if fatal_error_score == "yes":
                fatal_error_score = 0
            else:
                fatal_error_score = 1
            
            # Calculate the total score
            total_score = (opening_score_value + communication_score_value + chat_handling_score_value + product_knowledge_score_value) * fatal_error_score

            overall_scores = {
                "Opening Score": opening_score_value,
                "Communication Skills Score": communication_score_value,
                "Chat Handling Score": chat_handling_score_value,
                "Product Knowledge Score": product_knowledge_score_value,
                "Fatal Error": "No" if fatal_error_score == 1 else "Yes",
                "Total Score": total_score,
            }
            return overall_scores, llm_response.get('Summary', ''), llm_response.get('Sentiment', '')
        except Exception as e:
            logging.error(f"Error calculating score: {e}")
            return None, '', ''

    def evaluate_conversation(self, transcript):
        # Analyze the conversation using the LLM
        try:
            analysis_results = self.analyze_customer_sentiment_and_responses(transcript)

            if analysis_results:
                total_scores, summary, sentiment = self.calculate_score(analysis_results)
                return total_scores, summary, sentiment, analysis_results
            else:
                logging.error("Failed to analyze the conversation.")
                return None
        except Exception as e:
            logging.error(f"Error evaluating conversation: {e}")
            return None

if __name__ == "__main__":
    try:
        with open("transcript.txt", "r") as file:   
            transcripts = file.read()
        
        transcripts = transcripts.split(f"{'*' * 100}")[:-1]
        
        dataframes = []
        for transcript in transcripts[:1]:
            main_transcript = transcript.split("Transcript:")[-1].strip()
            code = transcript.split("Code:")[1].split("Transcript:")[0].strip()


            print(f"Processing transcript for code: {code}")
            evaluator = ChatAgentEvaluator()

            total_scores, summary, sentiment, llm_response = evaluator.evaluate_conversation(transcript)
            if total_scores:
                opening_score = total_scores['Opening Score']
                communication_score = total_scores['Communication Skills Score']
                chat_handling_score = total_scores['Chat Handling Score']
                product_knowledge_score = total_scores['Product Knowledge Score']
                fatal_error = total_scores['Fatal Error']
                total_score = total_scores['Total Score']
            else:
                logging.warning(f"Skipping row {code} due to evaluation failure.")
            
            dataframes.append({
                "Code": code,
                "Opening Score": opening_score,
                "Communication Skills Score": communication_score,
                "Chat Handling Score": chat_handling_score,
                "Product Knowledge Score": product_knowledge_score,
                "Fatal Error": fatal_error,
                "Total Score": total_score,
                "Summary": summary,
                "Sentiment": sentiment,
                "llm_response": json.dumps(llm_response)  # Convert JSON to string
            })
            print("sleeping for 20 seconds....")
            time.sleep(20)

        df = pd.DataFrame(dataframes)
        df.to_csv("evaluation_results.csv", index=False)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise e
