from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the OpenAI key and initialize the client
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")
client = OpenAI(
    organization='org-IC2QjfG5rq9J5Tau63TeZAJZ',  # Replace with your organization ID
    api_key=API_KEY,
)

# Flask app setup
app = Flask(__name__)

class ScamDetector:
    def __init__(self, model="gpt-4", max_tokens=300, temperature=0.7, top_p=1.0,
                 frequency_penalty=0.0, presence_penalty=0.0, logprobs=None, user=None, n=1):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs
        self.user = user
        self.n = n

    def generate_response(self, ad_text, metadata):
        """
        Generate a response from OpenAI using the initialized client.
        :param ad_text: The text of the job advertisement.
        :param metadata: Metadata associated with the ad.
        :return: String response from the model.
        """
        try:
            # Define the message role and content
            messages = [
                {"role": "system", "content": (
                    "You are a model that detects if a job advertisement is suspicious of human trafficking. "
                    "Given the following ad text and metadata, provide the probability "
                    "of it being a scam and explain your reasoning.\n\n"
                    f"Ad Text: {ad_text}\n"
                    f"Metadata: {metadata}\n\n"
                    "Output the probability as a percentage and reasons for your decision."
                )}
            ]

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                logprobs=self.logprobs,
                user=self.user,
                n=self.n,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return "There was an error processing your request. Please try again later."

# Initialize ScamDetector
scam_detector = ScamDetector()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Parse JSON data from the request
        data = request.get_json()
        ad_text = data.get('ad_text', '').strip()
        metadata = data.get('metadata', '').strip()

        # Error handling for empty inputs
        if not ad_text:
            return jsonify({"error": "Ad text is required."})

        # Get the analysis
        result = scam_detector.generate_response(ad_text, metadata)
        return jsonify({"result": result})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
