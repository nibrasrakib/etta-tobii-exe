from openai import OpenAI

# Load your OpenAI API key from an environment variable or directly
# For security reasons, it's better to use environment variables in production
import os
def load_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")
    return api_key

def generate_summary(user_content):
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. I will provide you with clustered news from the rss_feed database collected based on a search query."
                    "Your task is to:"
                    "1. Provide a short title (2-5 words) that reflects the essence of the cluster."
                    "2. Write a concise summary (under 15 words) that captures the overall theme of the news."
                    "Format your response as follows:"
                    "1. <Title>"
                    "2. <Summary>"
                    "If there is insufficient information to summarize a cluster, return as:"
                    "1. N/A"
                    "2. N/A"
                    "Ensure the title is between 2-5 words and the summary is a single, concise sentence."
                    "Strictly follow the two-point response format and keep it relevant to the cluster."
                )
            },
            {"role": "user", "content": user_content}
        ]
    )
    return completion.choices[0].message.content


# "You are a helpful assistant. Given a search query from a user, the platform generates topic clusters. "
#                     "Each cluster contains a list of relevant publications. To enhance this system, I want to refine and expand the topic clusters for better clarity. "
#                     "I'll provide the clustered publications with their assigned topics. Please expand each assigned topic cluster into a single, concise sentence "
#                     "that effectively summarizes all the included information, making the clusters more legible and coherent. I don't want any long explanations. "
#                     "Final results should only be a single sentence overall with less than 10-15 words in total and keep it related to the assigned cluster but also capture the essense of the publications. "
#                     "Don't emphasize only one publication, try to get a holistic view of the cluster."