import streamlit as st
import requests
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import os

load_dotenv()

def get_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content (customize based on website structure)
        main_content = soup.find('main') or soup.find('article') or soup.body
        if main_content:
            text = ' '.join(main_content.stripped_strings)
            return text[:3000]  # Return first 3000 characters to avoid token limits
        return "Could not extract main content from the webpage"
    except Exception as e:
        return f"Error fetching URL content: {str(e)}"

def getLLMResponse(query, age_option, tasktype_option, social_media, content_style, num_outputs):
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    groq_api_key = os.getenv('GROQ_API_TOKEN')

    examples = []
    if age_option == "Kid":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a magical device that fits in your pocket!"},
            {"query": "Why is the sky blue?", "answer": "The sky wears its favorite blue color every day!"}
        ]
    elif age_option == "Adult":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a portable communication device."},
            {"query": "Why is the sky blue?", "answer": "Due to Rayleigh scattering of sunlight."}
        ]
    elif age_option == "Senior Citizen":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile phone is a device for calls, messages, and internet."},
            {"query": "Why is the sky blue?", "answer": "Atmospheric scattering makes the sky appear blue."}
        ]

    example_template = """Question: {query}\nResponse: {answer}"""
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    prefix = f"""You are a {age_option} creating {content_style} content for {social_media}. 
    Task: {tasktype_option}. Examples:"""
    suffix = "\nQuestion: {template_userInput}\nResponse: "

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=200
    )

    new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector, 
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_userInput"],
        example_separator="\n"
    )

    prompt_data = new_prompt_template.format(template_userInput=query)

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt_data}],
        "n": num_outputs
    }

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(groq_url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return [choice['message']['content'] for choice in result['choices']]
    return f"API Error: {response.status_code} - {response.text}"

# Streamlit UI
st.set_page_config(page_title="Content Generator", layout='centered')
st.header("AI Content Generator")

form_input = st.text_area('Type the topic or URL', height=150)
tasktype_option = st.selectbox('Task type:', ['Write a sales copy', 'Create a tweet', 'Write a product description', 'Explain a concept'])
age_option = st.selectbox('Target age group:', ['Kid', 'Adult', 'Senior Citizen'])
social_media = st.selectbox('Platform:', ['Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'TikTok'])
content_style = st.selectbox('Style:', ['Persuasive', 'Humorous', 'Inspirational', 'Serious'])
num_outputs = st.slider('Number of outputs:', 1, 5, 1)

if st.button("Generate Content"):
    with st.spinner('Processing...'):
        # Check if input is URL
        if form_input.startswith(('http://', 'https://')):
            with st.status("Fetching URL content..."):
                url_content = get_url_content(form_input)
                if url_content.startswith("Error"):
                    st.error(url_content)
                    st.stop()
                else:
                    st.write("URL content fetched successfully!")
                    query = f"Based on this content: {url_content[:2000]}..."  # Truncate for prompt
        else:
            query = form_input

        responses = getLLMResponse(
            query, 
            age_option, 
            tasktype_option,
            social_media,
            content_style,
            num_outputs
        )
        
        if isinstance(responses, list):
            st.success("Generated Content:")
            for i, response in enumerate(responses):
                st.subheader(f"Version {i+1}:")
                st.write(response)
        else:
            st.error(responses)