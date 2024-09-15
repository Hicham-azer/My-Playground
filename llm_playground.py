import streamlit as st
import re
import yaml
import os
import pandas as pd
import io
import json  # Import json module

# Import required libraries for providers
from openai import OpenAI
import cohere

# Import load_dotenv to read .env file
from dotenv import load_dotenv

# Set page configuration to use wide layout
st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()

# Set API Keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
# For Hugging Face, if using the API (optional)
# hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define LLM Provider Classes
class LLMProvider:
    def __init__(self, name):
        self.name = name

    def generate_response(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

# Placeholder for OpenAIProvider (since actual implementation may vary)
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__('OpenAI')
        self.api_key = api_key

    def generate_response(self, messages, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content

class CohereProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__('Cohere')
        self.client = cohere.Client(api_key)

    def generate_response(self, prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
        response = self.client.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=[]
        )
        return response.generations[0].text.strip()

# Function to get provider instance
def get_provider_instance(config):
    if config['provider'] == 'OpenAI':
        return OpenAIProvider(openai_api_key)
    elif config['provider'] == 'Cohere':
        return CohereProvider(cohere_api_key)
    else:
        raise ValueError(f"Unsupported provider: {config['provider']}")

# Load providers and models from config.yaml
def load_providers_and_models(config_file='config.yaml'):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        providers = config.get('providers', {})
    else:
        # Default providers and models if config.yaml doesn't exist
        providers = {
            'OpenAI': {
                'models': ['gpt-3.5-turbo', 'gpt-4']
            },
            'Cohere': {
                'models': ['command', 'xlarge']
            },
            'HuggingFace': {
                'models': ['gpt2', 'distilgpt2']
            }
        }
    return providers

providers_dict = load_providers_and_models()
providers_list = list(providers_dict.keys())

# Define the session state file
SESSION_STATE_FILE = 'session_state.json'

# Load the session state from the file
def load_session_state():
    if os.path.exists(SESSION_STATE_FILE):
        with open(SESSION_STATE_FILE, 'r') as f:
            saved_session_state = json.load(f)
        for key, value in saved_session_state.items():
            st.session_state[key] = value

load_session_state()

# Initialize session state for configurations
if 'configs' not in st.session_state:
    st.session_state.configs = [{
        'name': 'Config 1',
        'provider': 'OpenAI',
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 150,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }]

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello, I need help with {topic}.'}
    ]

# Define save_session_state() function
def save_session_state():
    state_to_save = {
        'configs': st.session_state.get('configs', []),
        'messages': st.session_state.get('messages', [])
    }
    with open(SESSION_STATE_FILE, 'w') as f:
        json.dump(state_to_save, f)

# Application Title
st.title("MY Playground")

# Function to display configurations
def display_configs():
    for i, config in enumerate(st.session_state.configs):
        expander = st.sidebar.expander(f"{config['name']}", expanded=True)
        with expander:
            # Name and Delete Button
            col1, col2 = st.columns([1, 1])
            with col1:
                name = st.text_input(f"Config Name", value=config['name'], key=f"config_name_{i}")
            with col2:
                if st.button("Delete Config", key=f"delete_config_{i}"):
                    del st.session_state.configs[i]
                    save_session_state()
                    st.rerun()
            # Provider Selection
            previous_provider = config.get('provider', 'OpenAI')
            provider = st.selectbox(
                "Provider",
                providers_list,
                index=providers_list.index(previous_provider),
                key=f"provider_{i}"
            )
            # If provider changed, reset the model
            if provider != previous_provider:
                model = providers_dict[provider]['models'][0]
            else:
                model = config.get('model', providers_dict[provider]['models'][0])
                if model not in providers_dict[provider]['models']:
                    model = providers_dict[provider]['models'][0]
            # Model Selection
            provider_models = providers_dict[provider]['models']
            model = st.selectbox(
                "Model",
                provider_models,
                index=provider_models.index(model),
                key=f"model_{i}"
            )
            # Temperature
            temperature = st.slider(
                "Temperature", 0.0, 1.0,
                value=config.get('temperature', 0.7),
                key=f"temperature_{i}"
            )
            # Max Tokens
            max_tokens = st.slider(
                "Max Tokens", 50, 1024,
                value=config.get('max_tokens', 150),
                key=f"max_tokens_{i}"
            )
            # Top_p
            top_p = st.slider(
                "Top_p (Nucleus Sampling)", 0.0, 1.0,
                value=config.get('top_p', 1.0),
                key=f"top_p_{i}"
            )
            # Frequency Penalty
            frequency_penalty = st.slider(
                "Frequency Penalty", -2.0, 2.0,
                value=config.get('frequency_penalty', 0.0),
                key=f"freq_penalty_{i}"
            )
            # Presence Penalty
            presence_penalty = st.slider(
                "Presence Penalty", -2.0, 2.0,
                value=config.get('presence_penalty', 0.0),
                key=f"pres_penalty_{i}"
            )
            # Update the config in session state
            st.session_state.configs[i] = {
                'name': name,
                'provider': provider,
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'frequency_penalty': frequency_penalty,
                'presence_penalty': presence_penalty
            }
            # Save the session state
            save_session_state()

# Display the configurations
st.sidebar.header("LLM Configurations")
display_configs()

# Button to add a new configuration
if st.sidebar.button("Add Configuration"):
    st.session_state.configs.append({
        'name': f'Config {len(st.session_state.configs)+1}',
        'provider': 'OpenAI',
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 150,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    })
    save_session_state()
    st.rerun()

st.header("Conversation Messages")

# Function to display message inputs
def display_messages():
    for i, message in enumerate(st.session_state.messages):
        expander = st.expander(f"Message {i+1} ({message['role']})", expanded=True)
        with expander:
            col1, col2 = st.columns([1, 1])
            with col1:
                role = st.selectbox(
                    "Role",
                    ['system', 'user', 'assistant'],
                    index=['system', 'user', 'assistant'].index(message['role']),
                    key=f"role_{i}"
                )
            with col2:
                if st.button("Delete Message", key=f"delete_{i}"):
                    del st.session_state.messages[i]
                    save_session_state()
                    st.rerun()
            content = st.text_area(
                "Content",
                value=message['content'],
                key=f"content_{i}"
            )
            # Update the message in session state
            st.session_state.messages[i] = {
                'role': role,
                'content': content
            }
            # Save the session state
            save_session_state()

# Display the messages
display_messages()

# Button to add a new message
if st.button("Add Message"):
    st.session_state.messages.append({'role': 'user', 'content': ''})
    save_session_state()
    st.rerun()

# Function to extract placeholders
def extract_placeholders(text):
    return re.findall(r"\{(.*?)\}", text)

# Extract placeholders from all messages
all_placeholders = set()
for message in st.session_state.messages:
    placeholders = extract_placeholders(message['content'])
    all_placeholders.update(placeholders)

# Add a file uploader for CSV and XLSX files
st.header("Upload CSV or XLSX for Batch Processing")
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    if file_extension == '.csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
        st.stop()
    st.write("Uploaded Data:")
    st.dataframe(df)
    
    # Get placeholders from messages
    required_placeholders = all_placeholders
    csv_columns = set(df.columns)
    
    # Check if all placeholders are present in the data
    if not required_placeholders.issubset(csv_columns):
        missing_placeholders = required_placeholders - csv_columns
        st.error(f"The following placeholders are missing in the uploaded file: {', '.join(missing_placeholders)}")
        st.stop()
    placeholder_values = None
else:
    # Dynamic Inputs for Placeholders
    st.header("Placeholder Values")
    placeholder_values = {}
    for placeholder in all_placeholders:
        value = st.text_input(f"Value for '{placeholder}':", key=f"placeholder_{placeholder}")
        placeholder_values[placeholder] = value

# Generate Responses Button
if st.button("Generate Responses"):
    if uploaded_file is not None:
        with st.spinner("Generating responses for each row in the uploaded file..."):
            try:
                all_responses = []  # To store responses for each row
                for idx, row in df.iterrows():
                    placeholder_values = row.to_dict()
                    # Replace placeholders in messages
                    final_messages = []
                    for message in st.session_state.messages:
                        content = message['content']
                        for placeholder, value in placeholder_values.items():
                            content = content.replace(f"{{{placeholder}}}", str(value))
                        final_messages.append({'role': message['role'], 'content': content})
                    # Generate responses for each configuration
                    row_responses = []
                    for config in st.session_state.configs:
                        provider = get_provider_instance(config)
                        if config['provider'] == 'OpenAI':
                            messages = final_messages
                            response_text = provider.generate_response(
                                messages=messages,
                                model=config['model'],
                                temperature=config['temperature'],
                                max_tokens=config['max_tokens'],
                                top_p=config['top_p'],
                                frequency_penalty=config['frequency_penalty'],
                                presence_penalty=config['presence_penalty']
                            )
                        else:
                            # For providers that require a prompt
                            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in final_messages])
                            response_text = provider.generate_response(
                                prompt=prompt,
                                model=config['model'],
                                max_tokens=config['max_tokens'],
                                temperature=config['temperature'],
                                top_p=config['top_p'],
                                frequency_penalty=config['frequency_penalty'],
                                presence_penalty=config['presence_penalty']
                            )
                        row_responses.append({
                            'config_name': config['name'],
                            'provider': config['provider'],
                            'model': config['model'],
                            'temperature': config['temperature'],
                            'max_tokens': config['max_tokens'],
                            'top_p': config['top_p'],
                            'frequency_penalty': config['frequency_penalty'],
                            'presence_penalty': config['presence_penalty'],
                            'response': response_text
                        })
                    # Store responses along with the input data
                    all_responses.append({
                        'input_data': placeholder_values,
                        'responses': row_responses
                    })
                st.success("Responses Generated!")
                # Store all_responses in session state
                st.session_state['all_responses'] = all_responses
                
                # Display the responses
                st.header("Responses for Each Input Row")
                for idx, item in enumerate(all_responses):
                    st.subheader(f"Input Row {idx+1}")
                    st.write("**Input Data:**")
                    st.write(item['input_data'])
                    for res in item['responses']:
                        st.write(f"### {res['config_name']} - {res['provider']} - {res['model']}")
                        # Configuration details in an expander
                        with st.expander("Configuration Details"):
                            st.write(f"**Temperature:** {res['temperature']}")
                            st.write(f"**Max Tokens:** {res['max_tokens']}")
                            st.write(f"**Top_p:** {res['top_p']}")
                            st.write(f"**Frequency Penalty:** {res['frequency_penalty']}")
                            st.write(f"**Presence Penalty:** {res['presence_penalty']}")
                        st.write("**Response:**")
                        st.code(res['response'], language='')
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        # Single input processing
        with st.spinner("Generating responses..."):
            try:
                # Replace placeholders in messages
                final_messages = []
                for message in st.session_state.messages:
                    content = message['content']
                    for placeholder, value in placeholder_values.items():
                        content = content.replace(f"{{{placeholder}}}", value)
                    final_messages.append({'role': message['role'], 'content': content})
                # Generate responses for each configuration
                responses = []
                for config in st.session_state.configs:
                    provider = get_provider_instance(config)
                    if config['provider'] == 'OpenAI':
                        messages = final_messages
                        response_text = provider.generate_response(
                            messages=messages,
                            model=config['model'],
                            temperature=config['temperature'],
                            max_tokens=config['max_tokens'],
                            top_p=config['top_p'],
                            frequency_penalty=config['frequency_penalty'],
                            presence_penalty=config['presence_penalty']
                        )
                    else:
                        # For providers that require a prompt
                        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in final_messages])
                        response_text = provider.generate_response(
                            prompt=prompt,
                            model=config['model'],
                            max_tokens=config['max_tokens'],
                            temperature=config['temperature'],
                            top_p=config['top_p'],
                            frequency_penalty=config['frequency_penalty'],
                            presence_penalty=config['presence_penalty']
                        )
                    responses.append({
                        'config_name': config['name'],
                        'provider': config['provider'],
                        'model': config['model'],
                        'temperature': config['temperature'],
                        'max_tokens': config['max_tokens'],
                        'top_p': config['top_p'],
                        'frequency_penalty': config['frequency_penalty'],
                        'presence_penalty': config['presence_penalty'],
                        'response': response_text
                    })
                st.success("Responses Generated!")
                # Store responses in session state
                st.session_state['responses'] = responses
                # Display responses
                st.header("Responses Comparison")
                # Use tabs to display each response
                tabs = st.tabs([f"{res['config_name']}" for res in responses])
                for idx, tab in enumerate(tabs):
                    with tab:
                        res = responses[idx]
                        # Configuration details in an expander
                        with st.expander("Configuration Details"):
                            st.write(f"**Provider:** {res['provider']}")
                            st.write(f"**Model:** {res['model']}")
                            st.write(f"**Temperature:** {res['temperature']}")
                            st.write(f"**Max Tokens:** {res['max_tokens']}")
                            st.write(f"**Top_p:** {res['top_p']}")
                            st.write(f"**Frequency Penalty:** {res['frequency_penalty']}")
                            st.write(f"**Presence Penalty:** {res['presence_penalty']}")
                        st.write("**Response:**")
                        st.code(res['response'], language='')
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Option to export results to XLSX
if st.button("Export Results to XLSX"):
    if uploaded_file is not None and 'all_responses' in st.session_state:
        all_responses = st.session_state['all_responses']
        # Prepare data for export
        export_data = []
        for item in all_responses:
            input_data = item['input_data']
            for res in item['responses']:
                row = input_data.copy()
                row.update({
                    'config_name': res['config_name'],
                    'provider': res['provider'],
                    'model': res['model'],
                    'temperature': res['temperature'],
                    'max_tokens': res['max_tokens'],
                    'top_p': res['top_p'],
                    'frequency_penalty': res['frequency_penalty'],
                    'presence_penalty': res['presence_penalty'],
                    'response': res['response']
                })
                export_data.append(row)
        export_df = pd.DataFrame(export_data)
        # Convert DataFrame to Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Responses')
        processed_data = output.getvalue()
        # Provide download button
        st.download_button(
            label="Download Results as XLSX",
            data=processed_data,
            file_name='responses.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    elif uploaded_file is None and 'responses' in st.session_state:
        responses = st.session_state['responses']
        # Prepare data for export
        export_data = []
        for res in responses:
            row = {
                'config_name': res['config_name'],
                'provider': res['provider'],
                'model': res['model'],
                'temperature': res['temperature'],
                'max_tokens': res['max_tokens'],
                'top_p': res['top_p'],
                'frequency_penalty': res['frequency_penalty'],
                'presence_penalty': res['presence_penalty'],
                'response': res['response']
            }
            export_data.append(row)
        export_df = pd.DataFrame(export_data)
        # Convert DataFrame to Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Responses')
        processed_data = output.getvalue()
        # Provide download button
        st.download_button(
            label="Download Results as XLSX",
            data=processed_data,
            file_name='responses.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    else:
        st.error("No responses to export. Please generate responses first.")

# Add a button in the sidebar to clear session data
if st.sidebar.button("Clear Session Data"):
    st.session_state['confirm_clear_session'] = True

# Confirmation for clearing session data
if st.session_state.get('confirm_clear_session', False):
    st.sidebar.warning("Are you sure you want to clear all session data? This action cannot be undone.")
    confirm_clicked = st.sidebar.button("Yes, clear data", key='confirm_clear_yes')
    cancel_clicked = st.sidebar.button("Cancel", key='confirm_clear_cancel')
    if confirm_clicked:
        if os.path.exists(SESSION_STATE_FILE):
            os.remove(SESSION_STATE_FILE)
        st.session_state.clear()
        st.rerun()
    elif cancel_clicked:
        st.session_state['confirm_clear_session'] = False
