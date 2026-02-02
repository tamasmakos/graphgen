import os
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_google_community import GoogleTranslateTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('translator')

# Set up Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"

def translate_text(text, translator, source_lang=None):
    """Translate text using Google Translate"""
    try:
        document = Document(page_content=text)
        documents = [document]
        
        # Specify source language if provided, otherwise let Google detect it
        kwargs = {"target_language_code": "en"}
        if source_lang:
            kwargs["source_language_code"] = source_lang
            
        translated_documents = translator.transform_documents(documents, **kwargs)
        
        if translated_documents and len(translated_documents) > 0:
            return translated_documents[0].page_content
        else:
            logger.error(f"Translation failed for: {text[:50]}...")
            return text
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return text

def process_file(file_path, translator, output_dir):
    """Process a single file, translating non-English content"""
    logger.info(f"Processing file: {file_path}")
    
    # Create output file path
    output_file = output_dir / file_path.name
    
    translated_lines = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                translated_lines.append("")
                continue
            
            # Extract language code, speaker info, and speech
            match = re.match(r'^\[([A-Z]{2})\] (.*?) said from the (.*?) that (.*)', line)
            
            if match:
                lang_code, speaker, political_group, speech = match.groups()
                
                # Translate the speech part while preserving speaker info
                source_for_google = None if lang_code.upper() == 'EN' else lang_code.lower()
                translated_speech = translate_text(speech, translator, source_lang=source_for_google)
                
                # Construct new line with [EN] prefix
                new_line = f"[EN] {speaker} said from the {political_group} that {translated_speech}"
                translated_lines.append(new_line)
            else:
                # If line doesn't match pattern, keep it as is
                translated_lines.append(line)
        
        # Write translated content to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in translated_lines:
                f.write(f"{line}\n")
        
        logger.info(f"Translation completed for: {file_path}, saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")

def main():
    # Initialize Google Translate transformer
    try:
        project_id = "bible-chat-58a44"  # Replace with your project ID
        translator = GoogleTranslateTransformer(project_id=project_id)
        logger.info("Initialized Google Translate transformer")
    except Exception as e:
        logger.error(f"Failed to initialize translator: {str(e)}")
        return
    
    # Set up directories
    input_dir = Path('/workspaces/kg/input/txt/proceedings')
    output_dir = Path('/workspaces/kg/input/txt/translated')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all text files from input directory
    txt_files = list(input_dir.glob('*.txt'))
    
    if not txt_files:
        logger.error(f"No text files found in {input_dir}")
        return
    
    logger.info(f"Found {len(txt_files)} text files to process")
    
    # Process each file
    for file_path in txt_files:
        process_file(file_path, translator, output_dir)
    
    logger.info("Translation process completed")

if __name__ == "__main__":
    main()
