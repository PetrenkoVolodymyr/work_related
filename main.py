# import os
# import time
# import requests
# import json
import pandas as pd
# import openai
# from langchain.docstore.document import Document
# from langchain.llms import OpenAI
# from langchain.chains.summarize import load_summarize_chain
# from langchain.docstore.document import Document
# from langchain.llms.openai import OpenAI, OpenAIChat
# from langchain_openai import AzureChatOpenAI
# from langchain.prompts import PromptTemplate
pd.set_option('display.max_rows', 100)


# Library to manpilate docx file
from docx import Document


# Function to read docx file and return text
def read_docx(file_path):
    doc = Document(file_path) # Open docx file
    full_text = [] # Initiate list obkect
    for para in doc.paragraphs:# Read each paragrapgh until reading whole text
        full_text.append(para.text) # add paragrapgh read to the list
    return '\n'.join(full_text)

# Parsing the transcript text
def parse_transcript(transcript):
    lines = transcript.strip().split("\n") # delete the branks in the text and split to tream text by record unit.
    data = []
    i = 0
    while i < len(lines):
        if "-->" in lines[i]:  # Checks if the line contains time range
            time_line = lines[i].strip()
            speaker_line = lines[i+1].strip()
            speech_line = lines[i+2].strip()
            start_time = time_line.split(" --> ")[0].strip()
            speaker = speaker_line
            speech = speech_line
            data.append([start_time, speaker, speech])
            i += 3  # Move to the next block of time, speaker, and speech
        else:
            i += 1  # Move to the next line if the current one doesn't contain a time range
    return data

# File path to your .docx transcript
##################################################################
file_path = 'Management Meeting_April_1._2024-04-19.docx'  # put the file name after iuploading it to your working directory
##################################################################

# Reading the transcript text from a .docx file
transcript_text = read_docx(file_path)

# Creating a DataFrame
data = parse_transcript(transcript_text)
df = pd.DataFrame(data, columns=['Time', 'Speaker', 'Speech'])

# display(df.head(3))

# Export the table, and add "Agenda" mannualy
df.to_excel("20240419.xlsx", index=False)