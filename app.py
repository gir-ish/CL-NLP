import warnings
warnings.filterwarnings('ignore')

import os
import re
import nest_asyncio
from datetime import datetime, timedelta
import uuid
import json
import logging

from dateutil.parser import parse  # To handle loosely formatted dates

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

from typing import List
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# --------------------------
# LangChain & Related Imports
# --------------------------
from langchain.agents import Tool, ConversationalAgent, AgentExecutor
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory  # Using full conversation history
# We are no longer using ChatOpenAI but the OllamaLLM model.
from langchain_openai import OpenAIEmbeddings  # We still need embeddings for vector search.
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, AIMessage  # For converting dicts to message objects

# --------------------------
# Pinecone Setup for RetrievalQA
# --------------------------
from langchain_pinecone import PineconeVectorStore

# --------------------------
# Environment Variables
# (Ensure these are defined in your .env file or replace with your keys)
# --------------------------
os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""
logging.info("API keys loaded.")

# Initialize embeddings & Pinecone vector store
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name='health-info',  # Update with your actual index name
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# --------------------------
# Instantiate the OllamaLLM with the gemma3:4b model.
# --------------------------
ollama_llm = OllamaLLM(model="gemma3:4b", temperature=0.7)
logging.info("Ollama gemma3:4b model loaded.")

# Build a RetrievalQA chain for general hospital FAQs using the Ollama model.
qa = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever
)
logging.info("Retriever initialized.")

faq_tool = Tool(
    name="IMA Hospital FAQ Bot",
    func=qa.run,
    description="Use this tool for general hospital information such as services, directions, or FAQs."
)
logging.info("FAQ tool created.")

# --------------------------
# MongoDB Setup Atlas
# --------------------------
from db.mongodb import db  
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]  # Ensure this collection exists
logging.info("MongoDB connected and collections loaded.")

print("___________________________________________________________")
print("___________________________________________________________")
print("___________________________________________________________")
if "doctors" in db.list_collection_names():
    print("The 'doctors' collection exists.")
else:
    print("The 'doctors' collection does not exist.")

if "appointments" in db.list_collection_names():
    print("The 'appointments' collection exists.")
else:
    print("The 'appointments' collection does not exist.")
print("___________________________________________________________")
print("___________________________________________________________")
print("___________________________________________________________")

def lookup_doctor_or_appointment(query: str) -> str:
    logging.info("lookup_doctor_or_appointment called with query: %s", query)
    # First, try to match a doctor by name, filtering on availability.
    match = re.search(r"Dr\.?\s+([A-Za-z]+)", query, re.IGNORECASE)
    if match:
        doctor_name = match.group(1)
        logging.info("Doctor name extracted: %s", doctor_name)
        doctor = doctors_collection.find_one({
            "name": {"$regex": doctor_name, "$options": "i"},
            "availability": True
        })
        logging.info("Database query for doctor by name executed.")
        if doctor:
            logging.info("Doctor found: %s", doctor.get('name', 'N/A'))
            languages = doctor.get('languages', [])
            lang_str = ', '.join(languages) if languages else "N/A"
            return (
                f"Doctor Details:\n"
                f"Name: {doctor.get('name', 'N/A')}\n"
                f"Specialty: {doctor.get('specialty', 'N/A')}\n"
                f"Languages: {lang_str}\n"
                f"Availability: {doctor.get('availability', 'Not Available')}"
            )
        else:
            logging.info("No available doctor found by name: %s", doctor_name)
            return "No available doctor details found for that name."
    else:
        logging.info("No doctor name found in query, checking for specialization.")
        # Attempt to search by specialist keyword using the "specialty" field.
        specializations = ["cardiologist", "dermatologist", "neurologist", "pediatrician", "orthopedic", "oncologist"]
        for spec in specializations:
            if spec in query.lower():
                logging.info("Specialization '%s' found in query.", spec)
                doctor = doctors_collection.find_one({
                    "specialty": {"$regex": spec, "$options": "i"},
                    "availability": True
                })
                if doctor:
                    logging.info("Doctor with specialization %s found: %s", spec, doctor.get('name', 'N/A'))
                    languages = doctor.get('languages', [])
                    lang_str = ', '.join(languages) if languages else "N/A"
                    return (
                        f"Doctor Details:\n"
                        f"Name: {doctor.get('name', 'N/A')}\n"
                        f"Specialty: {doctor.get('specialty', 'N/A')}\n"
                        f"Languages: {lang_str}\n"
                        f"Availability: {doctor.get('availability', 'Not Available')}"
                    )
                else:
                    logging.info("No available doctor found for specialization: %s", spec)
                    return f"No available doctor found for {spec}."
        logging.info("No specific doctor information detected in query.")
        return "No specific doctor information detected in your query."

def find_next_available_slot(doctor: str, requested_dt: datetime) -> str:
    logging.info("find_next_available_slot called for doctor: %s at requested time: %s", doctor, requested_dt)
    closing_dt = requested_dt.replace(hour=21, minute=0)
    current_dt = requested_dt + timedelta(minutes=10)
    while current_dt < closing_dt:
        slot_str = current_dt.strftime("%Y-%m-%d %H:%M")
        logging.debug("Checking slot: %s", slot_str)
        if not appointments_collection.find_one({
            "doctor": doctor,
            "appointment": slot_str
        }):
            logging.info("Next available slot found: %s", slot_str)
            return slot_str
        current_dt += timedelta(minutes=10)
    logging.info("No available slot found on that day for doctor: %s", doctor)
    return None

def parse_appointment_time(time_str: str) -> str:
    logging.info("parse_appointment_time called with input: %s", time_str)
    try:
        # First, try the strict format.
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        logging.info("Time parsed using strict format: %s", dt)
    except Exception as e:
        logging.info("Strict parsing failed, attempting fuzzy parsing: %s", e)
        # If strict parsing fails, use dateutil's parser with a default date set to January 1, 2025.
        default_dt = datetime(2025, 1, 1)
        dt = parse(time_str, fuzzy=True, default=default_dt)
        logging.info("Time parsed using fuzzy parsing: %s", dt)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
    logging.info("Formatted appointment time: %s", formatted_time)
    return formatted_time

def ensure_dict(input_data):
    logging.info("ensure_dict called with input data type: %s", type(input_data))
    if isinstance(input_data, dict):
        return input_data
    elif isinstance(input_data, str):
        try:
            # First, try to parse it as JSON.
            result = json.loads(input_data)
            logging.info("Input string successfully parsed as JSON.")
            return result
        except Exception as e:
            logging.info("JSON parsing failed (%s), attempting naive conversion.", e)
            # Fallback: Split by commas and colons.
            result = {}
            for part in input_data.split(","):
                if ":" in part:
                    key, val = part.split(":", 1)
                    result[key.strip().lower()] = val.strip()
            logging.info("Naive conversion result: %s", result)
            return result
    else:
        logging.error("Unexpected type for appointment details: %s", type(input_data))
        raise ValueError("Unexpected type for appointment details. Expected dict or str.")

KEY_MAPPING = {
    "appointment date and time": "appointment_time",
    "appointment_date_time": "appointment_time",
    "appointment_time": "appointment_time",
    "doctor name": "doctor",
    "dr": "doctor",
    "doctor": "doctor",
    "symptom description": "symptom_description",
    "symptom_description": "symptom_description",
    "patient details": "patient",
    "patient info": "patient",
    "patient": "patient"
}

def normalize_details(details: dict) -> dict:
    # Convert keys to lower case and map to standard names
    return { KEY_MAPPING.get(key.lower(), key): value for key, value in details.items() }

def book_appointment(appointment_details: dict) -> str:
    logging.info("book_appointment called with details: %s", appointment_details)
    appointment_details = ensure_dict(appointment_details)
    # Normalize keys using a generalized mapping
    appointment_details = normalize_details(appointment_details)
    logging.info("Normalized details: %s", appointment_details)
    
    # Parse appointment_time
    try:
        formatted_time = parse_appointment_time(appointment_details["appointment_time"])
        appointment_details["appointment_time"] = formatted_time
        requested_dt = datetime.strptime(formatted_time, "%Y-%m-%d %H:%M")
        logging.info("Parsed appointment time: %s", formatted_time)
    except Exception as e:
        logging.error("Error parsing appointment time: %s", e)
        return str(e)
    
    # Verify working hours (9 AM to 9 PM)
    opening_dt = requested_dt.replace(hour=9, minute=0)
    closing_dt = requested_dt.replace(hour=21, minute=0)
    if not (opening_dt <= requested_dt < closing_dt):
        logging.info("Time %s outside working hours", formatted_time)
        return "Appointment time must be within hospital working hours (9 AM to 9 PM)."
    
    # Book appointment (direct insert)
    try:
        result = appointments_collection.insert_one(appointment_details)
        if result.inserted_id:
            ack = f"Appointment booked successfully with ID: {result.inserted_id}. Details: {appointment_details}"
            logging.info("Booking acknowledged: %s", ack)
            return ack
        else:
            logging.error("No booking acknowledgment received.")
            return "Booking failed, please try again."
    except Exception as e:
        logging.error("Exception during booking: %s", e)
        return "Error booking appointment, please try again."

appointment_booking_tool = Tool(
    name="IMA Hospital Appointment Booking",
    func=book_appointment,
    description=(
        "Use this tool to book an appointment. Provide complete details in JSON format including doctor's name or specialty, "
        "patient's symptom description, appointment date and time (the tool accepts loosely formatted inputs and converts them to YYYY-MM-DD HH:MM, defaulting the year to 2025 if missing), and patient details "
        "(full name, age, gender, contact number, optionally email). Note: Hospital working hours are 9 AM to 9 PM, "
        "and appointments are available every 10 minutes."
    )
)

def search_appointment(query: dict) -> list:
    logging.info("search_appointment called with query: %s", query)
    query = ensure_dict(query)
    appointments = appointments_collection.find(query)
    result = []
    for appointment in appointments:
        appointment["_id"] = str(appointment["_id"])
        result.append(appointment)
    logging.info("search_appointment returning %d appointments", len(result))
    return result

appointment_search_tool = Tool(
    name="IMA Hospital Appointment Search",
    func=search_appointment,
    description="Use this tool to search for appointments. The query should include keys like 'doctor', 'appointment_time', or the patient's name."
)

# --------------------------
# Assemble All Tools
# --------------------------
tools = [
    faq_tool, 
    Tool(
        name="IMA Hospital Doctor Lookup",
        func=lookup_doctor_or_appointment,
        description="Use this tool to retrieve available doctor details by name or specialist from our MongoDB Atlas database."
    ),
    appointment_booking_tool,
    appointment_search_tool
]

# --------------------------
# Revised System Prompt with Detailed Instructions
# --------------------------
system_message = """
You are Helix, an advanced, customer-oriented, chain-of-thought AI assistant for IMA Hospital. Your role is to analyze each query carefully and decide the best action based on the following guidelines:

1. GENERAL HOSPITAL INFORMATION:
   - If the query is about hospital services, directions, or FAQs, use the "IMA Hospital FAQ Bot" tool to provide clear and accurate information.

2. DOCTOR OR SPECIALIST LOOKUP:
   - If the query directly mentions a specific doctor (e.g., "Dr. Smith") or asks for details about a specialty (e.g., "cardiologist", "dermatologist"), search the MongoDB database.
   - If a matching doctor is available in the hospital, return their details such as name, specialty, languages, and availability.
   - If no available doctor is found, inform the user politely that the requested doctor or specialist is not currently available.

3. SYMPTOM-BASED SPECIALIST SUGGESTION:
   - If the query describes patient symptoms without directly requesting an appointment, use the Pinecone database to determine the most relevant specialist based on those symptoms.
   - Then, cross-check the MongoDB database to confirm whether a doctor with that specialty is available in the hospital.
   - Provide the suggested specialist’s details if available, or notify the user if the specialist is not available.

4. APPOINTMENT BOOKING:
   - If the query includes a specific appointment time (either in the strict format YYYY-MM-DD HH:MM or loosely formatted), first verify the availability of the slot by checking the appointments database.
   - If the requested slot is already booked, suggest the next available 10-minute slot before proceeding with the booking.
   - Make sure that all required details—doctor's name/specialty, patient's symptom description, appointment date and time (converted to YYYY-MM-DD HH:MM with the year defaulting to 2025 if missing), and patient details (full name, age, gender, contact number, and optionally email)—are provided.
   - Always confirm with the user before finalizing the booking.

5. SYMPTOM-RELATED CONCERNS:
   - If the query describes concerning symptoms (for example, chest pain) but does not explicitly state an intent to book an appointment, ask clarifying questions before suggesting a specialist or booking an appointment.

Other Guidelines:
   - Use the Pinecone database solely for matching symptoms to specialists.
   - Analyze every query step by step to decide which tool or combination of tools to invoke.
   - Combine results from multiple sources (FAQ, doctor lookup, specialist suggestion, and appointment booking) to provide a concise, informative final response.
   - Always maintain a professional, empathetic, and polite tone, and consider hospital operating hours (9 AM to 9 PM) and 10-minute appointment intervals in all recommendations.

Begin!
"""


human_message = """
Begin!

{chat_history}
Question: {input}
{agent_scratchpad}
"""

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=system_message,
    suffix=human_message,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

# Create the LLM chain using the OllamaLLM instance.
llm_chain = LLMChain(llm=ollama_llm, prompt=prompt)

def get_agent_executor() -> AgentExecutor:
    logging.info("Creating agent executor with full conversation memory.")
    agent = ConversationalAgent(
        llm_chain=llm_chain,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory
    )
    logging.info("Agent executor created successfully.")
    return agent_executor

# --------------------------
# FastAPI Application Setup with Session Support
# --------------------------
app = FastAPI()

# Updated ChatRequest model: now only "input" is required.
class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    response: str
    chat_history: List[dict]

# Global dictionary to hold session histories.
session_histories = {}

@app.get("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())
    session_histories[session_id] = []  # Initialize an empty history for this session.
    logging.info("New session created with session_id: %s", session_id)
    return {"session_id": session_id}

@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_endpoint(session_id: str, chat_request: ChatRequest):
    logging.info("chat_endpoint called for session_id: %s", session_id)
    # Retrieve existing history or default to empty list if not found.
    existing_history = session_histories.get(session_id, [])
    logging.info("Existing history length for session_id %s: %d", session_id, len(existing_history))

    # Convert existing_history from list of dicts to list of LangChain message objects.
    def convert_dict_to_message(item):
        role = item.get("role")
        content = item.get("content")
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        else:
            return HumanMessage(content=content)

    message_objects = [convert_dict_to_message(item) for item in existing_history if isinstance(item, dict)]
    logging.info("Converted chat history to message objects for session_id: %s", session_id)

    agent_executor = get_agent_executor()
    # Load the full conversation history (as message objects) into memory.
    agent_executor.memory.chat_memory.messages = message_objects

    response = agent_executor.run(chat_request.input)
    logging.info("Agent executor completed processing for session_id: %s", session_id)

    # Convert conversation memory messages back to plain dictionaries.
    updated_history = []
    for msg in agent_executor.memory.chat_memory.messages:
        if isinstance(msg, dict):
            updated_history.append(msg)
        else:
            msg_type = msg.__class__.__name__
            if msg_type == "HumanMessage":
                role = "user"
            elif msg_type == "AIMessage":
                role = "assistant"
            else:
                role = "unknown"
            updated_history.append({"role": role, "content": msg.content})
    logging.info("Updated chat history length for session_id %s: %d", session_id, len(updated_history))
    
    # Update the session history.
    session_histories[session_id] = updated_history

    return ChatResponse(response=response, chat_history=updated_history)

if __name__ == "__main__":
    logging.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logging.info("FastAPI server started successfully.")
    logging.info("FastAPI server stopped.")
