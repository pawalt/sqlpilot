import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import json
import os

load_dotenv()

GPT_4 = "gpt-4-0125-preview"
GPT_3 = "gpt-3.5-turbo-0125"

client = instructor.patch(OpenAI())

SCHEMA_COUNT = 1
MAX_TABLES = 5

TOPICS_FILE = "data/topics.json"

class Topics(BaseModel):
    topics: List[str]

def generate_topics():
    TOPIC_COUNT = 100
    # generate a list of topics to build schemas about
    topics = client.chat.completions.create(
        model=GPT_3,
        response_model=Topics,
        messages=[
            {
                "role": "user",
                "content": f"""Generate a list of {TOPIC_COUNT} topics/applications that a SQL database might be used for. Each topic should be 1-2 words."""
            },
        ]
    )

    with open(TOPICS_FILE, "w") as f:
        f.write(topics.model_dump_json(indent=2))

class TableSchema(BaseModel):
    table_schema: str

class DatabaseSchema(BaseModel):
    tables: List[TableSchema]

TABLE_DIR = "data/tables"

def generate_tables(topics: List[str]):
    topic_index = -1
    for topic in topics:
        topic_index += 1
        topic_filename = f"{TABLE_DIR}/{topic_index}_{topic.lower().replace(' ', '_')}.json"

        # shit fails sometimes, so resume from our last savepoint
        if os.path.exists(topic_filename):
            continue

        print(f"Generating tables for topic: {topic} ({topic_index + 1}/{len(topics)})")

        schemas = []
        for i in range(1, MAX_TABLES + 1):
            print(i)
            schema = client.chat.completions.create(
                model=GPT_3,
                # limit max tokens to fail early if the model is going haywire
                max_tokens=512,
                response_model=DatabaseSchema,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Generate an example CockroachDB database schema which has exactly {i} tables.
The schema must be related the following topic: {topic}.
Schemas must be formatted as the output of SHOW CREATE TABLE."""
                    },
                ]
            )
            schemas.append(schema.dict())

        with open(topic_filename, "w") as f:
            f.write(json.dumps(schemas, indent=2))

def clean_tables():
    for filename in os.listdir(TABLE_DIR):
        to_keep = []
        with open(f"{TABLE_DIR}/{filename}", "r") as f:
            schema_data = json.loads(f.read())
            for schema in schema_data:
                schema_valid = True
                for table in schema["tables"]:
                    if not "CREATE TABLE" in table["table_schema"]:
                        schema_valid = False
                        break
                if schema_valid:
                    to_keep.append(schema)

        with open(f"{TABLE_DIR}/{filename}", "w") as f:
            f.write(json.dumps(to_keep, indent=2))

def read_topics():
    with open(TOPICS_FILE, "r") as f:
        topics = Topics.model_validate_json(f.read())
        return topics.topics
    
clean_tables()