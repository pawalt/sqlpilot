import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

GPT_4 = "gpt-4-0125-preview"
GPT_3 = "gpt-3.5-turbo-0125"

class TableSchema(BaseModel):
    table_schema: str

class DatabaseSchema(BaseModel):
    tables: List[TableSchema]

client = instructor.patch(OpenAI())

SCHEMA_COUNT = 1
MAX_TABLES = 5

class Topics(BaseModel):
    topics: List[str]

TOPIC_COUNT = 10
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

for topic in topics.topics:
    for i in range(1, MAX_TABLES + 1):
        schemas = client.chat.completions.create(
            model=GPT_3,
            response_model=DatabaseSchema,
            messages=[
                {
                    "role": "user",
                    "content": f"""Generate an example CockroachDB database schema which has exactly {i} tables.
The schema should be related the following topic: {topic}.
Schemas should be formatted as the output of SHOW CREATE TABLE"""
                },
            ]
        )
        print(schemas.model_dump_json(indent=2))

