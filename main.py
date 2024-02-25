import instructor
import random
import sys
from openai import OpenAI
from pydantic import BaseModel
import re
from typing import List
from dotenv import load_dotenv
import json
import os
from itertools import permutations

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

def slugify_topic(ind: int, topic: str) -> str:
    return f"{ind}_{topic.lower().replace(' ', '_')}"

TABLE_DIR = "data/tables"

def generate_tables(topics: List[str]):
    topic_index = -1
    for topic in topics:
        topic_index += 1
        topic_filename = f"{TABLE_DIR}/{slugify_topic(topic_index, topic)}.json"

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

NUM_BIG_STATEMENTS = 20
BIG_STATEMENT_TYPES = [
    "SELECT",
    "INSERT SELECT FROM",
]

# we don't need much ddl
NUM_SMALL_STATEMENTS = 3
SMALL_STATEMENT_TYPES = [
    "UPSERT",
    "DELETE",
    "UPDATE",
    "INSERT",
    "TRUNCATE",
    "CREATE TABLE",
    "ALTER",
    "DROP",
    "GRANT",
    "REVOKE",
    "SHOW",
]

STATEMENTS_DIR = "data/statements"

class SQLStatements(BaseModel):
    statements: List[str]

def generate_statements():
    topic_files = os.listdir(TABLE_DIR)
    # sort so we move up
    topic_files.sort()

    for topic_file in topic_files:
        topic_header = topic_file.replace(".json", "")

        with open(f"{TABLE_DIR}/{topic_file}", "r") as f:
            schemas = json.loads(f.read())

        table_creates = []
        for schema in schemas:
            # build header with all the table schemas
            creates = "\n\n".join(map(lambda jawn: jawn["table_schema"], schema["tables"]))
            table_creates.append(creates)

        topic_dir = f"{STATEMENTS_DIR}/{topic_header}"
        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir)

        for statement_type in BIG_STATEMENT_TYPES + SMALL_STATEMENT_TYPES:
            num_statements = NUM_BIG_STATEMENTS if statement_type in BIG_STATEMENT_TYPES else NUM_SMALL_STATEMENTS

            statement_filename = f"{topic_dir}/{statement_type.lower().replace(' ', '_')}.json"

            if os.path.exists(statement_filename):
                continue

            print(f"Generating {statement_type} statements for {topic_header}")

            max_tokens = 2048 if statement_type in BIG_STATEMENT_TYPES else 512
            timeout = 60 if statement_type in BIG_STATEMENT_TYPES else 10

            created_statements = []
            for create_stmt in table_creates:
                max_attempts = 5

                for attempt in range(max_attempts):
                    try:
                        statements = client.chat.completions.create(
                            timeout=timeout,
                            # limit max tokens to fail early if the model is going haywire
                            max_tokens=max_tokens,
                            model=GPT_3,
                            response_model=SQLStatements,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""Generate {num_statements} example {statement_type} SQL statements for the following database schema:
{create_stmt}"""
                                }
                            ]
                        )
                        
                        created_statements.append(statements.dict())
                        break
                    except Exception as e:
                        print(f"Error generating {statement_type} statements for {topic_header}: {e}")
                        if attempt == max_attempts - 1:
                            print(f"Failed to generate {statement_type} statements for {topic_header} after {max_attempts} attempts")
                            raise e

            with open(statement_filename, "w") as f:
                f.write(json.dumps(created_statements, indent=2))

def generate_training_str(tables: List[str], statement: str) -> str:
    table_perms = permutations(tables)

    ret = []
    for perm in table_perms:
        joined_tables = "\n\n".join(perm)
        # support both upper and lower case sql
        for stmt in [statement, statement.lower()]:
            ret.append(f"TABLEDATA\n\n{joined_tables}\n\nSTATEMENT\n\n{stmt}")

    return ret

TRAINING_DATA_DIR = "data/training_data"
def generate_training_data():
    topic_files = os.listdir(TABLE_DIR)
    # sort so we move up
    topic_files.sort()

    for topic_file in topic_files:
        topic_header = topic_file.replace(".json", "")

        # get all table schemas for this topic
        topic_table_filepath = f"{TABLE_DIR}/{topic_file}"
        with open(topic_table_filepath, "r") as f:
            table_schemas_raw = json.loads(f.read())
        # generate list of table schemas
        table_schemas = list(map(
            lambda tab: list(map(
                lambda jawn: jawn["table_schema"],
                tab["tables"],
            )),
            table_schemas_raw,
        ))

        topic_statement_dir = f"{STATEMENTS_DIR}/{topic_header}"

        statement_files = os.listdir(topic_statement_dir)
        statement_files.sort()

        for statement_file in statement_files:
            statement_type = statement_file.replace(".json", "")

            # skip if we already have training data for this statement type
            if os.path.exists(f"{topic_data_dir}/{statement_type}.txt"):
                continu

            print(f"Generating training data for {topic_header} {statement_type}")

            statement_filepath = f"{topic_statement_dir}/{statement_file}"

            with open(statement_filepath, "r") as f:
                statements_raw = json.loads(f.read())

            training_strs = []
            for i, table_statements in enumerate(statements_raw):
                # match up each statement with the schema it was generated against
                matched_table = table_schemas[i]
                for individual_statement in table_statements["statements"]:
                    training_strs += generate_training_str(matched_table, individual_statement)

            with open(f"{topic_data_dir}/{statement_type}.txt", "w") as f:
                f.write("<divider>".join(training_strs))

def retry_loop(func, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            print(f"Error: {e}")
            if attempt == max_attempts - 1:
                print(f"Failed after {max_attempts} attempts")
                raise e

class TopicExamples(BaseModel):
    examples: List[str]

def generate_topic_detail():
    topics = read_topics()
    
    topic_detail_map = {}
    for i, topic in enumerate(topics):
            NUM_TOPIC_EXAMPLES = 10

            max_attempts = 5

            for attempt in range(max_attempts):
                try:
                    schema = client.chat.completions.create(
                        model=GPT_3,
                        max_tokens=1024,
                        response_model=TopicExamples,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Generate {NUM_TOPIC_EXAMPLES} examples of how a SQL database could
        be used in the context of {topic}. Write exactly one sentence for each example."""
                            },
                        ]
                    )
                    print(f"Generated examples for {topic} ({i + 1}/{len(topics)})")
                    print(schema.examples)
                    topic_detail_map[slugify_topic(i, topic)] = schema.examples
                    break
                except Exception as e:
                    print(f"Error generating examples for {topic}: {e}")
                    if attempt == max_attempts - 1:
                        print(f"Failed to generate examples for {topic} after {max_attempts} attempts")
                        raise e
    
    with open("data/topic_detail.json", "w") as f:
        f.write(json.dumps(topic_detail_map, indent=2))

def clean_topic_detail():
    with open("data/topic_detail.json", "r") as f:
        topic_detail = json.loads(f.read())

    for topic, examples in topic_detail.items():
        topic_detail[topic] = list(map(lambda ex: re.sub(r'^\d+\.\s+', '', ex), examples))

    with open("data/topic_detail.json", "w") as f:
        f.write(json.dumps(topic_detail, indent=2))

DETAIL_TABLE_DIR = "data/detail_tables"

def generate_topic_tables(start_ind: int, num_topics: int):
    with open('data/topic_detail.json', 'r') as f:
        raw_details = json.loads(f.read())

    # go through the keys of topic detail and only include the ones that are in the range
    # start_ind to start_ind + num_topics
    topic_deets = {
        key: value for key, value in raw_details.items()
        if start_ind <= int(key.split('_')[0]) < start_ind + num_topics
    }

    for topic, details in topic_deets.items():
        topic_basedir = f"{DETAIL_TABLE_DIR}/{topic}"
        if not os.path.exists(topic_basedir):
            os.makedirs(topic_basedir)

        for i, detail in enumerate(details):
            detail_filename = f"{topic_basedir}/{i}.json"
            if os.path.exists(detail_filename):
                continue

            print(f"Generating tables for {topic} ({i})")

            schemas = []
            for i in range(1, MAX_TABLES + 1):
                schema = retry_loop(lambda: client.chat.completions.create(
                    model=GPT_3,
                    max_tokens=1024,
                    response_model=DatabaseSchema,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Generate an example CockroachDB database schema which has exactly {i} tables.
The schema must be related the following theme: {detail}.
Schemas must be formatted as the output of SHOW CREATE TABLE. The should all start with CREATE TABLE.
Table schemas should be formatted with newlines between each column definition."""
                        },
                    ]
                ))
                schemas.append(schema.dict())

            with open(detail_filename, "w") as f:
                f.write(json.dumps(schemas, indent=2))

DETAIL_STATEMENT_DIR = "data/detail_statements"

NUM_BIG_STATEMENTS = 20
DEETBIG_STATEMENT_TYPES = [
    "SELECT",
    "INSERT",
    "UPSERT",
]

# we don't need much ddl
NUM_SMALL_STATEMENTS = 3
DEETSMALL_STATEMENT_TYPES = [
    "DELETE",
    "UPDATE",
    "TRUNCATE",
]

def generate_detail_statements(start_ind: int, num_topics: int):
    topic_files = os.listdir(DETAIL_TABLE_DIR)
    # sort so we move up
    topic_files.sort()
    topic_files = topic_files[start_ind:start_ind + num_topics]

    for topic_ind, topic_dir in enumerate(topic_files):
        to_list = f"{DETAIL_TABLE_DIR}/{topic_dir}"
        listed_files = os.listdir(to_list)
        listed_files.sort()

        for listed in listed_files:
            stripped = listed.replace('.json', '')

            topic_basedir = f"{DETAIL_STATEMENT_DIR}/{topic_dir}/{stripped}"
            if not os.path.exists(topic_basedir):
                os.makedirs(topic_basedir)

            with open(f"{DETAIL_TABLE_DIR}/{topic_dir}/{listed}", "r") as f:
                schemas = json.loads(f.read())

            table_creates = []
            for schema in schemas:
                # build header with all the table schemas
                creates = "\n\n".join(map(lambda jawn: jawn["table_schema"], schema["tables"]))
                table_creates.append(creates)

            for statement_type in DEETBIG_STATEMENT_TYPES + DEETSMALL_STATEMENT_TYPES:
                num_statements = NUM_BIG_STATEMENTS if statement_type in DEETBIG_STATEMENT_TYPES else NUM_SMALL_STATEMENTS

                statement_filename = f"{topic_basedir}/{statement_type.lower().replace(' ', '_')}.json"

                if os.path.exists(statement_filename):
                    continue

                print(f"({topic_ind + 1}/{len(topic_files)}) Generating {statement_type} statements for {topic_dir} ({stripped})")

                max_tokens = 4096
                timeout = 60 if statement_type in DEETBIG_STATEMENT_TYPES else 10

                created_statements = []
                for create_stmt in table_creates:
                    individual = []
                    for stmt_type in ["simple", "complex"]:
                        stmts = retry_loop(lambda: client.chat.completions.create(
                            timeout=timeout,
                            # limit max tokens to fail early if the model is going haywire
                            max_tokens=max_tokens,
                            model=GPT_3,
                            response_model=SQLStatements,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""Generate {num_statements} example {statement_type} SQL statements.
These should be {stmt_type} statements for the following database schema:
{create_stmt}"""
                                }
                            ]
                        ))
                        individual += stmts.statements
                    created_statements.append({
                        "statements": individual,
                    })

                with open(statement_filename, "w") as f:
                    f.write(json.dumps(created_statements, indent=2))

def paralellize_statement_gen():
    from multiprocessing import Process

    processes = []
    BATCH_SIZE = 10 
    for i in range(0, 90, BATCH_SIZE):
        p = Process(target=generate_detail_statements, args=(i, BATCH_SIZE))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

def generate_detailed_training_data():
    topic_files = os.listdir(DETAIL_STATEMENT_DIR)
    # sort so we move up
    topic_files.sort()

    RESULT_SPLITS = 10
    result_statements = []

    for topic_dir in topic_files:
        listed_files = os.listdir(f"{DETAIL_STATEMENT_DIR}/{topic_dir}")
        listed_files.sort()

        for stripped in listed_files:
            listed = f"{stripped}.json"

            topic_statement_basedir = f"{DETAIL_STATEMENT_DIR}/{topic_dir}/{stripped}"

            with open(f"{DETAIL_TABLE_DIR}/{topic_dir}/{listed}", "r") as f:
                table_schemas_raw = json.loads(f.read())

            # generate list of table schemas
            table_schemas = list(map(
                lambda tab: list(map(
                    lambda jawn: jawn["table_schema"],
                    tab["tables"],
                )),
                table_schemas_raw,
            ))


            statement_files = os.listdir(topic_statement_basedir)
            statement_files.sort()

            for statement_file in statement_files:
                statement_filepath = f"{topic_statement_basedir}/{statement_file}"

                with open(statement_filepath, "r") as f:
                    statements_raw = json.loads(f.read())

                for i, table_statements in enumerate(statements_raw):
                    # match up each statement with the schema it was generated against
                    matched_table = table_schemas[i]
                    create_table_st = "\n\n".join(matched_table)

                    for individual_statement in table_statements["statements"]:
                        training_line = f"""### TABLEDATA

{create_table_st.strip()}

### STATEMENT

{individual_statement}"""

                        result_statements.append(training_line)

    # shuffle the result statements
    random.shuffle(result_statements)

    # split into RESULT_SPLITS
    result_lists = [result_statements[i::RESULT_SPLITS] for i in range(RESULT_SPLITS)]

    if not os.path.exists(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR)
        
    # write out result lists to training data dir
    for i, result_list in enumerate(result_lists):
        with open(f"{TRAINING_DATA_DIR}/{i}.json", "w") as f:
            f.write(json.dumps(result_list, indent=2))


if __name__ == "__main__":
    generate_detailed_training_data()
