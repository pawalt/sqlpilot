"""
Modal-based synthetic data generation for SQLPilot.

Uses a self-hosted vLLM server (vllm_server.py) with instructor for structured outputs.
Uses Modal's .map() for parallel execution.

Usage:
    # First deploy the vLLM server and get its URL
    modal deploy datagen/vllm_server.py

    # Then run data generation (add /v1 to the URL!)
    modal run datagen/modal_datagen.py::generate_all --vllm-url https://YOUR-URL.modal.run/v1

    # Or run specific steps
    modal run datagen/modal_datagen.py::generate_topics --vllm-url YOUR_URL/v1
    modal run datagen/modal_datagen.py::generate_topic_details --vllm-url YOUR_URL/v1
    modal run datagen/modal_datagen.py::generate_schemas --vllm-url YOUR_URL/v1
    modal run datagen/modal_datagen.py::generate_statements --vllm-url YOUR_URL/v1
    modal run datagen/modal_datagen.py::generate_training_data
"""

import modal
import json
import os
import random
import string
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel

# Modal app and volumes
app = modal.App("sqlpilot-datagen")
data_volume = modal.Volume.from_name("sqlpilot-data", create_if_missing=True)

DATA_DIR = "/data"

# Subdirectories within the volume
TOPICS_FILE = f"{DATA_DIR}/topics.json"
TOPIC_DETAIL_FILE = f"{DATA_DIR}/topic_detail.json"
TABLES_DIR = f"{DATA_DIR}/tons_of_tables"
STATEMENTS_DIR = f"{DATA_DIR}/diverse_statements"
TRAINING_DATA_DIR = f"{DATA_DIR}/raw"  # Final training data goes here

# Generation parameters
MAX_TABLES = 5
NUM_SELECT_STATEMENTS = 5
NUM_INSERT_STATEMENTS = 5
NUM_OTHER_STATEMENTS = 2

# Statement types
BIG_STATEMENT_TYPES = ["SELECT", "INSERT"]
SMALL_STATEMENT_TYPES = ["DELETE", "UPDATE", "TRUNCATE"]

# Image with dependencies
datagen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "instructor>=1.0.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    )
)

# ============================================================================
# Pydantic models for structured outputs
# ============================================================================


class Topics(BaseModel):
    """List of database application topics."""
    topics: List[str]


class TopicExamples(BaseModel):
    """Examples of how a database could be used for a topic."""
    examples: List[str]


class TableSchema(BaseModel):
    """A single CREATE TABLE statement."""
    table_schema: str


class DatabaseSchema(BaseModel):
    """A database schema with multiple tables."""
    tables: List[TableSchema]


class SQLStatements(BaseModel):
    """A list of SQL statements."""
    statements: List[str]


# ============================================================================
# Helper functions
# ============================================================================


def get_client(base_url: str):
    """Get an instructor-patched OpenAI client pointing to the vLLM server."""
    import instructor
    from openai import OpenAI

    return instructor.patch(
        OpenAI(
            base_url=base_url,
            api_key="not-needed",  # vLLM doesn't require auth
        ),
        mode=instructor.Mode.JSON,  # Use JSON mode for structured outputs
    )


def slugify_topic(ind: int, topic: str) -> str:
    """Convert a topic to a filesystem-safe slug."""
    return f"{ind}_{topic.lower().replace(' ', '_').replace('/', '_')}"


def retry_loop(func, max_attempts: int = 5):
    """Retry a function up to max_attempts times."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
            if attempt == max_attempts - 1:
                raise e


# ============================================================================
# Data augmentation functions (kept from original)
# ============================================================================


def randomize_name(length: int) -> str:
    """Generate a random identifier name."""
    chars = string.ascii_letters + string.digits + '_' * 5
    return ''.join(random.choices(chars, k=length))


def shuffle_columns(create_table_stmt: str):
    """Shuffle columns and randomize names in a CREATE TABLE statement."""
    create_table_stmt = create_table_stmt.strip()
    create_table_stmt = create_table_stmt.replace('`', '')

    lines = create_table_stmt.split('\n')
    if len(lines) < 2:
        return None, None, None

    try:
        first_line_parts = lines[0].split('(')
        table_name = first_line_parts[0].split()[2]
        new_table_name = randomize_name(random.randint(1, 20))
        lines[0] = first_line_parts[0].replace(table_name, new_table_name) + '('

        if len(first_line_parts) > 1 and first_line_parts[1].strip():
            columns = [first_line_parts[1].strip().rstrip(',')] + [
                line.strip().rstrip(',') for line in lines[1:-1]
            ]
        else:
            columns = [line.strip().rstrip(',') for line in lines[1:-1]]

        column_mapping = {}
        for i in range(len(columns)):
            column_name = columns[i].split()[0]
            new_column_name = randomize_name(random.randint(1, 20))
            column_mapping[column_name] = new_column_name
            columns[i] = columns[i].replace(column_name, new_column_name)

        for i in range(len(columns)):
            columns[i] = ' ' * random.randint(0, 10) + columns[i]

        random.shuffle(columns)
        shuffled_columns = lines[0] + '\n' + ',\n'.join(columns) + '\n'

        return f"{shuffled_columns});", {table_name: new_table_name}, column_mapping

    except Exception as e:
        print(f"shuffle_columns: {e}")
        return None, None, None


def replace_names(query: str, table_mapping: dict, column_mapping: dict) -> str:
    """Replace table and column names in a query."""
    merged_mapping = {**table_mapping, **column_mapping}
    sorted_mapping = sorted(merged_mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for old_name, new_name in sorted_mapping:
        query = query.replace(old_name, new_name)

    return query


def process_statements(create_table_stmts: List[str], query: str):
    """Augment a query by randomizing table/column names."""
    # Flatten in case multiple tables are packed together
    create_table_stmts = [
        item
        for stmt in create_table_stmts
        for item in stmt.split('\n\n')
    ]

    table_mapping = {}
    column_mapping = {}
    shuffled_stmts = []

    for stmt in create_table_stmts:
        shuffled_stmt, new_table_mapping, new_column_mapping = shuffle_columns(stmt)

        if shuffled_stmt is None:
            return None, None

        if new_table_mapping:
            table_mapping.update(new_table_mapping)
            column_mapping.update(new_column_mapping)
        shuffled_stmts.append(shuffled_stmt)

    shuffled_stmts = [
        replace_names(stmt, table_mapping, column_mapping)
        for stmt in shuffled_stmts
    ]
    random.shuffle(shuffled_stmts)
    modified_query = replace_names(query, table_mapping, column_mapping)

    return shuffled_stmts, modified_query


# ============================================================================
# Worker functions for parallel execution
# ============================================================================


@app.function(
    image=datagen_image,
    timeout=7200,  # 2 hours
)
def generate_topic_detail_worker(
    vllm_url: str,
    topic_idx: int,
    topic: str,
) -> Tuple[str, List[str]] | None:
    """Generate details for a single topic. Returns (topic_slug, examples) or None on failure."""
    try:
        client = get_client(vllm_url)
        NUM_SUBTOPICS = 10
        NUM_EXAMPLES = 10

        print(f"Processing topic {topic_idx}: {topic}")

        # Generate subtopics
        subtopics = retry_loop(lambda: client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            response_model=TopicExamples,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate {NUM_SUBTOPICS} topics that are related but diverse from: {topic}"
                },
            ],
            max_tokens=1024,
        ))

        all_examples = []
        # Generate examples for each subtopic
        for subtopic in subtopics.examples:
            try:
                examples = retry_loop(lambda: client.chat.completions.create(
                    model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
                    response_model=TopicExamples,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Generate {NUM_EXAMPLES} examples of how a SQL database could be used 
in the context of {subtopic}. Write exactly one sentence for each example. 
Examples should be diverse with little overlapping vocabulary."""
                        },
                    ],
                    max_tokens=1024,
                ))
                all_examples.extend(examples.examples)
            except Exception as e:
                print(f"Failed to generate examples for subtopic '{subtopic}': {e}")
                continue

        topic_slug = slugify_topic(topic_idx, topic)
        return topic_slug, all_examples
    except Exception as e:
        print(f"Failed to generate topic details for '{topic}': {e}")
        return None


@app.function(
    image=datagen_image,
    timeout=7200,  # 2 hours
)
def generate_schema_worker(
    vllm_url: str,
    topic_slug: str,
    detail_idx: int,
    detail: str,
) -> Tuple[str, int, List[Dict]] | None:
    """Generate schemas for a single topic detail. Returns (topic_slug, detail_idx, schemas) or None on failure."""
    try:
        client = get_client(vllm_url)

        print(f"Generating schemas for {topic_slug}/{detail_idx}: {detail[:50]}...")

        schemas = []
        for num_tables in range(1, MAX_TABLES + 1):
            try:
                schema = retry_loop(lambda: client.chat.completions.create(
                    model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
                    response_model=DatabaseSchema,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Generate an example CockroachDB database schema with exactly {num_tables} tables.
The schema must be related to: {detail}
Format each table as CREATE TABLE statements.
Include appropriate column types, primary keys, and foreign keys where relevant."""
                        },
                    ],
                    max_tokens=2048,
                ))
                schemas.append(schema.model_dump())
            except Exception as e:
                print(f"Failed to generate schema with {num_tables} tables for {topic_slug}/{detail_idx}: {e}")
                continue

        if not schemas:
            print(f"No schemas generated for {topic_slug}/{detail_idx}")
            return None

        return topic_slug, detail_idx, schemas
    except Exception as e:
        print(f"Failed to generate schemas for {topic_slug}/{detail_idx}: {e}")
        return None


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=7200,  # 2 hours
)
def generate_statements_worker(
    vllm_url: str,
    topic_slug: str,
    schema_file_path: str,
) -> Tuple[str, str, Dict[str, List[str]]] | None:
    """Generate statements for a single schema. Returns (topic_slug, schema_id, statements_by_type) or None on failure."""
    try:
        # Extract schema_id from file path
        schema_id = os.path.basename(schema_file_path).replace('.json', '')
        
        # Read the schema file
        with open(schema_file_path, "r") as f:
            schemas = json.loads(f.read())
        
        # Build table CREATE statements for each schema
        table_creates = []
        for schema in schemas:
            creates = "\n\n".join(
                table["table_schema"] for table in schema["tables"]
            )
            table_creates.append(creates)
        
        client = get_client(vllm_url)

        print(f"Generating statements for {topic_slug}/{schema_id}")

        all_statement_types = BIG_STATEMENT_TYPES + SMALL_STATEMENT_TYPES
        statements_by_type = {}

        for statement_type in all_statement_types:
            num_statements = (
                NUM_SELECT_STATEMENTS if statement_type in BIG_STATEMENT_TYPES
                else NUM_OTHER_STATEMENTS
            )

            created_statements = []
            for create_stmt in table_creates:
                for complexity in ["simple", "complex"]:
                    try:
                        stmts = retry_loop(lambda: client.chat.completions.create(
                            model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
                            response_model=SQLStatements,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""Generate {num_statements} {complexity} {statement_type} SQL statements.
Statements should be diverse in their orderings, values, and formatting.
Database schema:
{create_stmt}"""
                                },
                            ],
                            max_tokens=4096,
                        ))
                        created_statements.extend(stmts.statements)
                    except Exception as e:
                        print(f"Failed to generate {complexity} {statement_type} statements: {e}")
                        continue

            statements_by_type[statement_type.lower()] = created_statements

        # Check if we got any statements at all
        total_statements = sum(len(s) for s in statements_by_type.values())
        if total_statements == 0:
            print(f"No statements generated for {topic_slug}/{schema_id}")
            return None

        return topic_slug, schema_id, statements_by_type
    except Exception as e:
        print(f"Failed to generate statements for {topic_slug}/{schema_id}: {e}")
        return None


# ============================================================================
# Main orchestration functions
# ============================================================================


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=57600,  # 16 hours
)
def generate_topics(
    vllm_url: str = "https://your-vllm-server.modal.run/v1",
    topic_count: int = 75,
):
    """Generate a list of database application topics."""
    client = get_client(vllm_url)

    print(f"Generating {topic_count} topics...")
    topics = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        response_model=Topics,
        messages=[
            {
                "role": "user",
                "content": f"""Generate a list of {topic_count} topics/applications that a SQL database might be used for. 
Each topic should be 1-2 words and represent a distinct domain or use case.
Examples: e-commerce, healthcare, logistics, social media, banking, etc."""
            },
        ],
        max_tokens=2048,
    )

    os.makedirs(os.path.dirname(TOPICS_FILE), exist_ok=True)
    with open(TOPICS_FILE, "w") as f:
        f.write(topics.model_dump_json(indent=2))

    data_volume.commit()
    print(f"Generated {len(topics.topics)} topics")
    return topics.topics


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=57600,  # 16 hours
)
async def generate_topic_details(
    vllm_url: str = "https://your-vllm-server.modal.run/v1",
):
    """Generate detailed examples for each topic using parallel workers."""
    # Load topics
    with open(TOPICS_FILE, "r") as f:
        topics_data = Topics.model_validate_json(f.read())

    # Prepare inputs for parallel execution
    inputs = [
        (vllm_url, i, topic)
        for i, topic in enumerate(topics_data.topics)
    ]

    print(f"Processing {len(inputs)} topics in parallel...")

    # Run in parallel using async starmap to avoid blocking heartbeat
    topic_detail_map = {}
    succeeded = 0
    failed = 0
    total_examples = 0

    async for result in generate_topic_detail_worker.starmap.aio(inputs):
        if result is None:
            failed += 1
            continue
        topic_slug, examples = result
        topic_detail_map[topic_slug] = examples
        succeeded += 1
        total_examples += len(examples)
        print(f"Completed: {topic_slug} ({len(examples)} examples)")

    with open(TOPIC_DETAIL_FILE, "w") as f:
        f.write(json.dumps(topic_detail_map, indent=2))

    data_volume.commit()
    
    print("\n" + "=" * 60)
    print("TOPIC DETAILS GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total topics attempted: {len(inputs)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {succeeded / len(inputs) * 100:.1f}%")
    print(f"  Total examples generated: {total_examples}")
    print("=" * 60)


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=57600,  # 16 hours
)
async def generate_schemas(
    vllm_url: str = "https://your-vllm-server.modal.run/v1",
    topic_count: int = 0,
):
    """Generate database schemas for each topic detail using parallel workers."""
    with open(TOPIC_DETAIL_FILE, "r") as f:
        topic_details = json.loads(f.read())

    os.makedirs(TABLES_DIR, exist_ok=True)

    # Prepare inputs for parallel execution
    inputs = []
    topics_to_process = list(topic_details.items())
    if topic_count > 0:
        topics_to_process = topics_to_process[:topic_count]

    for topic_slug, details in topics_to_process:
        topic_dir = f"{TABLES_DIR}/{topic_slug}"
        os.makedirs(topic_dir, exist_ok=True)

        for i, detail in enumerate(details):
            detail_filename = f"{topic_dir}/{i}.json"
            if not os.path.exists(detail_filename):
                inputs.append((vllm_url, topic_slug, i, detail))

    print(f"Processing {len(inputs)} schema tasks in parallel...")

    # Run in parallel using async starmap to avoid blocking heartbeat
    succeeded = 0
    failed = 0
    total_schemas = 0

    async for result in generate_schema_worker.starmap.aio(inputs):
        if result is None:
            failed += 1
            continue
        topic_slug, detail_idx, schemas = result
        detail_filename = f"{TABLES_DIR}/{topic_slug}/{detail_idx}.json"
        with open(detail_filename, "w") as f:
            f.write(json.dumps(schemas, indent=2))
        succeeded += 1
        total_schemas += len(schemas)
        print(f"Completed: {topic_slug}/{detail_idx} ({len(schemas)} schemas)")

    data_volume.commit()
    
    print("\n" + "=" * 60)
    print("SCHEMA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total tasks attempted: {len(inputs)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {succeeded / max(len(inputs), 1) * 100:.1f}%")
    print(f"  Total schemas generated: {total_schemas}")
    print("=" * 60)


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=57600,  # 16 hours
)
async def generate_statements(
    vllm_url: str = "https://your-vllm-server.modal.run/v1",
    topic_count: int = 0,
):
    """Generate SQL statements for each schema using parallel workers."""
    topic_dirs = sorted(os.listdir(TABLES_DIR))
    if topic_count > 0:
        topic_dirs = topic_dirs[:topic_count]

    # Prepare inputs for parallel execution - just collect file paths, don't read files
    inputs = []
    for topic_dir in topic_dirs:
        tables_path = f"{TABLES_DIR}/{topic_dir}"
        if not os.path.isdir(tables_path):
            continue

        schema_files = sorted(os.listdir(tables_path))

        for schema_file in schema_files:
            schema_id = schema_file.replace('.json', '')
            statements_dir = f"{STATEMENTS_DIR}/{topic_dir}/{schema_id}"

            # Check if already done
            if os.path.exists(statements_dir) and os.listdir(statements_dir):
                continue

            # Just pass the file path - worker will read it
            schema_file_path = f"{tables_path}/{schema_file}"
            inputs.append((vllm_url, topic_dir, schema_file_path))

    print(f"Processing {len(inputs)} statement tasks in parallel...")

    # Run in parallel using async starmap to avoid blocking heartbeat
    succeeded = 0
    failed = 0
    total_statements = 0

    async for result in generate_statements_worker.starmap.aio(inputs):
        if result is None:
            failed += 1
            continue
        topic_slug, schema_id, statements_by_type = result
        statements_dir = f"{STATEMENTS_DIR}/{topic_slug}/{schema_id}"
        os.makedirs(statements_dir, exist_ok=True)

        stmt_count = sum(len(stmts) for stmts in statements_by_type.values())
        for stmt_type, statements in statements_by_type.items():
            with open(f"{statements_dir}/{stmt_type}.json", "w") as f:
                f.write(json.dumps({"statements": statements}, indent=2))

        succeeded += 1
        total_statements += stmt_count
        print(f"Completed: {topic_slug}/{schema_id} ({stmt_count} statements)")

    data_volume.commit()
    
    print("\n" + "=" * 60)
    print("STATEMENT GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total tasks attempted: {len(inputs)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {succeeded / max(len(inputs), 1) * 100:.1f}%")
    print(f"  Total statements generated: {total_statements}")
    print("=" * 60)


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=57600,  # 16 hours
)
def generate_training_data(num_splits: int = 30):
    """Process generated data into final training format with augmentation."""
    if not os.path.exists(STATEMENTS_DIR):
        print(f"No statements directory found at {STATEMENTS_DIR}")
        return

    topic_dirs = sorted(os.listdir(STATEMENTS_DIR))

    result_statements = []

    for topic_dir in topic_dirs:
        statements_topic_path = f"{STATEMENTS_DIR}/{topic_dir}"
        tables_topic_path = f"{TABLES_DIR}/{topic_dir}"

        if not os.path.isdir(statements_topic_path):
            continue

        schema_dirs = sorted(os.listdir(statements_topic_path))

        for schema_id in schema_dirs:
            schema_path = f"{statements_topic_path}/{schema_id}"

            if not os.path.isdir(schema_path):
                continue

            # Load corresponding table schemas
            tables_file = f"{tables_topic_path}/{schema_id}.json"
            if not os.path.exists(tables_file):
                continue

            with open(tables_file, "r") as f:
                schemas = json.loads(f.read())

            # Extract table schemas
            table_schemas = [
                [table["table_schema"] for table in schema["tables"]]
                for schema in schemas
            ]

            # Process each statement file
            statement_files = sorted(os.listdir(schema_path))

            for stmt_file in statement_files:
                stmt_path = f"{schema_path}/{stmt_file}"
                if not stmt_path.endswith('.json'):
                    continue

                with open(stmt_path, "r") as f:
                    stmt_data = json.loads(f.read())

                statements = stmt_data.get("statements", [])

                # Match statements with schemas (cyclically if needed)
                for i, statement in enumerate(statements):
                    if not table_schemas:
                        continue
                    matched_tables = table_schemas[i % len(table_schemas)]

                    # Augment: randomize names and shuffle
                    shuffled_creates, modified_stmt = process_statements(
                        matched_tables, statement
                    )

                    if shuffled_creates is None:
                        continue

                    create_table_str = "\n\n".join(shuffled_creates)

                    training_line = f"""### TABLEDATA

{create_table_str.strip()}

### STATEMENT

{modified_stmt}"""

                    result_statements.append(training_line)

    print(f"Generated {len(result_statements)} training examples")

    if not result_statements:
        print("No training examples generated!")
        return

    # Shuffle and split
    random.shuffle(result_statements)
    result_splits = [result_statements[i::num_splits] for i in range(num_splits)]

    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    for i, split in enumerate(result_splits):
        with open(f"{TRAINING_DATA_DIR}/{i}.json", "w") as f:
            f.write(json.dumps(split, indent=2))

    data_volume.commit()
    print(f"Wrote {num_splits} training data files to {TRAINING_DATA_DIR}")


@app.function(
    image=datagen_image,
    volumes={DATA_DIR: data_volume},
    timeout=57600,  # 16 hours
)
def generate_all(
    vllm_url: str = "https://your-vllm-server.modal.run/v1",
    topic_count: int = 75,
    force: bool = False,
):
    """Run the full data generation pipeline. Resumes from where it left off unless force=True."""
    
    # Check what's already done
    has_topics = os.path.exists(TOPICS_FILE)
    has_topic_details = os.path.exists(TOPIC_DETAIL_FILE)
    has_schemas = os.path.exists(TABLES_DIR) and len(os.listdir(TABLES_DIR)) > 0
    has_statements = os.path.exists(STATEMENTS_DIR) and len(os.listdir(STATEMENTS_DIR)) > 0
    has_training_data = os.path.exists(TRAINING_DATA_DIR) and len(os.listdir(TRAINING_DATA_DIR)) > 0
    
    print("=" * 60)
    print("PIPELINE STATUS CHECK")
    print("=" * 60)
    print(f"  Topics:        {'✓ EXISTS' if has_topics else '✗ MISSING'}")
    print(f"  Topic details: {'✓ EXISTS' if has_topic_details else '✗ MISSING'}")
    print(f"  Schemas:       {'✓ EXISTS' if has_schemas else '✗ MISSING'}")
    print(f"  Statements:    {'✓ EXISTS' if has_statements else '✗ MISSING'}")
    print(f"  Training data: {'✓ EXISTS' if has_training_data else '✗ MISSING'}")
    print("=" * 60)
    
    if force:
        print("Force mode enabled - running all steps")
    
    # Step 1: Topics
    if force or not has_topics:
        print("\n" + "=" * 60)
        print("STEP 1: Generating topics")
        print("=" * 60)
        generate_topics.remote(vllm_url=vllm_url, topic_count=topic_count)
    else:
        print("\n[SKIP] Step 1: Topics already exist")

    # Step 2: Topic details
    if force or not has_topic_details:
        print("\n" + "=" * 60)
        print("STEP 2: Generating topic details (parallel)")
        print("=" * 60)
        generate_topic_details.remote(vllm_url=vllm_url)
    else:
        print("[SKIP] Step 2: Topic details already exist")

    # Step 3: Schemas
    if force or not has_schemas:
        print("\n" + "=" * 60)
        print("STEP 3: Generating schemas (parallel)")
        print("=" * 60)
        generate_schemas.remote(vllm_url=vllm_url)
    else:
        print("[SKIP] Step 3: Schemas already exist")

    # Step 4: Statements (always run - it has internal idempotency to skip completed schemas)
    print("\n" + "=" * 60)
    print("STEP 4: Generating statements (parallel)")
    print("=" * 60)
    if has_statements and not force:
        print("Note: Existing statements found - will only generate for schemas without statements")
    generate_statements.remote(vllm_url=vllm_url)

    # Step 5: Training data
    if force or not has_training_data:
        print("\n" + "=" * 60)
        print("STEP 5: Generating training data")
        print("=" * 60)
        generate_training_data.remote()
    else:
        print("[SKIP] Step 5: Training data already exists")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# ============================================================================
# Local entrypoints
# ============================================================================


@app.local_entrypoint()
def main():
    """Print usage instructions."""
    print("SQLPilot Data Generation")
    print("=" * 40)
    print()
    print("First, deploy the vLLM server:")
    print("  modal deploy datagen/vllm_server.py")
    print()
    print("Get the URL from the deployment output, then run (add /v1 to URL!):")
    print("  modal run datagen/modal_datagen.py::generate_all --vllm-url https://YOUR-URL.modal.run/v1")
    print()
    print("Or run individual steps:")
    print("  modal run datagen/modal_datagen.py::generate_topics --vllm-url URL/v1")
    print("  modal run datagen/modal_datagen.py::generate_topic_details --vllm-url URL/v1")
    print("  modal run datagen/modal_datagen.py::generate_schemas --vllm-url URL/v1")
    print("  modal run datagen/modal_datagen.py::generate_statements --vllm-url URL/v1")
    print("  modal run datagen/modal_datagen.py::generate_training_data")
