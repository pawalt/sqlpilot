import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json
import os

LLAMA = "llama3:8b"
PHI = "phi3"

# enables `response_model` in create call
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

class Recipe(BaseModel):
    description: str
    steps: List[str]
    title: str
    ingredients: List[str]
    total_time_minutes: int

class Recipes(BaseModel):
    recipes: List[Recipe]

def generate_recipes():
    NUM_RECIPES = 5
    # generate a list of topics to build schemas about
    return client.chat.completions.create(
        model=LLAMA,
        response_model=Recipes,
        messages=[
            {
                "role": "user",
                "content": f"""Generate a list of {NUM_RECIPES} recipes that use canned mussels in them."""
            },
        ]
    )

if __name__ == "__main__":
    recipes = generate_recipes()
    print(json.dumps(recipes.dict(), indent=2))