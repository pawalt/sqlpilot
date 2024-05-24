from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast
import os
import re


def get_tokenizer():
    # Load the trained tokenizer
    tokenizer = LlamaTokenizerFast(
        tokenizer_file="smol_tokenizer/tokenizer.json",

        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",

        padding_side="right"
    )
    # tokenizer = LlamaTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
    # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_latest_checkpoint():
    # Specify the directory containing the checkpoints
    checkpoint_dir = "output"

    # Get a list of all checkpoint directories
    checkpoint_dirs = [dir for dir in os.listdir(checkpoint_dir) if dir.startswith("checkpoint-")]

    # Find the checkpoint with the highest index
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(re.findall(r'\d+', x)[0]))


    # Construct the path to the latest checkpoint
    return os.path.join(checkpoint_dir, latest_checkpoint)

tokenizer = get_tokenizer()

checkpoint_loc = get_latest_checkpoint()
# checkpoint_loc = "tosave/embd_15M/checkpoint-7500"
print(f"using checkpoint {checkpoint_loc}")

# Load the model from the checkpoint
model = LlamaForCausalLM.from_pretrained(checkpoint_loc)
# model = LlamaForCausalLM.from_pretrained("tosave/260k_long_customtok")

print(f"num params: {model.num_parameters()}")

# Set the model to evaluation mode
model.eval()

prompts = [
    """### TABLEDATA""",
    """
### TABLEDATA

CREATE TABLE sendy_stats (
    powerful UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ind INT,
    forreal VARCHAR(50) NOT NULL,
    eggplant JSONB
);

CREATE TABLE hello_twitter (
    its_crazy UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    how INT,
    much_performance VARCHAR(50) NOT NULL,
    8M_params_can_hold JSONB
);

### STATEMENT

SELECT * from sendy_stats WHERE
""",
""" 
### TABLEDATA

CREATE TABLE renewable_energy_production (
    id INT PRIMARY KEY,
    energy_type TEXT,
    production_amount FLOAT
);

CREATE TABLE resource_allocation (
    id INT PRIMARY KEY,
    location TEXT,
    allocation_date DATE,
);

### STATEMENT
""",

"""
### TABLEDATA

CREATE TABLE xCWhlg9E0OIE4jqkbs4DFx2K (
 nf8bf_ZCC9OHi97FzqUzOx TEXT,
        Q7koov423 TIMESTAMP,
    A8wHuC9kWQFfJx94Qbs70rcEP INT PRIMARY KEY,
        tYUZt8MI INT
);

CREATE TABLE hwugNEpYcJKC (
mM6lnPxm4E1w7 INT PRIMARY KEY,
         md8vdeRYLvai1gMz92qTW507iPqTM TEXT,
     6Wo9nR7Lk TEXT
);

### STATEMENT""",
"""
### TABLEDATA

CREATE TABLE xCWhlg9E0OIE4jqkbs4DFx2K (
 nf8bf_ZCC9OHi97FzqUzOx TEXT,
        Q7koov423 TIMESTAMP,
    A8wHuC9kWQFfJx94Qbs70rcEP INT PRIMARY KEY,
        tYUZt8MI INT
);

CREATE TABLE hwugNEpYcJKC (
mM6lnPxm4E1w7 INT PRIMARY KEY,
         md8vdeRYLvai1gMz92qTW507iPqTM TEXT,
     6Wo9nR7Lk TEXT
);

CREATE TABLE p8QP9BHxSf7fca (
     ifd0gF INT PRIMARY KEY,
       uvohcl0RZE2BAgFNJRHi TIMESTAMP,
   jzrqNL0yas3yqfk0IOpa9WMb TEXT,
 1SW INT
);

CREATE TABLE BIPQGnJUMR9EU6TaOa18 (
VQy_wlXPVzi58AFmQbXXggwHf_CtH TEXT,
yzACflmu TEXT,
     OBefvst9dNqgIdXrnclNlf INT PRIMARY KEY
);

### STATEMENT""",
"""
### TABLEDATA

CREATE TABLE f (
 _a VARCHAR(255),
       UxF0Rp3PSbhHV INT PRIMARY KEY,
   Mt_vExhBqrm VARCHAR(255)
);

CREATE TABLE y (
        Iz INT,
         p4TRJFQiKKrWq VARCHAR(255),
      49G DATE,
   UEy KEY (Iz) REFERENCES f(Iz),
       t_A INT PRIMARY KEY
);

### STATEMENT
""",
]

prompts = list(map(lambda x: x.strip(), prompts))

# Function to generate text based on a prompt
def generate_text(prompt, max_length=512):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 100,
        num_return_sequences=3,
        num_beams=10,
        # do_sample=True,
        temperature=0.5,
        eos_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    return list(map(lambda x: tokenizer.decode(x), output))

for i, prompt in enumerate(prompts):
    print(f"GENERATION {i}:")
    print("-" * 40)
    generations = generate_text(prompt)
    for j, gen in enumerate(generations):
        print(f"SUB_GENERATION {j}: =====================")
        print(gen)

