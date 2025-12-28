#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "tokenizers",
#   "sentencepiece",
# ]
# ///
"""
Interactive SQL autocomplete demo.

Usage:
    uv run autocomplete.py [MODEL_PATH]

    MODEL_PATH defaults to ./downloaded_model
"""

import sys
import curses
import threading
from transformers import LlamaForCausalLM, LlamaTokenizerFast


def get_tokenizer():
    tokenizer = LlamaTokenizerFast(
        tokenizer_file="tokenizer/tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        padding_side="right"
    )
    return tokenizer


def get_model(checkpoint_loc: str):
    print(f"Loading model from {checkpoint_loc}...")
    model = LlamaForCausalLM.from_pretrained(checkpoint_loc)
    model.eval()
    print(f"Model loaded ({model.num_parameters():,} parameters)")
    return model


def get_predictions(model, tokenizer, table, statement):
    text = f"""### TABLEDATA

{table}

### STATEMENT

{statement}"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 30,
        num_return_sequences=5,
        num_beams=10,
        eos_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    ret = [
        tokenizer.decode(x, skip_special_tokens=True)[len(text):].replace("\n", "")
        for x in output
    ]
    return ret


def get_multiline_input(prompt):
    print(prompt)
    print("(Enter an empty line when done)")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return '\n'.join(lines).strip()


def update_suggestions_thread(model, tokenizer, table_data, text_box, stdscr, height):
    try:
        suggestions = get_predictions(model, tokenizer, table_data, text_box)
        stdscr.clear()
        stdscr.addstr(0, 0, "Query: " + text_box)
        for i, suggestion in enumerate(suggestions[:height - 3]):
            if i == 0:
                stdscr.addstr(i + 2, 0, suggestion, curses.color_pair(2))
            else:
                stdscr.addstr(i + 2, 0, suggestion)

        # Show the table schema below
        START_BOX = 7
        stdscr.addstr(START_BOX, 0, "----------------------------------")
        stdscr.addstr(START_BOX + 1, 0, "Table Schema:")
        for i, line in enumerate(table_data.split('\n')[:height - START_BOX - 3]):
            stdscr.addstr(START_BOX + i + 2, 0, line)

        stdscr.refresh()
    except Exception:
        pass  # Ignore errors in background thread


def main_curses(stdscr, model, tokenizer, table_data):
    curses.curs_set(1)
    stdscr.nodelay(0)

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)

    height, width = stdscr.getmaxyx()
    text_box = ""

    # Initial display
    stdscr.addstr(0, 0, "Query: ")
    stdscr.addstr(7, 0, "----------------------------------")
    stdscr.addstr(8, 0, "Table Schema:")
    for i, line in enumerate(table_data.split('\n')[:height - 10]):
        stdscr.addstr(9 + i, 0, line)
    stdscr.refresh()

    while True:
        key = stdscr.getch()

        if key == curses.KEY_ENTER or key in [10, 13]:
            break
        elif key == curses.KEY_BACKSPACE or key == 127:
            text_box = text_box[:-1]
        elif key != -1 and chr(key).isprintable():
            text_box += chr(key)

        threading.Thread(
            target=update_suggestions_thread,
            args=(model, tokenizer, table_data, text_box, stdscr, height),
            daemon=True,
        ).start()


def main():
    # Parse model path from args
    model_path = sys.argv[1] if len(sys.argv) > 1 else "downloaded_model"
    
    # Load model and tokenizer
    tokenizer = get_tokenizer()
    model = get_model(model_path)
    
    # Get table schema from user
    print("\n" + "="*50)
    table_data = get_multiline_input("Enter your CREATE TABLE statement(s):")
    
    if not table_data:
        print("No table data provided. Using example...")
        table_data = """CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);"""
    
    print("\nStarting interactive mode...")
    print("Type SQL and see completions. Press Enter to exit.\n")
    
    # Run curses UI
    curses.wrapper(main_curses, model, tokenizer, table_data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

