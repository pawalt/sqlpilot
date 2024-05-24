import curses
import threading
from transformers import LlamaForCausalLM, LlamaTokenizerFast

def get_tokenizer():
    tokenizer = LlamaTokenizerFast(
        tokenizer_file="smol_tokenizer/tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        padding_side="right"
    )
    return tokenizer

def get_model():
    # checkpoint_loc = "tosave/embd_15M/checkpoint-7500"
    checkpoint_loc = "output/checkpoint-29500"
    model = LlamaForCausalLM.from_pretrained(checkpoint_loc)
    model.eval()
    return model

def get_predictions(model, tokenizer, table, statement, max_length=512):
    text = f"""### TABLEDATA

{table}

### STATEMENT

{statement}"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 30,
        num_return_sequences=5,
        # do_sample=True,
        num_beams=10,
        eos_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    MAX_EXTRA_CHARS = 1000000
    ret = list(map(lambda x: tokenizer.decode(
        x,
        skip_special_tokens=True,
    )[len(text):][:MAX_EXTRA_CHARS].replace("\n", ""), output))
    return ret

def get_multiline_input(prompt):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return '\n'.join(lines).strip()

def update_suggestions_thread(
    model,
    tokenizer,
    table_data,
    text_box,
    stdscr,
    height,
    table_stmt,
):
    suggestions = get_predictions(model, tokenizer, table_data, text_box)
    stdscr.clear()
    stdscr.addstr(0, 0, "Query: " + text_box)
    for i, suggestion in enumerate(suggestions[:height - 3]):
        if i == 0:
            stdscr.addstr(i + 2, 0, suggestion, curses.color_pair(2))
        else:
            stdscr.addstr(i + 2, 0, suggestion)

    # add the create table statement to the right of the suggestions
    START_IND = 50
    START_BOX = 7
    stdscr.addstr(START_BOX, 0, "----------------------------------")
    stdscr.addstr(START_BOX + 1, 0, "Table: ")
    for i, line in enumerate(table_stmt.split('\n')[:height - 3]):
        stdscr.addstr(START_BOX + i + 3, 0, line)

    stdscr.refresh()

def main(stdscr, table_data):
    model = get_model()
    tokenizer = get_tokenizer()

    curses.curs_set(1)
    stdscr.nodelay(0)

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)

    height, width = stdscr.getmaxyx()

    text_box = ""

    while True:
        key = stdscr.getch()

        if key == curses.KEY_ENTER or key in [10, 13]:
            stdscr.addstr(height - 1, 0, "Entered: " + text_box)
            stdscr.refresh()
            stdscr.getch()
            break
        elif key == curses.KEY_BACKSPACE or key == 127:
            text_box = text_box[:-1]
        elif key != -1 and chr(key).isprintable():
            text_box += chr(key)

        threading.Thread(target=update_suggestions_thread, args=(model, tokenizer, table_data, text_box, stdscr, height, table_data)).start()

if __name__ == "__main__":
    table_data = get_multiline_input("Enter table data: ")
    curses.wrapper(main, table_data)
