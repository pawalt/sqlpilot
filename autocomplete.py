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
import time
import torch
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


class CachedCompleter:
    """
    SQL completer with KV cache for the table schema prefix.
    
    Caches the prefix token IDs and embeddings to avoid re-computing them.
    """
    
    def __init__(self, model, tokenizer, table_schema: str):
        self.model = model
        self.tokenizer = tokenizer
        self.table_schema = table_schema
        
        # Build the prefix that never changes
        self.prefix = f"""### TABLEDATA

{table_schema}

### STATEMENT

"""
        # Pre-tokenize the prefix (saves tokenization time on each call)
        self.prefix_ids = tokenizer.encode(self.prefix, add_special_tokens=True)
        self.prefix_len = len(self.prefix_ids)
        
        # Pre-compute embeddings for the prefix
        print("Pre-computing prefix embeddings...")
        prefix_tensor = torch.tensor([self.prefix_ids])
        with torch.no_grad():
            self.prefix_embeds = model.get_input_embeddings()(prefix_tensor)
        print(f"Cached {self.prefix_len} prefix tokens!")
        
        # Cache the vocab for constraint function
        self.vocab = tokenizer.get_vocab()
    
    def get_predictions(self, statement: str, num_candidates: int = 20):
        """Generate completions for the given statement."""
        
        # Find the partial word at the end of the statement (if any)
        partial_word = ""
        statement_for_model = statement
        
        if statement:
            # Find last whitespace
            last_space = max(statement.rfind(' '), statement.rfind('\n'), statement.rfind('\t'))
            if last_space == -1:
                # No whitespace - entire statement is the partial word
                partial_word = statement
                statement_for_model = ""
            elif last_space < len(statement) - 1:
                # There's text after the last whitespace - that's our partial word
                partial_word = statement[last_space + 1:]
                statement_for_model = statement[:last_space + 1]
        
        # Tokenize just the statement (prefix is already tokenized)
        if statement_for_model:
            statement_ids = self.tokenizer.encode(statement_for_model, add_special_tokens=False)
            full_ids = self.prefix_ids + statement_ids
            
            # Get embeddings for statement and concatenate with cached prefix
            statement_tensor = torch.tensor([statement_ids])
            with torch.no_grad():
                statement_embeds = self.model.get_input_embeddings()(statement_tensor)
            inputs_embeds = torch.cat([self.prefix_embeds, statement_embeds], dim=1)
        else:
            full_ids = self.prefix_ids
            inputs_embeds = self.prefix_embeds
        
        prompt_len = len(full_ids)
        
        # Build constraint function if we have a partial word
        prefix_allowed_tokens_fn = None
        if partial_word:
            prefix_allowed_tokens_fn = self._make_prefix_constraint(partial_word, prompt_len)
        
        # Generate using inputs_embeds (faster than re-tokenizing)
        # We still need input_ids for the decoder
        input_ids = torch.tensor([full_ids])
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                max_length=prompt_len + 30,
                num_return_sequences=num_candidates,
                num_beams=max(num_candidates, 10),
                eos_token_id=self.tokenizer.pad_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        
        # Decode completions
        full_prompt = self.prefix + statement_for_model
        completions = []
        for x in output:
            decoded = self.tokenizer.decode(x, skip_special_tokens=True)
            completion = decoded[len(full_prompt):].replace("\n", " ").strip()
            if completion:
                completions.append(completion)
        
        # Filter and strip the partial word from results
        if partial_word:
            partial_lower = partial_word.lower()
            filtered = []
            for c in completions:
                if c.lower().startswith(partial_lower):
                    # Strip the partial word so user sees what comes after
                    remainder = c[len(partial_word):]
                    filtered.append(remainder)
                else:
                    filtered.append(c)
            completions = filtered
        
        # Dedupe while preserving order
        seen = set()
        results = []
        for c in completions:
            if c not in seen:
                seen.add(c)
                results.append(c)
        return results[:5]
    
    def _make_prefix_constraint(self, partial: str, prompt_len: int):
        """Create a constraint function that forces completions to match the partial word."""
        partial_l = partial.lower()
        vocab = self.vocab
        tokenizer = self.tokenizer
        
        def prefix_allowed_tokens(batch_id, input_ids):
            # How many tokens have we generated so far?
            generated_len = len(input_ids) - prompt_len
            
            if generated_len == 0:
                # First token - must match partial word
                allowed = []
                for token_str, token_id in vocab.items():
                    clean = token_str.replace("▁", " ").replace("Ġ", " ").strip()
                    clean_lower = clean.lower()
                    
                    if clean_lower.startswith(partial_l) or partial_l.startswith(clean_lower):
                        allowed.append(token_id)
                
                return allowed if allowed else list(vocab.values())
            
            # Check if we've completed the partial word
            generated = tokenizer.decode(input_ids[prompt_len:], skip_special_tokens=True)
            generated_clean = generated.replace("▁", " ").replace("Ġ", " ").strip().lower()
            
            if generated_clean.startswith(partial_l) or len(generated_clean) >= len(partial_l):
                return list(vocab.values())
            
            # Still building - constrain to matching tokens
            remaining = partial_l[len(generated_clean):]
            allowed = []
            for token_str, token_id in vocab.items():
                clean = token_str.replace("▁", " ").replace("Ġ", " ").strip()
                clean_lower = clean.lower()
                
                if clean_lower.startswith(remaining) or remaining.startswith(clean_lower):
                    allowed.append(token_id)
            
            return allowed if allowed else list(vocab.values())
        
        return prefix_allowed_tokens


class DebouncedCompleter:
    """
    Wraps a completer with debouncing and background generation.
    
    - Debouncing: waits for user to stop typing before generating
    - Background: generation runs in a thread, UI stays responsive
    - Staleness: ignores results if input changed while generating
    """
    
    def __init__(self, completer, debounce_ms: int = 150):
        self.completer = completer
        self.debounce_ms = debounce_ms
        
        # Current state
        self.current_input = ""
        self.current_suggestions = []
        self.lock = threading.Lock()
        
        # Generation tracking
        self.generation_id = 0  # Incremented on each input change
        self.generating = False
        
        # Debounce timer
        self.timer = None
    
    def update_input(self, text: str):
        """Called when user types. Debounces and triggers background generation."""
        with self.lock:
            self.current_input = text
            self.generation_id += 1
            gen_id = self.generation_id
        
        # Cancel pending timer
        if self.timer:
            self.timer.cancel()
        
        # Start new debounce timer
        self.timer = threading.Timer(
            self.debounce_ms / 1000.0,
            self._start_generation,
            args=(text, gen_id)
        )
        self.timer.start()
    
    def _start_generation(self, text: str, gen_id: int):
        """Start generation in background thread."""
        # Check if still current
        with self.lock:
            if gen_id != self.generation_id:
                return  # Input changed, don't generate
            self.generating = True
        
        # Run generation (this is the slow part)
        try:
            suggestions = self.completer.get_predictions(text)
        except Exception:
            suggestions = []
        
        # Only update if still current
        with self.lock:
            if gen_id == self.generation_id:
                self.current_suggestions = suggestions
            self.generating = False
    
    def get_suggestions(self) -> list[str]:
        """Get current suggestions (non-blocking)."""
        with self.lock:
            return list(self.current_suggestions)
    
    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        with self.lock:
            return self.generating


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


def main_curses(stdscr, completer, table_data):
    curses.curs_set(1)
    stdscr.nodelay(1)  # Non-blocking input
    stdscr.timeout(50)  # Poll every 50ms for UI updates

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # For "generating..." indicator

    height, width = stdscr.getmaxyx()
    text_box = ""
    
    # Wrap completer with debouncing
    debounced = DebouncedCompleter(completer, debounce_ms=150)
    last_suggestions = []

    def redraw():
        """Redraw the entire screen."""
        try:
            stdscr.clear()
            
            # Query line
            query_line = "Query: " + text_box
            if debounced.is_generating():
                query_line += " (generating...)"
            stdscr.addstr(0, 0, query_line[:width-1])
            
            # Suggestions
            suggestions = debounced.get_suggestions()
            for i, suggestion in enumerate(suggestions[:5]):
                if i == 0:
                    stdscr.addstr(i + 2, 0, suggestion[:width-1], curses.color_pair(2))
                else:
                    stdscr.addstr(i + 2, 0, suggestion[:width-1])
            
            # Table schema
            START_BOX = 8
            stdscr.addstr(START_BOX, 0, "-" * min(34, width-1))
            stdscr.addstr(START_BOX + 1, 0, "Table Schema:")
            for i, line in enumerate(table_data.split('\n')[:height - START_BOX - 3]):
                stdscr.addstr(START_BOX + i + 2, 0, line[:width-1])
            
            stdscr.refresh()
        except curses.error:
            pass  # Ignore curses errors (e.g., writing past screen edge)

    # Initial display
    redraw()

    while True:
        key = stdscr.getch()

        if key == curses.KEY_ENTER or key in [10, 13]:
            break
        elif key == curses.KEY_BACKSPACE or key == 127:
            text_box = text_box[:-1]
            debounced.update_input(text_box)
        elif key != -1 and 32 <= key <= 126:  # Printable ASCII
            text_box += chr(key)
            debounced.update_input(text_box)
        
        # Always redraw to show updated suggestions and generating status
        redraw()


def run_tests(model_path: str = "downloaded_model"):
    """Run tests to verify completion is working."""
    print("=" * 60)
    print("AUTOCOMPLETE TESTS")
    print("=" * 60)
    
    # Load model and tokenizer
    tokenizer = get_tokenizer()
    model = get_model(model_path)
    
    table_schema = """CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);"""
    
    print(f"\nTable schema:\n{table_schema}\n")
    print("-" * 60)
    
    # Test 1: Basic generation without caching
    print("\n[TEST 1] Basic generation (no caching)")
    prompt = f"""### TABLEDATA

{table_schema}

### STATEMENT

SELECT """
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt tokens: {len(input_ids[0])}")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 20,
            num_return_sequences=3,
            num_beams=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.pad_token_id,
        )
    
    print("Completions:")
    for i, seq in enumerate(output):
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        completion = decoded[len(prompt):].replace("\n", " ").strip()
        print(f"  {i+1}: {completion[:60]}...")
    
    # Test 2: CachedCompleter with empty statement
    print("\n[TEST 2] CachedCompleter - empty statement")
    completer = CachedCompleter(model, tokenizer, table_schema)
    
    results = completer.get_predictions("")
    print("Completions for '':")
    for i, r in enumerate(results):
        print(f"  {i+1}: {r[:60]}...")
    
    # Test 3: CachedCompleter with "SELECT"
    print("\n[TEST 3] CachedCompleter - 'SELECT'")
    results = completer.get_predictions("SELECT ")
    print("Completions for 'SELECT ':")
    for i, r in enumerate(results):
        print(f"  {i+1}: {r[:60]}...")
    
    # Test 4: CachedCompleter with partial word "SEL"
    print("\n[TEST 4] CachedCompleter - 'SEL' (partial word)")
    results = completer.get_predictions("SEL")
    print("Completions for 'SEL':")
    for i, r in enumerate(results):
        print(f"  {i+1}: {r[:60]}...")
    
    # Test 5: CachedCompleter with "SELECT ema" (partial column)
    print("\n[TEST 5] CachedCompleter - 'SELECT ema' (partial column)")
    results = completer.get_predictions("SELECT ema")
    print("Completions for 'SELECT ema':")
    for i, r in enumerate(results):
        print(f"  {i+1}: {r[:60]}...")
    
    # Test 6: Longer query
    print("\n[TEST 6] CachedCompleter - 'SELECT * FROM users WHERE'")
    results = completer.get_predictions("SELECT * FROM users WHERE ")
    print("Completions for 'SELECT * FROM users WHERE ':")
    for i, r in enumerate(results):
        print(f"  {i+1}: {r[:60]}...")
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


def main():
    # Check for --test flag
    if "--test" in sys.argv:
        model_path = "downloaded_model"
        for i, arg in enumerate(sys.argv):
            if arg == "--test" and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                model_path = sys.argv[i + 1]
                break
            elif not arg.startswith("-") and arg != sys.argv[0]:
                model_path = arg
        run_tests(model_path)
        return
    
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
    
    # Create cached completer (pre-computes KV cache for table schema)
    completer = CachedCompleter(model, tokenizer, table_data)
    
    print("\nStarting interactive mode...")
    print("Type SQL and see completions. Press Enter to exit.\n")
    
    # Run curses UI
    curses.wrapper(main_curses, completer, table_data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
