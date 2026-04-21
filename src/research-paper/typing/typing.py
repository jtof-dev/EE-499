#!/bin/python

import csv
import random
import time
import tkinter as tk
import urllib.request

# configuration
TEST_DURATION_SECONDS = 300
CSV_OUTPUT_FILE = "metrics.csv"
WORD_CACHE_FILE = "local_word_cache.txt"
WORD_LIST_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"


def fetch_words():
    """Fetches dictionary. Caches locally to ensure tests run offline without network latency."""
    # if os.path.exists(WORD_CACHE_FILE):
    #     with open(WORD_CACHE_FILE, "r") as f:
    # filter out 1-2 letter words to maintain typing difficulty
    # return [w.strip() for w in f.readlines() if len(w.strip()) >= 3]

    try:
        req = urllib.request.Request(
            WORD_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            words = response.read().decode("utf-8").splitlines()

        valid_words = [w.strip() for w in words if len(w.strip()) >= 3]

        # Save cache for future offline runs
        with open(WORD_CACHE_FILE, "w") as f:
            f.write("\n".join(valid_words))

        return valid_words
    except Exception:
        # Fallback list prevents the script from crashing if internet drops
        return ["error", "network", "failure", "fallback", "offline", "typing"]


class EEGTypingTest:
    def __init__(self, root, words):
        self.root = root
        self.root.title("EEG Sudden Death Typing")
        self.root.geometry("1024x600")
        self.root.configure(bg="black")
        self.words = words

        # state variables
        self.is_running = False
        self.start_time = 0
        self.word_left = ""
        self.word_center = ""
        self.word_right = ""
        self.target_word = ""
        self.typed_progress = ""

        # telemetry
        self.keys_pressed_this_sec = 0
        self.errors_this_sec = 0
        self.total_words_completed = 0

        # initialize CSV and write headers
        self.csv_file = open(CSV_OUTPUT_FILE, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestampMs", "keys_pressed", "errors", "total_correct"]
        )

        # UI elements
        # left peripheral word. locked to screen center (relx=0.5) but pushed left 280px (x=-280)
        self.left_label = tk.Label(
            self.root, text="", font=("Courier New", 24), bg="black", fg="#444444"
        )
        self.left_label.place(relx=0.5, x=-280, rely=0.4, anchor="e")

        # right peripheral word. locked to screen center, pushed right 280px
        self.right_label = tk.Label(
            self.root, text="", font=("Courier New", 24), bg="black", fg="#444444"
        )
        self.right_label.place(relx=0.5, x=280, rely=0.4, anchor="w")

        # center typing area. uses Text widget instead of Label to lock character width and prevent visual jitter
        self.center_text = tk.Text(
            self.root,
            height=1,
            width=15,
            font=("Courier New", 48, "bold"),
            bg="black",
            bd=0,
            highlightthickness=0,
            state="disabled",
            cursor="arrow",
        )
        self.center_text.place(relx=0.5, rely=0.4, anchor="center")

        # define text colors for the center widget
        self.center_text.tag_configure(
            "typed", foreground="#00ff00"
        )  # green for correct keys
        self.center_text.tag_configure(
            "untyped", foreground="white"
        )  # white for remaining keys
        self.center_text.tag_configure("center", justify="center")

        # bottom instructions
        self.info_label = tk.Label(
            self.root,
            text="Type the center word. A typo wipes all words.\nEyes center. Hands still.",
            font=("Helvetica", 14),
            bg="black",
            fg="gray",
        )
        self.info_label.place(relx=0.5, rely=0.8, anchor="center")

        # listen for any keyboard input
        self.root.bind("<Key>", self.handle_keypress)
        self.render_center_text("PRESS SPACE", "")

    def get_timestamp_ms(self):
        """Returns Unix epoch time. Syncs exactly with BrainFlow CSV timestamps."""
        return int(time.time() * 1000)

    def start_test(self):
        """Initializes timers, populates words, and starts telemetry loop."""
        self.is_running = True
        self.start_time = time.time()
        self.refresh_all_words()
        self.update_timer()
        self.log_telemetry()

    def refresh_all_words(self):
        """Pulls three random words. Used on startup and when penalizing a mistake."""
        self.word_left = random.choice(self.words)
        self.word_center = random.choice(self.words)
        self.word_right = random.choice(self.words)
        self.setup_current_word()

    def advance_queue(self):
        """Shifts words leftward upon successful typing. Pulls a new right word."""
        self.word_left = self.word_center
        self.word_center = self.word_right
        self.word_right = random.choice(self.words)
        self.setup_current_word()

    def setup_current_word(self):
        """Appends a space to force spacebar usage, resets typing progress."""
        self.target_word = self.word_center + " "
        self.typed_progress = ""
        self.update_display()

    def render_center_text(self, remaining, typed):
        """Injects text into the locked Text widget, coloring typed/untyped segments."""
        self.center_text.config(state="normal")
        self.center_text.delete("1.0", tk.END)

        self.center_text.insert("1.0", typed, "typed")
        self.center_text.insert(tk.END, remaining, "untyped")

        self.center_text.tag_add("center", "1.0", tk.END)
        self.center_text.config(state="disabled")

    def update_display(self):
        """Updates peripheral labels and recalculates center split."""
        self.left_label.config(text=self.word_left)
        self.right_label.config(text=self.word_right)

        remaining_word = self.target_word[len(self.typed_progress) :]
        self.render_center_text(remaining_word, self.typed_progress)

    def trigger_error(self):
        """Executes sudden death penalty: logs error, flushes queue, flashes screen."""
        self.errors_this_sec += 1
        self.refresh_all_words()

        # visual penalty
        error_bg = "#4a0000"
        self.root.configure(bg=error_bg)
        self.center_text.configure(bg=error_bg)
        self.left_label.configure(bg=error_bg)
        self.right_label.configure(bg=error_bg)

        # reset background to black after 100 milliseconds
        self.root.after(100, self.reset_background)

    def reset_background(self):
        """Restores normal black theme after an error flash."""
        self.root.configure(bg="black")
        self.center_text.configure(bg="black")
        self.left_label.configure(bg="black")
        self.right_label.configure(bg="black")

    def handle_keypress(self, event):
        """Processes keystrokes, tracks correctness, and triggers progression/errors."""
        if not self.is_running and event.keysym == "space":
            self.start_test()
            return

        if not self.is_running:
            return

        # ignore modifier keys
        if len(event.char) == 0:
            return

        self.keys_pressed_this_sec += 1
        expected_char = self.target_word[len(self.typed_progress)]

        if event.char == expected_char:
            self.typed_progress += event.char
            self.update_display()

            # if word is fully typed (including space), advance the queue
            if len(self.typed_progress) == len(self.target_word):
                self.total_words_completed += 1
                self.advance_queue()
        else:
            self.trigger_error()

    def log_telemetry(self):
        """Background loop. Writes data to CSV exactly once per second."""
        if not self.is_running:
            return

        timestamp = self.get_timestamp_ms()
        self.csv_writer.writerow(
            [
                timestamp,
                self.keys_pressed_this_sec,
                self.errors_this_sec,
                self.total_words_completed,
            ]
        )
        self.csv_file.flush()  # force write to disk to prevent data loss

        # reset per-second counters
        self.keys_pressed_this_sec = 0
        self.errors_this_sec = 0

        # reschedule this function to run 1000ms from now
        self.root.after(1000, self.log_telemetry)

    def update_timer(self):
        """Background loop. Updates the visible countdown clock."""
        if not self.is_running:
            return

        elapsed = time.time() - self.start_time
        remaining = int(TEST_DURATION_SECONDS - elapsed)

        if remaining <= 0:
            self.end_test()
        else:
            self.info_label.config(text=f"Time Remaining: {remaining}s")
            self.root.after(1000, self.update_timer)

    def end_test(self):
        """Halts telemetry, closes CSV, and displays final stats."""
        self.is_running = False
        self.csv_file.close()

        self.left_label.config(text="")
        self.right_label.config(text="")
        self.render_center_text("TEST COMPLETE", "")
        self.info_label.config(
            text=f"Total Words: {self.total_words_completed}\nData saved to {CSV_OUTPUT_FILE}."
        )


if __name__ == "__main__":
    word_list = fetch_words()

    root = tk.Tk()
    app = EEGTypingTest(root, word_list)
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)
    root.mainloop()
