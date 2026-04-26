#!/bin/python

import csv
import random
import time
import tkinter as tk

# configuration
TEST_DURATION_SECONDS = 300
CSV_OUTPUT_FILE = "metrics.csv"
COLORS = ["red", "green", "blue", "yellow"]

KEY_MAPPING = {"h": "red", "j": "green", "k": "blue", "l": "yellow"}


class StroopTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capstone EEG Stroop Test")
        self.root.geometry("800x600")
        self.root.configure(bg="black")

        # state variables
        self.is_running = False
        self.start_time = 0
        self.current_ink_color = ""
        self.current_streak = 0
        self.max_streak = 0

        # telemetry
        self.keys_pressed_this_sec = 0
        self.errors_this_sec = 0
        self.total_correct_cumulative = 0
        self.total_attempts = 0

        # initialize CSV and write standardized headers to match the typing test
        self.csv_file = open(CSV_OUTPUT_FILE, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestampMs", "keys_pressed", "errors", "total_correct", "current_streak"]
        )

        # UI elements

        # top instructions / timer
        self.info_label = tk.Label(
            self.root,
            text="press SPACE to begin\neyes center",
            font=("Helvetica", 24),  # increased font size
            bg="black",
            fg="gray",
        )
        self.info_label.place(relx=0.5, rely=0.1, anchor="center")

        # live streak counter
        self.streak_label = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 64, "bold"),
            bg="black",
            fg="white",
        )
        self.streak_label.place(relx=0.5, rely=0.28, anchor="center")

        # main word display
        self.word_label = tk.Label(
            self.root,
            text="EEG STROOP",
            font=("Helvetica", 72, "bold"),
            bg="black",
            fg="white",
        )
        self.word_label.place(relx=0.5, rely=0.5, anchor="center")

        # visual key guide
        self.canvas = tk.Canvas(
            self.root, width=400, height=100, bg="black", highlightthickness=0
        )
        self.canvas.place(relx=0.5, rely=0.8, anchor="center")
        self.draw_key_guide()

        # listen for any keyboard input globally
        self.root.bind("<Key>", self.handle_keypress)

    def draw_key_guide(self):
        """Draws 4 colored circles with H J K L inside."""
        keys = ["H", "J", "K", "L"]
        colors = ["red", "green", "blue", "yellow"]

        circle_radius = 30
        spacing = 100
        start_x = 50
        y_pos = 50

        for i in range(4):
            x_pos = start_x + (i * spacing)
            self.canvas.create_oval(
                x_pos - circle_radius,
                y_pos - circle_radius,
                x_pos + circle_radius,
                y_pos + circle_radius,
                fill=colors[i],
                outline="white",
                width=2,
            )
            self.canvas.create_text(
                x_pos, y_pos, text=keys[i], fill="black", font=("Helvetica", 24, "bold")
            )

    def get_timestamp_ms(self):
        """Returns Unix epoch time."""
        return int(time.time() * 1000)

    def start_test(self):
        """Initializes timers, resets attempts, and starts the background logging loop."""
        self.is_running = True
        self.total_attempts = 0
        self.current_streak = 0
        self.max_streak = 0
        self.start_time = time.time()

        self.streak_label.config(text="0")

        self.next_word()
        self.update_timer()
        self.log_telemetry()

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

    def next_word(self):
        """Generates the next word and color randomly."""
        word_text = random.choice(COLORS)
        self.current_ink_color = random.choice(COLORS)
        self.word_label.config(text=word_text.upper(), fg=self.current_ink_color)

    def trigger_error_flash(self):
        """Flashes screen dark red."""
        self.errors_this_sec += 1

        error_bg = "#4a0000"
        self.root.configure(bg=error_bg)
        self.word_label.configure(bg=error_bg)
        self.info_label.configure(bg=error_bg)
        self.streak_label.configure(bg=error_bg)
        self.canvas.configure(bg=error_bg)

        # resets back to black after 100 milliseconds
        self.root.after(100, self.reset_background)

    def reset_background(self):
        """Restores normal black theme after an error flash."""
        self.root.configure(bg="black")
        self.word_label.configure(bg="black")
        self.info_label.configure(bg="black")
        self.streak_label.configure(bg="black")
        self.canvas.configure(bg="black")

    def handle_keypress(self, event):
        """Tracks input, evaluates cognitive correctness, and logs attempts."""
        key = event.char.lower()

        # start game on spacebar
        if not self.is_running and event.keysym == "space":
            self.start_test()
            return

        # ignore inputs if game is over or if key isn't H, J, K, or L
        if not self.is_running or key not in KEY_MAPPING:
            return

        self.keys_pressed_this_sec += 1
        self.total_attempts += 1

        # check if key is correct
        if KEY_MAPPING[key] == self.current_ink_color:
            self.total_correct_cumulative += 1
            self.current_streak += 1

            # update max streak if current streak breaks the record
            if self.current_streak > self.max_streak:
                self.max_streak = self.current_streak
        else:
            # reset streak on mistake
            self.current_streak = 0
            self.trigger_error_flash()

        # update the visible streak label to just the number
        self.streak_label.config(text=str(self.current_streak))

        # load the next word
        self.next_word()

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
                self.total_correct_cumulative,
                self.current_streak,
            ]
        )
        self.csv_file.flush()  # force write to disk to prevent data loss

        # reset counters for the next second of data collection
        self.keys_pressed_this_sec = 0
        self.errors_this_sec = 0

        # reschedule this function to run exactly 1000ms from now
        self.root.after(1000, self.log_telemetry)

    def end_test(self):
        """Halts test, closes the CSV safely, and shows final accuracy for self-review."""
        self.is_running = False
        self.csv_file.close()

        accuracy = (
            (self.total_correct_cumulative / self.total_attempts * 100)
            if self.total_attempts > 0
            else 0
        )

        self.word_label.config(text="TEST COMPLETE", fg="white")
        self.streak_label.config(text="")  # hide the live streak counter at the end
        self.info_label.config(
            text=f"total attempts: {self.total_attempts} | accuracy: {accuracy:.1f}%\n"
            f"longest streak: {self.max_streak}\n\n"
            f"data saved to {CSV_OUTPUT_FILE}."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = StroopTestApp(root)

    # forces the Tkinter window to the front over the terminal
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    root.mainloop()
