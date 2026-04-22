#!/bin/python

import csv
import random
import time
import tkinter as tk

# configuration
TEST_DURATION_SECONDS = 300
CSV_OUTPUT_FILE = "metrics.csv"
N_BACK_LEVEL = 3  # 3 is a very high difficulty level
TARGET_PROBABILITY = 0.30

LETTERS = "BCDFGHJKLMNPQRSTVWXYZ"  # excluding vowels here to avoid being able to make pronounceable words


class NBackTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{N_BACK_LEVEL}-Back EEG Test")
        self.root.geometry("800x600")
        self.root.configure(bg="black")

        # state variables
        self.is_running = False
        self.start_time = 0
        self.history = []
        self.is_current_target = False
        self.space_pressed_this_trial = False
        self.trial_active = False

        # telemetry
        self.keys_pressed_this_sec = 0
        self.errors_this_sec = 0
        self.total_correct_cumulative = 0
        self.total_targets_shown = 0

        # initialize CSV and write standardized headers
        self.csv_file = open(CSV_OUTPUT_FILE, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestampMs", "keys_pressed", "errors", "total_correct"]
        )

        # UI elements
        # main letter display
        self.letter_label = tk.Label(
            self.root, text="", font=("Helvetica", 120, "bold"), bg="black", fg="white"
        )
        self.letter_label.place(relx=0.5, rely=0.45, anchor="center")

        # bottom instructions
        self.info_label = tk.Label(
            self.root,
            text=f"press SPACE when the letter matches the one from {N_BACK_LEVEL} steps ago\npress SPACE to begin",
            font=("Helvetica", 16),
            bg="black",
            fg="gray",
        )
        self.info_label.place(relx=0.5, rely=0.15, anchor="center")

        # bind the spacebar
        self.root.bind("<space>", self.handle_keypress)

    def get_timestamp_ms(self):
        """Returns Unix epoch time."""
        return int(time.time() * 1000)

    def start_test(self):
        """Initializes timers and starts the background logging loop."""
        self.is_running = True
        self.start_time = time.time()
        self.info_label.config(text=f"Time Remaining: {TEST_DURATION_SECONDS}s")

        self.update_timer()
        self.log_telemetry()
        self.next_trial()

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

    def next_trial(self):
        """Generates the next letter and evaluates if the user missed the previous one."""
        if not self.is_running:
            return

        # evaluate "omission errors"
        if self.trial_active:
            if self.is_current_target and not self.space_pressed_this_trial:
                self.errors_this_sec += 1
                self.trigger_error_flash()

        # generate the next letter
        if len(self.history) >= N_BACK_LEVEL and random.random() < TARGET_PROBABILITY:
            # force a target match based on the n-back rule
            new_letter = self.history[-N_BACK_LEVEL]
            self.is_current_target = True
            self.total_targets_shown += 1
        else:
            # force a non-match and ensure it doesn't accidentally match the N-Back target
            new_letter = random.choice(LETTERS)
            if len(self.history) >= N_BACK_LEVEL:
                while new_letter == self.history[-N_BACK_LEVEL]:
                    new_letter = random.choice(LETTERS)
            self.is_current_target = False

        # update the rolling memory buffer
        self.history.append(new_letter)
        self.space_pressed_this_trial = False
        self.trial_active = True

        # flash the letter on screen
        self.letter_label.config(text=new_letter)

        # letter stays on screen for 500ms, then disappears
        self.root.after(500, self.hide_letter)

        # new trial starts every 2000ms
        self.root.after(2000, self.next_trial)

    def hide_letter(self):
        """Creates the blank space between letters."""
        if self.is_running:
            self.letter_label.config(text="")

    def trigger_error_flash(self):
        """Flashes screen dark red."""
        error_bg = "#4a0000"
        self.root.configure(bg=error_bg)
        self.letter_label.configure(bg=error_bg)
        self.info_label.configure(bg=error_bg)

        self.root.after(100, self.reset_background)

    def reset_background(self):
        """Restores normal black theme after an error flash."""
        self.root.configure(bg="black")
        self.letter_label.configure(bg="black")
        self.info_label.configure(bg="black")

    def handle_keypress(self, event):
        """Evaluates user input during a trial."""
        if not self.is_running:
            self.start_test()
            return

        # prevent holding down the spacebar or double-tapping in a single trial
        if self.space_pressed_this_trial:
            return

        self.space_pressed_this_trial = True
        self.keys_pressed_this_sec += 1

        # evaluate "commission errors"
        if self.is_current_target:
            self.total_correct_cumulative += 1
        else:
            self.errors_this_sec += 1
            self.trigger_error_flash()

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
            ]
        )
        self.csv_file.flush()  # force write to disk to prevent data loss

        # reset counters for the next second of data collection
        self.keys_pressed_this_sec = 0
        self.errors_this_sec = 0

        # reschedule this function to run exactly 1000ms from now
        self.root.after(1000, self.log_telemetry)

    def end_test(self):
        """Halts test, closes the CSV safely, and shows final accuracy."""
        self.is_running = False
        self.csv_file.close()

        accuracy = (
            (self.total_correct_cumulative / self.total_targets_shown * 100)
            if self.total_targets_shown > 0
            else 0
        )

        self.letter_label.config(text="TEST COMPLETE", font=("Helvetica", 64, "bold"))
        self.info_label.config(
            text=f"total Targets: {self.total_targets_shown} | correct Hits: {self.total_correct_cumulative} | accuracy: {accuracy:.1f}%\n\n"
            f"data saved to {CSV_OUTPUT_FILE}."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = NBackTestApp(root)
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)
    root.mainloop()
