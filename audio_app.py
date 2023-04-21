import tkinter as tk
from tkinter import ttk
import subprocess

class App:
    def __init__(self, master):
        self.master = master
        master.title("Audio App")

        # Create Live Audio and Play Music dropdown menu
        self.audio_options = ["Live Audio", "Play Music"]
        self.selected_audio_option = tk.StringVar(master)
        self.selected_audio_option.set(self.audio_options[0])
        self.audio_dropdown_menu = ttk.Combobox(master, values=self.audio_options, textvariable=self.selected_audio_option, state='readonly')
        self.audio_dropdown_menu.pack()

        # Create frequency dropdown menu
        self.options = ["1.4k", "5k", "10k", "20k"]
        self.selected_option = tk.StringVar(master)
        self.selected_option.set(self.options[0])
        self.frequency_dropdown_menu = ttk.Combobox(master, values=self.options, textvariable=self.selected_option, state='readonly')
        self.frequency_dropdown_menu.pack()

        # Create Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_options)
        self.submit_button.pack()

        # Set window size
        master.geometry("400x300")

    def submit_options(self):
        audio_option = self.selected_audio_option.get()
        selected_option = self.selected_option.get()
        #print(f"Selected audio option: {audio_option}")
        #print(f"Selected frequency option: {selected_option}")
        if audio_option == "Live Audio":
            subprocess.run(['python', 'test1.py', audio_option, selected_option])

root = tk.Tk()
app = App(root)
root.mainloop()

