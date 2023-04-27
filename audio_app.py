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
        self.options = {"Live Audio": ["1400", "5000", "10000", "20000"], "Play Music": ["test1.wav", "test2.wav", "test3.wav", "test4.wav", "5kHz.wav"]}
        self.selected_option = tk.StringVar(master)
        self.selected_option.set(self.options[self.audio_options[0]][0])
        self.frequency_dropdown_menu = ttk.Combobox(master, values=self.options[self.audio_options[0]], textvariable=self.selected_option, state='readonly')
        self.frequency_dropdown_menu.pack()

        # Create Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_options)
        self.submit_button.pack()
        
        # Quit Button
        self.quit_button = tk.Button(master, text="Quit", command=self.quit_options)
        self.quit_button.pack()

        # Set window size
        master.geometry("400x300")
        
        # Add trace callback to update frequency dropdown menu options
        self.selected_audio_option.trace("w", self.update_frequency_options)
        
        # Bind the close event to a method that kills the subprocess
        master.protocol("WM_DELETE_WINDOW", self.kill_subprocess)
        
        self.subprocess = None
        
    def update_frequency_options(self, *args):
        selected_audio_option = self.selected_audio_option.get()
        self.frequency_dropdown_menu.config(values=self.options[selected_audio_option])
        self.selected_option.set(self.options[selected_audio_option][0])

    def submit_options(self):
        audio_option = self.selected_audio_option.get()
        selected_option = self.selected_option.get()
        #print(f"Selected audio option: {audio_option}")
        #print(f"Selected frequency option: {selected_option}")
        if audio_option == "Live Audio":
            cmd = ['sudo', 'python', 'audio-visualizer-led.py', selected_option]
            self.subprocess = subprocess.Popen(cmd)
        elif audio_option == "Play Music":
            cmd1 = ['sudo', 'python', 'audio-visualizer-led-output.py', selected_option]
            proc = subprocess.Popen(cmd1)
            self.subprocesses.append(proc)
            cmd2 = ['python', 'audio-output.py', selected_option]
            proc = subprocess.Popen(cmd2)
            self.subprocesses.append(proc)
            
    def quit_options(self):
        if self.subprocess is not None:
            self.subprocess.kill()
            
    def kill_subprocess(self):
        if self.subprocess is not None:
            self.subprocess.kill()
        self.master.destroy()

root = tk.Tk()
app = App(root)
root.mainloop()
