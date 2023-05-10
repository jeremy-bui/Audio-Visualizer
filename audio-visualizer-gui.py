from tkinter import ttk

import tkinter as tk
import subprocess
import os
import signal

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
        
        self.subprocesses = []
        
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
            self.cmd = ['sudo', 'python', 'audio-visualizer-led.py', selected_option]
            proc = subprocess.Popen(self.cmd)
            self.subprocess_pid = proc.pid
            self.subprocesses.append(self.subprocess_pid)
            print("GUI Subprocess PID: ", self.subprocess_pid)
        elif audio_option == "Play Music":
            cmd1 = ['sudo', 'python', 'audio-visualizer-led-output.py', selected_option]
            proc = subprocess.Popen(cmd1)
            self.subprocess_pid = proc.pid
            self.subprocesses.append(self.subprocess_pid)
            cmd2 = ['python', 'audio-output.py', selected_option]
            proc = subprocess.Popen(cmd2)
            self.subprocess_pid = proc.pid
            self.subprocesses.append(self.subprocess_pid)

            
    def stop_subprocess(self):
        if len(self.subprocesses) == 1:
            os.kill(self.subprocess_pid + 1, signal.SIGTERM)
        elif len(self.subprocesses) == 2:
            for proc_id in self.subprocesses:
                os.kill(proc_id + 1, signal.SIGTERM)
                
        self.subprocesses = []
            
    def kill_subprocess(self):
        self.stop_subprocess()
        self.master.destroy()
        
    def quit_options(self):
        self.stop_subprocess()

root = tk.Tk()
app = App(root)
root.mainloop()
