import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ALMDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Atomic Language Model - Anomaly Detection Demo")
        self.root.geometry("800x600")

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(self.control_frame, text="Start Demo", command=self.start_demo)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.results_text = tk.Text(self.main_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

    def plot_data(self, data, anomalies=None):
        self.ax.clear()
        self.ax.plot(data, label="Bearing Data")
        if anomalies:
            self.ax.scatter(anomalies, data[anomalies], color='red', label="Anomalies")
        self.ax.legend()
        self.canvas.draw()

    def start_demo(self):
        self.log("Starting ALM Anomaly Detection Demo...")
        try:
            # Load the dataset
            # Since we can't unzip, we'll assume the data is in a CSV file
            # named 'bearing.csv' in the 'data' directory.
            # This will fail, but it's a placeholder for the actual data loading.
            data = pd.read_csv("nasa_demo/data/labeled_anomalies.csv")
            self.log("Dataset loaded successfully.")
        except FileNotFoundError:
            self.log("Error: Could not find 'labeled_anomalies.csv'.")
            self.log("Please make sure the bearing dataset is unzipped and in the 'nasa_demo/data' directory.")
            return

        # Simplified ALM validation logic
        def validate_syntax(sequence):
            # This is a simplified grammar for anomaly detection.
            # A sequence is "grammatical" if the standard deviation is within a threshold.
            threshold = 0.1
            if np.std(sequence) > threshold:
                return False  # "Ungrammatical" - anomaly
            return True  # "Grammatical" - normal

        # Process the data
        window_size = 100
        anomalies = []
        for i in range(len(data) - window_size):
            window = data['value'].iloc[i:i+window_size].values
            if not validate_syntax(window):
                anomalies.append(i + window_size // 2)
                self.log(f"Anomaly detected at index {i + window_size // 2}")

        self.log("Demo finished.")
        self.plot_data(data['value'].values, anomalies)

if __name__ == "__main__":
    root = tk.Tk()
    app = ALMDemo(root)
    root.mainloop()
