import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.players.MCTS.Treeplayer import TreePlayer, Node



# class GridConfigApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Grid Configuration Tool")
#         self.geometry("1080x720")
        
#         self.protocol("WM_DELETE_WINDOW", self.exit)

#         # Initialize the user interface
#         self.create_widgets()
#         self.create_plot()
        
#     def create_widgets(self):
#         self.frame = ttk.Frame(self)
        
#     def create_plot(self):
        