import tkinter as tk
import tkinter.messagebox
from typing import Union, Callable
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
import math
import cmath
import pandas as pd


customtkinter.set_appearance_mode("Dark")

customtkinter.set_default_color_theme("blue")

""" DataFrame Rectangular Plate for beta and alpha"""

columns_RP = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, np.inf]
data_RP = [[0.2874, 0.3762, 0.4530, 0.5172, 0.5688, 0.6102, 0.7134, 0.7410, 0.7476, 0.7500],
           [0.0444, 0.0616, 0.0770, 0.0906, 0.1017, 0.1110, 0.1335, 0.1400, 0.1417, 0.1421]]
index_RP = ['beta', 'alpha']
df_RP = pd.DataFrame(data_RP, columns=columns_RP, index=index_RP)

""" DataFrame Peak Velocity Pressure"""

columns_PVP = pd.MultiIndex.from_tuples(
    [('Area I', 'Coastal'), ('Area I', 'Rural'), ('Area I', 'Urban'),
     ('Area II', 'Coastal'), ('Area II', 'Rural'), ('Area II', 'Urban'),
     ('Area III', 'Rural'), ('Area III', 'Urban')],
    names=['Area', 'Type'])
index_PVP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40,
             45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150]

data_PVP = np.array([
    [0.93, 0.71, 0.69, 0.78, 0.60, 0.58, 0.49, 0.48],
    [1.11, 0.71, 0.69, 0.93, 0.60, 0.58, 0.49, 0.48],
    [1.22, 0.71, 0.69, 1.02, 0.60, 0.58, 0.49, 0.48],
    [1.30, 0.71, 0.69, 1.09, 0.60, 0.58, 0.49, 0.48],
    [1.37, 0.78, 0.69, 1.14, 0.66, 0.58, 0.54, 0.48],
    [1.42, 0.84, 0.69, 1.19, 0.71, 0.58, 0.58, 0.48],
    [1.47, 0.89, 0.69, 1.23, 0.75, 0.58, 0.62, 0.48],
    [1.51, 0.94, 0.73, 1.26, 0.79, 0.62, 0.65, 0.51],
    [1.55, 0.98, 0.77, 1.29, 0.82, 0.65, 0.68, 0.53],
    [1.58, 1.02, 0.81, 1.32, 0.85, 0.68, 0.70, 0.56],
    [1.71, 1.16, 0.96, 1.43, 0.98, 0.80, 0.80, 0.66],
    [1.80, 1.27, 1.07, 1.51, 1.07, 0.90, 0.88, 0.74],
    [1.88, 1.36, 1.16, 1.57, 1.14, 0.97, 0.94, 0.80],
    [1.94, 1.43, 1.23, 1.63, 1.20, 1.03, 0.99, 0.85],
    [2.00, 1.50, 1.30, 1.67, 1.25, 1.09, 1.03, 0.89],
    [2.04, 1.55, 1.35, 1.71, 1.30, 1.13, 1.07, 0.93],
    [2.09, 1.60, 1.40, 1.75, 1.34, 1.17, 1.11, 0.97],
    [2.12, 1.65, 1.45, 1.78, 1.38, 1.21, 1.14, 1.00],
    [2.16, 1.69, 1.49, 1.81, 1.42, 1.25, 1.17, 1.03],
    [2.19, 1.73, 1.53, 1.83, 1.45, 1.28, 1.19, 1.05],
    [2.22, 1.76, 1.57, 1.86, 1.48, 1.31, 1.22, 1.08],
    [2.25, 1.80, 1.60, 1.88, 1.50, 1.34, 1.24, 1.10],
    [2.27, 1.83, 1.63, 1.90, 1.53, 1.37, 1.26, 1.13],
    [2.30, 1.86, 1.66, 1.92, 1.55, 1.39, 1.28, 1.15],
    [2.32, 1.88, 1.69, 1.94, 1.58, 1.42, 1.30, 1.17],
    [2.34, 1.91, 1.72, 1.96, 1.60, 1.44, 1.32, 1.18],
    [2.36, 1.93, 1.74, 1.98, 1.62, 1.46, 1.33, 1.20],
    [2.38, 1.96, 1.77, 1.99, 1.64, 1.48, 1.35, 1.22],
    [2.42, 2.00, 1.81, 2.03, 1.68, 1.52, 1.38, 1.25],
    [2.45, 2.04, 1.85, 2.05, 1.71, 1.55, 1.41, 1.28],
    [2.48, 2.08, 1.89, 2.08, 1.74, 1.59, 1.44, 1.31],
    [2.51, 2.12, 1.93, 2.10, 1.77, 1.62, 1.46, 1.33],
    [2.54, 2.15, 1.96, 2.13, 1.80, 1.65, 1.48, 1.35]])
df_PVP = pd.DataFrame(data=data_PVP, index=index_PVP, columns=columns_PVP)


class FloatSpinbox(customtkinter.CTkFrame):
    """
    A class used to create a spinbox for floating point number selection.

    Attributes
    ----------
    master : object
       The parent widget.
    width : int
       The width of the spinbox.
    height : int
       The height of the spinbox.
    step_size : int or float
       The step size for the spinbox.
    command : Callable
       The function to call when the spinbox value changes.
    """
    def __init__(self, *args, width: int = 100, height: int = 32,
                 step_size: Union[int, float] = 1, command: Callable = None, **kwargs):
        """
        Initialize the FloatSpinbox object, setting up the initial state of the object,
        including the creation of buttons and entry widget.
        """
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = step_size
        self.command = command

        self.configure(fg_color=("gray78", "gray28"))
        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.subtract_button = customtkinter.CTkButton(
            self, text="-", width=height - 6, height=height - 6, command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = customtkinter.CTkEntry(
            self, width=width - (2 * height), height=height - 6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1,
                        padx=3, pady=3, sticky="ew")

        self.add_button = customtkinter.CTkButton(
            self, text="+", width=height - 6, height=height - 6, command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        self.entry.insert(0, "0.0")

    def add_button_callback(self):
        """
        Increase the spinbox value by the step size and call the command function.
        This method is called when the add button is clicked.
        """
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) + self.step_size
            self.entry.delete(0, "end")
            self.entry.insert(0, "{:.2f}".format(value))
        except ValueError:
            return

    def subtract_button_callback(self):
        """
        Decrease the spinbox value by the step size and call the command function.
        This method is called when the subtract button is clicked.
        """
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) - self.step_size
            self.entry.delete(0, "end")
            self.entry.insert(0, "{:.2f}".format(value))
        except ValueError:
            return

    def get(self) -> Union[float, None]:
        """
        Get the current value of the spinbox.

        :return: float or None
                    The current value of the spinbox, or None if the value is not a valid float.
        """
        try:
            return float(self.entry.get())
        except ValueError:
            return None

    def set(self, value: float):
        """
        Set the current value of the spinbox.

        :param value: float
                The new value of the spinbox.
        """
        self.entry.delete(0, "end")
        self.entry.insert(0, "{:.2f}".format(value))


class StructureHeightSpinBox(customtkinter.CTkFrame):
    """
     A class used to create a spinbox for structure height selection.

    Attributes
    ----------
    master : object
        The parent widget.
    width : int
        The width of the spinbox.
    height : int
        The height of the spinbox.
    values : list
        The list of values that the spinbox can take.
    command : Callable
        The function to call when the spinbox value changes.
    """
    def __init__(self, *args, width: int = 100, height: int = 32,
                 values: list = None, command: Callable = None, **kwargs):
        """
        Initialize the StructureHeightSpinBox object, setting up the initial state of the object,
        including the creation of buttons and entry widget.
        """
        super().__init__(*args, width=width, height=height, **kwargs)

        self.values = values or []
        self.command = command

        self.configure(fg_color=("gray78", "gray28"))
        self.grid_columnconfigure((0, 1, 3, 4), weight=0)
        self.grid_columnconfigure(2, weight=1)

        self.jump_subtract_button = customtkinter.CTkButton(
            self, text="- -", width=height - 6, height=height - 6, command=self.jump_subtract_button_callback)
        self.jump_subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.subtract_button = customtkinter.CTkButton(
            self, text="-", width=height - 6, height=height - 6, command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=1, padx=(3, 0), pady=3)

        self.entry_var = tk.StringVar(self)
        self.entry = customtkinter.CTkEntry(self, width=width - (2 * height), height=height - 6, border_width=0,
                                            state='disabled', textvariable=self.entry_var)
        self.entry.grid(row=0, column=2, columnspan=1,
                        padx=3, pady=3, sticky="ew")

        self.add_button = customtkinter.CTkButton(
            self, text="+", width=height - 6, height=height - 6, command=self.add_button_callback)
        self.add_button.grid(row=0, column=3, padx=(0, 3), pady=3)

        self.jump_add_button = customtkinter.CTkButton(
            self, text="+ +", width=height - 6, height=height - 6, command=self.jump_add_button_callback)
        self.jump_add_button.grid(row=0, column=4, padx=(0, 3), pady=3)

        self.entry_var.set("50.0")

        self.last_button_states = {
            'subtract': 'disabled',
            'jump_subtract': 'disabled',
            'add': 'disabled',
            'jump_add': 'disabled'
        }

        self.update_button_states()

    def update_button_states(self):
        """
        Update the states of the buttons based on the current value of the spinbox.
        """
        current_value = float(self.entry_var.get())
        idx = self.values.index(current_value)

        new_button_states = {
            'subtract': 'normal' if idx > 0 else 'disabled',
            'jump_subtract': 'normal' if idx >= 5 else 'disabled',
            'add': 'normal' if idx < len(self.values) - 1 else 'disabled',
            'jump_add': 'normal' if idx < len(self.values) - 5 else 'disabled'
        }

        if new_button_states != self.last_button_states:
            self.subtract_button.configure(state=new_button_states['subtract'])
            self.jump_subtract_button.configure(
                state=new_button_states['jump_subtract'])
            self.add_button.configure(state=new_button_states['add'])
            self.jump_add_button.configure(state=new_button_states['jump_add'])

            self.last_button_states = new_button_states

    def jump_subtract_button_callback(self):
        """
        Decrease the spinbox value by 5 steps and call the command function.
        This method is called when the jump subtract button is clicked.
        """
        current_value = float(self.entry_var.get())
        if current_value in self.values:
            idx = self.values.index(current_value)
            if idx >= 5:
                self.entry_var.set("{:.1f}".format(self.values[idx - 5]))
                if self.command:
                    self.command()
        self.update_button_states()

    def subtract_button_callback(self):
        """
        Decrease the spinbox value by 1 step and call the command function.
        This method is called when the subtract button is clicked.
        """
        current_value = float(self.entry_var.get())
        if current_value in self.values:
            idx = self.values.index(current_value)
            if idx > 0:
                self.entry_var.set("{:.1f}".format(self.values[idx - 1]))
                if self.command:
                    self.command()
        self.update_button_states()

    def add_button_callback(self):
        """
        Increase the spinbox value by 1 step and call the command function.
        This method is called when the add button is clicked.
        """
        current_value = float(self.entry_var.get())
        if current_value in self.values:
            idx = self.values.index(current_value)
            if idx < len(self.values) - 1:
                self.entry_var.set("{:.1f}".format(self.values[idx + 1]))
                if self.command:
                    self.command()
        self.update_button_states()

    def jump_add_button_callback(self):
        """
        Increase the spinbox value by 5 steps and call the command function.
        This method is called when the jump add button is clicked.
        """
        current_value = float(self.entry_var.get())
        if current_value in self.values:
            idx = self.values.index(current_value)
            if idx < len(self.values) - 5:
                self.entry_var.set("{:.1f}".format(self.values[idx + 5]))
                if self.command:
                    self.command()
        self.update_button_states()

    def get(self) -> Union[float, None]:
        """
        Get the value of the spinbox
        :return: float or None
                    The current value of the spinbox, or None if the value is not a float.
        """
        try:
            return float(self.entry.get())
        except ValueError:
            return None


class DimensionWindow(ctk.CTkFrame):
    """
    A class used to generate a dynamic window with sliders for the length and width of the window.
    It includes methods to update the axes, configure the dimensions of the frame, update the image,
    and calculate the beta, alpha and k_a.

    Attributes
    ----------
    master : object
        The parent widget.
    df : DataFrame
        The DataFrame used to define alpha and beta.
    unity_check_frame : object
        The frame in which the unity check is performed.
    """
    def __init__(self, master, df, unity_check_frame):
        """
        Initialize the DimensionWindow object, setting up the initial state of the object,
        including the creation of a matplotlib figure and axes, a rectangle patch and two sliders
        for width and height. It also calculates initial alpha and beta values and sets up an event
        binding for window configuration changes.
        """
        super().__init__(master)

        self.df = df
        self.unity_check_frame = unity_check_frame

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.max_dimension_x = 1200
        self.max_dimension_y = 1500

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(0, self.max_dimension_x)
        self.ax.set_ylim(0, self.max_dimension_y)
        self.ax.set_facecolor('lightslategray')
        self.ax.set_aspect('equal')

        self.rect = plt.Rectangle(
            (0, 0), self.max_dimension_x / 2, self.max_dimension_y / 2, fc='lightsteelblue')
        self.ax.add_patch(self.rect)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")


        self.width_slider = tk.Scale(self, from_=0, to=self.max_dimension_x, length=400,
                                     command=self.update_rectangle, orient='horizontal')

        self.width_slider.set(self.max_dimension_x / 2)
        self.width_slider.grid(row=1, column=0, sticky="ew")

        self.height_slider = tk.Scale(self, from_=self.max_dimension_y, to=0, length=400,
                                      command=self.update_rectangle, orient='vertical')

        self.height_slider.set(self.max_dimension_y / 2)
        self.height_slider.grid(row=0, column=1, sticky="ns")

        self.alpha, self.beta = self.calculate_ratio_and_get_alpha_beta(
            self.height_slider.get(), self.width_slider.get())
        self.unity_check_frame.set_SliderValues(
            self.height_slider.get(), self.width_slider.get())
        self.unity_check_frame.set_coefficients(self.alpha, self.beta)

        self.master.bind("<Configure>", self.on_window_configure)

    def on_window_configure(self, event):
        """
        Adjust the size of the canvas and the sliders to match the new window size,
        and update the axes limits and rectangle dimensions accordingly. This method is called
        when the window is resized.
        """
        if self.master.state() == "zoomed":
            new_width = self.master.winfo_width()
            new_height = self.master.winfo_height() - 170

            self.canvas.get_tk_widget().configure(width=new_width, height=new_height)
            self.width_slider.configure(length=new_width)
            self.height_slider.configure(length=new_height)

            self.update_axes_limits()
            self.update_rectangle(None)

    def update_axes_limits(self):
        """
        Set the limits of the x and y axes to the maximum dimensions.
        """
        self.ax.set_xlim(0, self.max_dimension_x)
        self.ax.set_ylim(0, self.max_dimension_y)

    def update_rectangle(self, _):
        """
        Adjust the width and height of the rectangle, redraw the canvas, recalculate alpha and beta,
        update the axes limits and update the slider values and coefficients in the unity_check_frame.
        This method is called when a slider value changes.
        """
        width = self.width_slider.get()
        height = self.height_slider.get()

        if width > height:
            width = height
            self.width_slider.set(width)

        self.rect.set_width(width)
        self.rect.set_height(height)
        self.fig.canvas.draw_idle()
        self.alpha, self.beta = self.calculate_ratio_and_get_alpha_beta(
            height, width)
        self.update_axes_limits()
        self.unity_check_frame.set_SliderValues(height, width)
        self.unity_check_frame.set_coefficients(self.alpha, self.beta)
        self.unity_check_frame.compute_k_a()

    def calculate_ratio_and_get_alpha_beta(self, length, width):
        """
        Calculate the ratio of length to width and find the closest column in the dataframe to this ratio.
        Retrieve the alpha and beta values from this column and return them.
        :param length: the length of the window set by the length slider
        :param width: the width of the window set by the width slider
        :return: the alpha and the beta used in the UnityCheck class
        """
        ratio = length / width if width != 0 else float('inf')

        closest_column = min(self.df.columns,
                             key=lambda x: abs(x - ratio) if x != 'infinity' else abs(float('inf') - ratio))

        alpha = self.df.loc['alpha', closest_column]
        beta = self.df.loc['beta', closest_column]

        return alpha, beta


class WindowParameters(customtkinter.CTkFrame):
    """
    A class used to create a window for setting parameters of a window pane.

    Attributes
    ----------
    master : object
        The parent widget.
    dimension_window : object
        The window for setting dimensions.
    unity_check_frame : object
        The frame for checking unity.
    """
    def __init__(self, master, dimension_window, unity_check_frame):
        super().__init__(master)
        """
        Initialize the WindowParameters object, setting up the initial state of the object, 
        including the creation of labels, entries and buttons. 
        """
        self.master = master
        self.dimension_window = dimension_window
        self.unity_check_frame = unity_check_frame

        self.identical_panes = tkinter.BooleanVar()

        self.outer_strength_var = tkinter.StringVar()
        self.outer_thickness_var = tkinter.StringVar()
        self.outer_height_pane_var = tkinter.StringVar()
        self.outer_width_pane_var = tkinter.StringVar()

        self.label_title = customtkinter.CTkLabel(self,
                                                  text="Glass-Parameters",
                                                  fg_color="gray28",
                                                  corner_radius=5)
        self.label_title.grid(row=0, column=0, columnspan=2)

        # Outer pane
        self.frame_outer_pane = customtkinter.CTkFrame(self)
        self.frame_outer_pane.grid(
            row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.frame_inner_pane = customtkinter.CTkFrame(self)
        self.frame_inner_pane.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.frame_extra_params = customtkinter.CTkFrame(self)
        self.frame_extra_params.grid(
            row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.update_values_button = customtkinter.CTkButton(
            self, text="Update Values", command=self.update_values)
        self.update_values_button.grid(row=5, column=0)

        self.reset_button = customtkinter.CTkButton(
            self, text="Reset", command=self.reset_values)
        self.reset_button.grid(row=5, column=1)


        self.label_outer_pane = customtkinter.CTkLabel(self.frame_outer_pane,
                                                       text="Outer Pane Window",
                                                       fg_color="gray28",
                                                       corner_radius=5)

        self.label_outer_pane.grid(row=0, column=0, columnspan=3)

        self.outer_strength_label = customtkinter.CTkLabel(
            self.frame_outer_pane, text="Strength:")
        self.outer_strength_label.grid(
            row=1, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_strength_entry = customtkinter.CTkEntry(self.frame_outer_pane,
                                                           textvariable=self.outer_strength_var)
        self.outer_strength_entry.configure(validate="key", validatecommand=(
            self.outer_strength_entry.register(self.validate_entry), '%P'))
        self.outer_strength_entry.grid(
            row=1, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_strength_unit = customtkinter.CTkLabel(
            self.frame_outer_pane, text="[N/mm^2]")
        self.outer_strength_unit.grid(
            row=1, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.outer_thickness_label = customtkinter.CTkLabel(
            self.frame_outer_pane, text="Thickness:")
        self.outer_thickness_label.grid(
            row=2, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_thickness_entry = customtkinter.CTkEntry(self.frame_outer_pane,
                                                            textvariable=self.outer_thickness_var)
        self.outer_thickness_entry.configure(validate="key", validatecommand=(
            self.outer_thickness_entry.register(self.validate_entry), '%P'))
        self.outer_thickness_entry.grid(
            row=2, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_thickness_unit = customtkinter.CTkLabel(
            self.frame_outer_pane, text="[mm]")
        self.outer_thickness_unit.grid(
            row=2, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.outer_height_pane_label = customtkinter.CTkLabel(
            self.frame_outer_pane, text="Height Pane:")
        self.outer_height_pane_label.grid(
            row=3, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_height_pane_entry = customtkinter.CTkEntry(self.frame_outer_pane,
                                                              textvariable=self.outer_height_pane_var)
        self.outer_height_pane_entry.configure(validate="key", validatecommand=(
            self.outer_height_pane_entry.register(self.validate_entry), '%P'))

        self.outer_height_pane_entry.grid(
            row=3, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_height_pane_unit = customtkinter.CTkLabel(
            self.frame_outer_pane, text="[mm]")
        self.outer_height_pane_unit.grid(
            row=3, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.outer_width_pane_label = customtkinter.CTkLabel(
            self.frame_outer_pane, text="Width Pane:")
        self.outer_width_pane_label.grid(
            row=4, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_width_pane_entry = customtkinter.CTkEntry(self.frame_outer_pane,
                                                             textvariable=self.outer_width_pane_var)
        self.outer_width_pane_entry.configure(validate="key", validatecommand=(
            self.outer_width_pane_entry.register(self.validate_entry), '%P'))
        self.outer_width_pane_entry.grid(
            row=4, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.outer_width_pane_unit = customtkinter.CTkLabel(
            self.frame_outer_pane, text="[mm]")
        self.outer_width_pane_unit.grid(
            row=4, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        # Variables for inner pane
        self.inner_strength_var = tkinter.StringVar()
        self.inner_thickness_var = tkinter.StringVar()
        self.inner_height_pane_var = tkinter.StringVar()
        self.inner_width_pane_var = tkinter.StringVar()

        self.label_inner_pane = customtkinter.CTkLabel(self.frame_inner_pane,
                                                       text="Inner Pane Window",
                                                       fg_color="gray28",
                                                       corner_radius=5)
        self.label_inner_pane.grid(row=0, column=0, columnspan=3)

        self.inner_strength_label = customtkinter.CTkLabel(
            self.frame_inner_pane, text="Strength:")
        self.inner_strength_label.grid(
            row=1, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_strength_entry = customtkinter.CTkEntry(self.frame_inner_pane,
                                                           textvariable=self.inner_strength_var)
        self.inner_strength_entry.configure(validate="key", validatecommand=(
            self.inner_strength_entry.register(self.validate_entry), '%P'))
        self.inner_strength_entry.grid(
            row=1, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_strength_unit = customtkinter.CTkLabel(
            self.frame_inner_pane, text="[N/mm^2]")
        self.inner_strength_unit.grid(
            row=1, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.inner_thickness_label = customtkinter.CTkLabel(
            self.frame_inner_pane, text="Thickness:")
        self.inner_thickness_label.grid(
            row=2, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_thickness_entry = customtkinter.CTkEntry(self.frame_inner_pane,
                                                            textvariable=self.inner_thickness_var)
        self.inner_thickness_entry.configure(validate="key", validatecommand=(
            self.inner_thickness_entry.register(self.validate_entry), '%P'))
        self.inner_thickness_entry.grid(
            row=2, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_thickness_unit = customtkinter.CTkLabel(
            self.frame_inner_pane, text="[mm]")
        self.inner_thickness_unit.grid(
            row=2, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.inner_height_pane_label = customtkinter.CTkLabel(
            self.frame_inner_pane, text="Height Pane:")
        self.inner_height_pane_label.grid(
            row=3, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_height_pane_entry = customtkinter.CTkEntry(self.frame_inner_pane,
                                                              textvariable=self.inner_height_pane_var)
        self.inner_height_pane_entry.configure(validate="key", validatecommand=(
            self.inner_height_pane_entry.register(self.validate_entry), '%P'))
        self.inner_height_pane_entry.grid(
            row=3, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_height_pane_unit = customtkinter.CTkLabel(
            self.frame_inner_pane, text="[mm]")
        self.inner_height_pane_unit.grid(
            row=3, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.inner_width_pane_label = customtkinter.CTkLabel(
            self.frame_inner_pane, text="Width Pane:")
        self.inner_width_pane_label.grid(
            row=4, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_width_pane_entry = customtkinter.CTkEntry(self.frame_inner_pane,
                                                             textvariable=self.inner_width_pane_var)
        self.inner_width_pane_entry.configure(validate="key", validatecommand=(
            self.inner_width_pane_entry.register(self.validate_entry), '%P'))
        self.inner_width_pane_entry.grid(
            row=4, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.inner_width_pane_unit = customtkinter.CTkLabel(
            self.frame_inner_pane, text="[mm]")
        self.inner_width_pane_unit.grid(
            row=4, column=2, sticky="w", padx=(5, 5), pady=(5, 5))


        self.width_gap_var = tkinter.StringVar()
        self.width_gap_var.set("12")
        self.k_a_var = tkinter.StringVar()
        self.k_a_var.set("1")
        self.k_e_var = tkinter.StringVar()
        self.k_e_var.set("1")
        self.k_mod_var = tkinter.StringVar()
        self.k_mod_var.set("1")
        self.k_sp_var = tkinter.StringVar()
        self.k_sp_var.set("1")
        self.gamma_var = tkinter.StringVar()
        self.gamma_var.set("1.6")

        self.label_extra_params = customtkinter.CTkLabel(self.frame_extra_params,
                                                         text="Extra Parameters",
                                                         fg_color="gray28",
                                                         corner_radius=5)
        self.label_extra_params.grid(row=0, column=0, columnspan=3)

        self.width_gap_label = customtkinter.CTkLabel(
            self.frame_extra_params, text="Width Gap:")
        self.width_gap_label.grid(
            row=1, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.width_gap_entry = customtkinter.CTkEntry(self.frame_extra_params,
                                                      textvariable=self.width_gap_var)
        self.width_gap_entry.configure(validate="key", validatecommand=(
            self.width_gap_entry.register(self.validate_entry), '%P'))

        self.width_gap_entry.grid(
            row=1, column=1, sticky="w", padx=(5, 5), pady=(5, 5))
        self.width_gap_unit = customtkinter.CTkLabel(
            self.frame_extra_params, text="[mm]")
        self.width_gap_unit.grid(
            row=1, column=2, sticky="w", padx=(5, 5), pady=(5, 5))

        self.k_a_label = customtkinter.CTkLabel(
            self.frame_extra_params, text="k_a:")
        self.k_a_label.grid(row=2, column=0, sticky="w",
                            padx=(5, 5), pady=(5, 5))
        self.k_a_entry = customtkinter.CTkEntry(self.frame_extra_params,
                                                textvariable=self.k_a_var)
        self.k_a_entry.configure(validate="key", validatecommand=(
            self.k_a_entry.register(self.validate_entry), '%P'))

        self.k_a_entry.grid(row=2, column=1, sticky="w",
                            padx=(5, 5), pady=(5, 5))

        self.k_e_label = customtkinter.CTkLabel(
            self.frame_extra_params, text="k_e:")
        self.k_e_label.grid(row=3, column=0, sticky="w",
                            padx=(5, 5), pady=(5, 5))
        self.k_e_entry = customtkinter.CTkEntry(
            self.frame_extra_params, textvariable=self.k_e_var)
        self.k_e_entry.configure(validate="key", validatecommand=(
            self.k_e_entry.register(self.validate_entry), '%P'))

        self.k_e_entry.grid(row=3, column=1, sticky="w",
                            padx=(5, 5), pady=(5, 5))

        self.k_mod_label = customtkinter.CTkLabel(
            self.frame_extra_params, text="k_mod:")
        self.k_mod_label.grid(row=4, column=0, sticky="w",
                              padx=(5, 5), pady=(5, 5))
        self.k_mod_entry = customtkinter.CTkEntry(
            self.frame_extra_params, textvariable=self.k_mod_var)
        self.k_mod_entry.configure(validate="key", validatecommand=(
            self.k_mod_entry.register(self.validate_entry), '%P'))

        self.k_mod_entry.grid(row=4, column=1, sticky="w",
                              padx=(5, 5), pady=(5, 5))

        self.k_sp_label = customtkinter.CTkLabel(
            self.frame_extra_params, text="k_sp:")
        self.k_sp_label.grid(row=5, column=0, sticky="w",
                             padx=(5, 5), pady=(5, 5))
        self.k_sp_entry = customtkinter.CTkEntry(
            self.frame_extra_params, textvariable=self.k_sp_var)
        self.k_sp_entry.configure(validate="key", validatecommand=(
            self.k_sp_entry.register(self.validate_entry), '%P'))

        self.k_sp_entry.grid(row=5, column=1, sticky="w",
                             padx=(5, 5), pady=(5, 5))

        self.gamma_label = customtkinter.CTkLabel(
            self.frame_extra_params, text="gamma:")
        self.gamma_label.grid(row=6, column=0, sticky="w",
                              padx=(5, 5), pady=(5, 5))
        self.gamma_entry = customtkinter.CTkEntry(
            self.frame_extra_params, textvariable=self.gamma_var)
        self.gamma_entry.configure(validate="key", validatecommand=(
            self.gamma_entry.register(self.validate_entry), '%P'))

        self.gamma_entry.grid(row=6, column=1, sticky="w",
                              padx=(5, 5), pady=(5, 5))

        self.check_box = customtkinter.CTkCheckBox(self,
                                                   text="Inner and Outer Pane Identical",
                                                   variable=self.identical_panes,
                                                   command=self.update_pane_values)
        self.check_box.grid(row=1, column=0, sticky="nsew")

        # Callbacks
        self.outer_strength_var.trace('w', self.update_pane_values)
        self.outer_thickness_var.trace('w', self.update_pane_values)
        self.outer_height_pane_var.trace('w', self.update_pane_values)
        self.outer_width_pane_var.trace('w', self.update_pane_values)
        self.unity_check_frame.register_callback(self.update_k_a_var)

    def update_k_a_var(self):
        """
        Update the k_a variable based on the unity check frame.
        """
        self.k_a_var.set("{:.2f}".format(self.unity_check_frame.k_a_var))

    def validate_entry(self, value):
        """
        Validate the entry value. It must be a numeric value

        :param value: str
                The value to validate
        :return: bool
                    True if the value is valid, False otherwise
        """
        if value == "":
            return True
        elif value.replace('.', '', 1).isdigit():
            return True
        else:
            tk.messagebox.showwarning(
                "Invalid Value", "Please enter a valid numeric value.")
            return False

    def reset_values(self):
        """
        Reset all the entry values to their default states.
        """
        self.outer_strength_var.set("")
        self.outer_thickness_var.set("")
        self.outer_height_pane_var.set("")
        self.outer_width_pane_var.set("")

        self.inner_strength_var.set("")
        self.inner_thickness_var.set("")
        self.inner_height_pane_var.set("")
        self.inner_width_pane_var.set("")

        self.width_gap_var.set("12")
        self.k_a_var.set("1")
        self.k_e_var.set("1")
        self.k_mod_var.set("1")
        self.k_sp_var.set("1")
        self.gamma_var.set("1.6")

    def update_pane_values(self, *args):
        """
        Update the pane values based on the identical panes checkbox.
        """
        if self.identical_panes.get():
            self.inner_strength_var.set(self.outer_strength_var.get())
            self.inner_thickness_var.set(self.outer_thickness_var.get())
            self.inner_height_pane_var.set(self.outer_height_pane_var.get())
            self.inner_width_pane_var.set(self.outer_width_pane_var.get())

            self.inner_strength_entry.configure(state='disabled')
            self.inner_thickness_entry.configure(state='disabled')
            self.inner_height_pane_entry.configure(state='disabled')
            self.inner_width_pane_entry.configure(state='disabled')
        else:
            self.inner_strength_entry.configure(state='normal')
            self.inner_thickness_entry.configure(state='normal')
            self.inner_height_pane_entry.configure(state='normal')
            self.inner_width_pane_entry.configure(state='normal')

    def is_Any_Empty_Or_Zero(self):
        """
        Check if any of the entry values are empty or zero.

        :return: str or bool
                The first variable that is empty or zero, or False if all variables are valid.
        """
        variables = [
            self.outer_strength_var.get(),
            self.outer_thickness_var.get(),
            self.outer_height_pane_var.get(),
            self.outer_width_pane_var.get(),
            self.inner_strength_var.get(),
            self.inner_thickness_var.get(),
            self.inner_height_pane_var.get(),
            self.inner_width_pane_var.get(),
            self.width_gap_var.get(),
            self.k_a_var.get(),
            self.k_e_var.get(),
            self.k_mod_var.get(),
            self.k_sp_var.get(),
            self.gamma_var.get()
        ]

        for variable in variables:
            if variable is None or len(variable.strip()) == 0 or float(variable) == 0:
                return variable
        return False

    def update_values(self):
        """
        Update the master values and the unity check frame based on the entry values.
        """
        self.master.values = {
            "outer_strength": self.outer_strength_var.get(),
            "outer_thickness": self.outer_thickness_var.get(),
            "outer_height_pane": self.outer_height_pane_var.get(),
            "outer_width_pane": self.outer_width_pane_var.get(),

            "inner_strength": self.inner_strength_var.get(),
            "inner_thickness": self.inner_thickness_var.get(),
            "inner_height_pane": self.inner_height_pane_var.get(),
            "inner_width_pane": self.inner_width_pane_var.get(),

            "width_gap": self.width_gap_var.get()
        }
        self.k_a_entry.configure(validate="none")
        self.k_a_var.set("{:.2f}".format(self.unity_check_frame.compute_k_a()))
        self.k_a_entry.configure(validate="key")

        flag = self.is_Any_Empty_Or_Zero()
        if (flag != False):
            tk.messagebox.showwarning("Invalid Value",
                                      "Please enter a non-zero or valid value of all    missing variables: " + flag)
        self.unity_check_frame.set_windows_values(self.master.values)
        self.unity_check_frame.set_Extra_Params(self.k_a_var.get(), self.k_e_var.get(
        ), self.k_mod_var.get(), self.k_sp_var.get(), self.gamma_var.get())
        self.dimension_window.max_dimension_x = min(float(self.outer_width_pane_var.get()),
                                                    float(self.inner_width_pane_var.get()))
        self.dimension_window.max_dimension_y = min(float(self.outer_height_pane_var.get()),
                                                    float(self.inner_height_pane_var.get()))

        self.dimension_window.width_slider.configure(
            to=self.dimension_window.max_dimension_x)
        self.dimension_window.height_slider.configure(
            from_=self.dimension_window.max_dimension_y)

        self.dimension_window.update_axes_limits()


class WindForce(customtkinter.CTkFrame):
    """
       A class used to set the parameters for the wind-pressure and calculate the adapted wind-pressure with the set
       parameters.

       Attributes
       ----------
       master : object
           The parent widget.
       unity_check_frame : object
           The frame in which the unity check is performed.
       """
    def __init__(self, master, unity_check_frame):
        """
        Initialize the WindForce object, setting up the initial state of the object,
        including the creation of labels, option menus, spinboxes and buttons.
        It also sets up an event binding for window configuration changes.
        """
        super().__init__(master)
        self.tab_data_dict = None

        self.wind_face_var = tk.StringVar()
        self.safety_factor_var = tk.StringVar()
        self.safety_factor_var.set("1.5")

        self.label_title = customtkinter.CTkLabel(self,
                                                  text="Wind-Parameters",
                                                  fg_color="gray28",
                                                  corner_radius=5)
        self.label_title.grid(row=0, column=0)
        self.unity_check_frame = unity_check_frame

        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=1, column=0, sticky="nsew")
        self.tabview.add("Location")
        self.tabview.add("Wind-pressure")

        self.tabview.tab("Location").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Wind-pressure").grid_columnconfigure(0, weight=1)

        self.optionmenu_1b = customtkinter.CTkOptionMenu(self.tabview.tab("Location"),

                                                         values=["Coastal", "Rural", "Urban"])
        self.optionmenu_1a = customtkinter.CTkOptionMenu(self.tabview.tab("Location"),
                                                         dynamic_resizing=False,
                                                         values=["Area I", "Area II", "Area III"],
                                                         command=self.validate_option_combinations)
        self.optionmenu_1a.grid(row=0, column=0, padx=20, pady=(10, 10))
        self.optionmenu_1b = customtkinter.CTkOptionMenu(self.tabview.tab("Location"),

                                                         values=["Coastal", "Rural", "Urban"])
        self.optionmenu_1b.grid(row=1, column=0, padx=20, pady=(10, 10))

        self.label_height_1 = customtkinter.CTkLabel(self.tabview.tab("Location"),
                                                     text=f'Structure height [m]',
                                                     fg_color="gray28",
                                                     corner_radius=5)
        self.label_height_1.grid(row=2, column=0, padx=20, pady=(10, 0))
        list_height_structures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                                  80, 85, 90, 95, 100, 110, 120, 130, 140, 150]
        self.spinbox_1 = StructureHeightSpinBox(self.tabview.tab("Location"),
                                                width=150,
                                                values=list_height_structures)
        self.spinbox_1.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.label_windface = customtkinter.CTkLabel(self.tabview.tab("Location"),
                                                     text=f'Wind-face',
                                                     fg_color="gray28",
                                                     corner_radius=5)
        self.label_windface.grid(row=4, column=0, padx=20, pady=(10, 0))

        self.optionmenu_1c = customtkinter.CTkOptionMenu(self.tabview.tab("Location"),
                                                         dynamic_resizing=False,
                                                         variable=self.wind_face_var,
                                                         values=["A", "B", "C", "D", "E"])
        self.optionmenu_1c.grid(row=5, column=0, padx=20, pady=(0, 10))

        self.label_safety_factor = customtkinter.CTkLabel(self.tabview.tab("Location"),
                                                          text=f'Safety-factor',
                                                          fg_color="gray28",
                                                          corner_radius=5)
        self.label_safety_factor.grid(row=6, column=0, padx=20, pady=(10, 0))

        self.safety_factor_entry = customtkinter.CTkEntry(self.tabview.tab("Location"),
                                                          textvariable=self.safety_factor_var)
        self.safety_factor_entry.configure(validate="key", validatecommand=(
            self.safety_factor_entry.register(self.validate_entry), '%P'))
        self.safety_factor_entry.grid(row=7, column=0, padx=20, pady=(0, 10))

        self.button_1 = customtkinter.CTkButton(self.tabview.tab("Location"),
                                                text="Update values",
                                                command=self.update_values)
        self.button_1.grid(row=8, column=0, padx=20, pady=5)

        self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Wind-pressure"),
                                                  text="Wind-pressure [kN/m^2]",
                                                  fg_color="gray28",
                                                  corner_radius=5)

        self.label_tab_2.grid(row=0, column=0, padx=20, pady=(20, 0))

        self.spinbox_2a = FloatSpinbox(self.tabview.tab("Wind-pressure"),
                                       width=150,
                                       step_size=0.1)
        self.spinbox_2a.grid(row=1, column=0, padx=20, pady=(0, 20))
        self.spinbox_2a.set(2.0)

        self.label_windface = customtkinter.CTkLabel(self.tabview.tab("Wind-pressure"),
                                                     text=f'Wind-zone',
                                                     fg_color="gray28",
                                                     corner_radius=5)
        self.label_windface.grid(row=2, column=0, padx=20, pady=(10, 0))

        self.optionmenu_1c = customtkinter.CTkOptionMenu(self.tabview.tab("Wind-pressure"),
                                                         dynamic_resizing=False,
                                                         variable=self.wind_face_var,
                                                         values=["A", "B", "C", "D", "E"])
        self.optionmenu_1c.set("A")
        self.optionmenu_1c.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.label_safety_factor = customtkinter.CTkLabel(self.tabview.tab("Wind-pressure"),
                                                          text=f'Safety-factor',
                                                          fg_color="gray28",
                                                          corner_radius=5)
        self.label_safety_factor.grid(row=4, column=0, padx=20, pady=(10, 0))

        self.safety_factor_entry = customtkinter.CTkEntry(self.tabview.tab("Wind-pressure"),
                                                          textvariable=self.safety_factor_var)
        self.safety_factor_entry.configure(validate="key", validatecommand=(
            self.safety_factor_entry.register(self.validate_entry), '%P'))
        self.safety_factor_entry.grid(row=5, column=0, padx=20, pady=(0, 10))
        self.button_2 = customtkinter.CTkButton(self.tabview.tab("Wind-pressure"),
                                                text="Update values",
                                                command=self.update_values)
        self.button_2.grid(row=6, column=0, padx=20, pady=5)

    def P_wind(self):
        """
        Calculate the wind pressure based on the selected tab and the data dictionary.
        The result is set in the unity_check_frame.
        """
        pwind = None
        selected_tab = self.tabview.get()
        cscd = 1
        sf = float(self.safety_factor_var.get())
        cf_dict = {'A': 1.2, 'B': 0.8, 'C': 0.5, 'D': 0.8, 'E': 0.7}
        if selected_tab == "Location" and self.tab_data_dict is not None:
            cf = cf_dict[self.tab_data_dict["wind_face"]]
            qp_ze = df_PVP.loc[self.tab_data_dict["structure_height"],
            (self.tab_data_dict["area"], self.tab_data_dict["location"])]
            pwind = sf * cscd * cf * qp_ze / 1000
        elif selected_tab == "Wind-pressure" and self.tab_data_dict is not None:
            cf = cf_dict[self.tab_data_dict["wind_face"]]
            qp_ze = self.tab_data_dict["wind_pressure"]
            pwind = sf * cscd * cf * qp_ze / 1000
        self.unity_check_frame.set_Pwind(pwind)

    def validate_option_combinations(self, _):
        """
        Validate the combinations of options selected in the option menus. This is necessary because Area III does not
        have an option for coastal, therefore the possible options have to be altered.
        This method is called when an option is selected in the first option menu: 'location'.
        """
        if self.optionmenu_1a.get() == "Area III":
            self.optionmenu_1b.set("Rural")
            self.optionmenu_1b.configure(values=["Rural", "Urban"])
        else:
            self.optionmenu_1b.set("Coastal")
            self.optionmenu_1b.configure(values=["Coastal", "Rural", "Urban"])

    def update_values(self):
        """
        Update the data dictionary based on the selected tab and the current values of the widgets.
        """
        selected_tab = self.tabview.get()
        if selected_tab == "Location":
            self.tab_data_dict = {
                "tab": selected_tab,
                "area": self.optionmenu_1a.get(),
                "location": self.optionmenu_1b.get(),
                "structure_height": self.spinbox_1.get(),
                "wind_face": self.wind_face_var.get(),
                "safety_factor": self.safety_factor_var.get()
            }
        elif selected_tab == "Wind-pressure":
            self.tab_data_dict = {
                "tab": selected_tab,
                "wind_pressure": self.spinbox_2a.get(),
                "wind_face": self.wind_face_var.get()
            }
        self.P_wind()

    def validate_entry(self, value):
        """
        Validate the entry value. It must be a numeric value
        :param value: str
                The value to validate
        :return: bool
                    True if the value is valid, False otherwise
        """
        if value == "":
            return True
        elif value.replace('.', '', 1).isdigit():
            return True
        else:
            tk.messagebox.showwarning(
                "Invalid Value", "Please enter a valid numeric value.")
            return False


class UnityCheck(customtkinter.CTkFrame):
    """
    A custom tkinter frame for performing Unity Checks. This class provides
    a GUI for inputting various parameters and performing calculations related
    to Unity Checks.
    """
    def __init__(self, master):
        """
        Initialize the UnityCheck frame.
        This method initializes the UnityCheck frame by setting up the grid layout,
        creating and configuring various widgets (labels, entry fields, etc.) and
        initializing a number of instance variables used for calculations.

        The instance variables include various parameters related to the Unity Check
        calculations (e.g., wind pressure, slider dimensions, coefficients, window values,
        extra parameters, etc.), as well as a callback function and a number of tkinter
        StringVar objects for updating the GUI.

        :param master: The parent widget
        """
        super().__init__(master)

        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.label_title = customtkinter.CTkLabel(self,
                                                  text="Unity-Check",
                                                  fg_color="gray28",
                                                  corner_radius=5)
        self.label_title.grid(row=0, column=0)

        self.Outer_ULS_var = tk.StringVar()
        self.Outer_SLS_var = tk.StringVar()
        self.Inner_ULS_var = tk.StringVar()
        self.Inner_SLS_var = tk.StringVar()

        self.UC_outer = customtkinter.CTkFrame(self)
        self.UC_outer.grid(row=1, column=0, sticky="nsew",
                           padx=(5, 5), pady=(5, 5))
        self.UC_outer.grid_rowconfigure((0, 1, 2), weight=1)
        self.UC_outer.grid_columnconfigure((0, 1), weight=1)
        self.label_outer_pane = customtkinter.CTkLabel(self.UC_outer,
                                                       text="U.C. outer pane",
                                                       fg_color="gray28",
                                                       corner_radius=5)

        self.label_outer_pane.grid(row=0, column=0, columnspan=2)

        self.ULS_outer_label = customtkinter.CTkLabel(
            self.UC_outer, text="ULS:")
        self.ULS_outer_label.grid(
            row=1, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.ULS_outer_entry = customtkinter.CTkEntry(
            self.UC_outer, width=80, textvariable=self.Outer_ULS_var, state='disabled')
        self.ULS_outer_entry.grid(
            row=1, column=1, sticky="ew", padx=(5, 5), pady=(5, 5))

        self.SLS_outer_label = customtkinter.CTkLabel(
            self.UC_outer, text="SLS:")
        self.SLS_outer_label.grid(
            row=2, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.SLS_outer_entry = customtkinter.CTkEntry(
            self.UC_outer, width=80, textvariable=self.Outer_SLS_var, state='disabled')
        self.SLS_outer_entry.grid(
            row=2, column=1, sticky="ew", padx=(5, 5), pady=(5, 5))

        self.UC_inner = customtkinter.CTkFrame(self)
        self.UC_inner.grid(row=3, column=0, sticky="nsew",
                           padx=(5, 5), pady=(5, 5))
        self.UC_inner.grid_rowconfigure((0, 1, 2), weight=1)
        self.UC_inner.grid_columnconfigure((0, 1), weight=1)

        self.label_inner_pane = customtkinter.CTkLabel(self.UC_inner,
                                                       text="U.C. inner pane",
                                                       fg_color="gray28",
                                                       corner_radius=5)

        self.label_inner_pane.grid(row=0, column=0, columnspan=2)

        self.ULS_inner_label = customtkinter.CTkLabel(
            self.UC_inner, text="ULS:")
        self.ULS_inner_label.grid(
            row=1, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.ULS_inner_entry = customtkinter.CTkEntry(
            self.UC_inner, width=80, textvariable=self.Inner_ULS_var, state='disabled')
        self.ULS_inner_entry.grid(
            row=1, column=1, sticky="ew", padx=(5, 5), pady=(5, 5))

        self.SLS_inner_label = customtkinter.CTkLabel(
            self.UC_inner, text="SLS:")
        self.SLS_inner_label.grid(
            row=2, column=0, sticky="w", padx=(5, 5), pady=(5, 5))
        self.SLS_inner_entry = customtkinter.CTkEntry(
            self.UC_inner, width=80, textvariable=self.Inner_SLS_var, state='disabled')
        self.SLS_inner_entry.grid(
            row=2, column=1, sticky="ew", padx=(5, 5), pady=(5, 5))

        self.p_wind = None
        self.sliderHeight = None
        self.sliderWidth = None
        self.alpha = None
        self.beta = None
        self.outer_strength = None
        self.outer_thickness = None
        self.outer_height_pane = None
        self.outer_width_pane = None
        self.inner_strength = None
        self.inner_thickness = None
        self.inner_height_pane = None
        self.inner_width_pane = None
        self.width_gap = None
        self.k_a_var = None
        self.k_e_var = None
        self.k_mod_var = None
        self.k_sp_var = None
        self.gamma_var = None
        self.E = 70000
        self.callback = None

    def register_callback(self, callback):
        """
        Register a callback function to be called when certain events occur
        (e.g. when certain calculations are completed.)

        :param callback:
        """
        self.callback = callback

    def notify(self):
        """
        This method calls the registered callback function, if one has been registered.
        """
        self.callback()

    def set_Pwind(self, value):
        """
        This method is used to set the wind pressure value and then update the values
        used in the calculations.

        :param value: The wind pressure value
        """
        self.p_wind = value
        self.update_values()

    def set_SliderValues(self, height, width):
        """
        Set the slider height and width.
        This method is used to set the slider height and width values

        :param height: The slider height
        :param width: The slider width
        """
        self.sliderHeight = height
        self.sliderWidth = width

    def set_coefficients(self, alpha, beta):
        """
        This method is used to set the alpha and beta coefficients and then update the values
        used for the calculations

        :param alpha: The alpha coefficient
        :param beta: The beta coefficient
        """
        self.alpha = alpha
        self.beta = beta
        self.update_values()

    def set_windows_values(self, value):
        """
        This method is used to set various window values (e.g., outer strength, outer thickness,
        outer height pane, outer width pane, inner strength, inner thickness, inner height pane,
        inner width pane, width gap) based on the provided dictionary. If any of the values cannot
        be converted to a float, an error message is generated.

        :param value: A dictionary containing the window values
        :return:
        """
        conversion_errors = []
        try:
            self.outer_strength = float(value["outer_strength"])
        except (ValueError, TypeError):
            conversion_errors.append("outer_strength")
        try:
            self.outer_thickness = float(value["outer_thickness"])
        except (ValueError, TypeError):
            conversion_errors.append("outer_thickness")
        try:
            self.outer_height_pane = float(value["outer_height_pane"])
        except (ValueError, TypeError):
            conversion_errors.append("outer_height_pane")
        try:
            self.outer_width_pane = float(value["outer_width_pane"])
        except (ValueError, TypeError):
            conversion_errors.append("outer_width_pane")
        try:
            self.inner_strength = float(value["inner_strength"])
        except (ValueError, TypeError):
            conversion_errors.append("inner_strength")
        try:
            self.inner_thickness = float(value["inner_thickness"])
        except (ValueError, TypeError):
            conversion_errors.append("inner_thickness")
        try:
            self.inner_height_pane = float(value["inner_height_pane"])
        except (ValueError, TypeError):
            conversion_errors.append("inner_height_pane")
        try:
            self.inner_width_pane = float(value["inner_width_pane"])
        except (ValueError, TypeError):
            conversion_errors.append("inner_width_pane")
        try:
            self.width_gap = float(value["width_gap"])
        except (ValueError, TypeError):
            conversion_errors.append("width_gap")

        if conversion_errors:
            error_message = "The following entries could not be converted to float or decimal [May be Empty]: {}".format(
                ", ".join(conversion_errors))

    def set_Extra_Params(self, k_a_var, k_e_var, k_mod_var, k_sp_var, gamma_var):
        """
        This method is used to set the extra parameters (k_a_var, k_e_var, k_mod_var, k_sp_var, gamma_var)
        and then update the values used in the calculations. If any of the values cannot be converted to a float,
        an error message is generated.

        :param k_a_var: The coefficient for the surface effect
        :param k_e_var: The coefficient for the edge quality of the glass pane
        :param k_mod_var: The modification coefficient
        :param k_sp_var: The coefficient for the surface structure of the glass pane
        :param gamma_var: The material coefficient of glass
        """
        conversion_errors = []
        try:
            self.k_a_var = float(k_a_var)
        except (ValueError, TypeError):
            conversion_errors.append("k_a_var")
        try:
            self.k_e_var = float(k_e_var)
        except (ValueError, TypeError):
            conversion_errors.append("k_e_var")
        try:
            self.k_mod_var = float(k_mod_var)
        except (ValueError, TypeError):
            conversion_errors.append("k_mod_var")
        try:
            self.k_sp_var = float(k_sp_var)
        except (ValueError, TypeError):
            conversion_errors.append("k_sp_var")
        try:
            self.gamma_var = float(gamma_var)
        except (ValueError, TypeError):
            conversion_errors.append("gamma_var")

        if conversion_errors:
            error_message = "The following parameters could not be converted to float or decimal: {}".format(
                ", ".join(conversion_errors))
        else:
            self.update_values()

    def compute_k_a(self):
        """
        This method computes the k_a value based on the slider width and height. The k_a value is
        used in the calculations of the glass strength. If the product of the slider width and height is zero,
        the k_a value is set to one.

        :return: The computed k_a value
        """
        width = self.sliderWidth
        length = self.sliderHeight
        if width * length != 0:
            self.k_a_var = 1.644 * (width * length) ** (-1 / 25)
        else:
            self.k_a_var = 1
        self.notify()
        return self.k_a_var

    def compute_outer_glass_strength(self):
        """
        This method calculates the strength of the outer glass based on various parameters including
        k_a_var, k_e_var, k_mod_var, k_sp_var, outer_strength and gamma_var.

        :return: The computed strength of the outer glass
        """
        glass_strength = self.k_a_var * self.k_e_var * self.k_mod_var * self.k_sp_var * self.outer_strength / self.gamma_var
        return glass_strength

    def compute_inner_glass_strength(self):
        """
        This method calculates the strength of the inner glass based on various parameters including
        k_a_var, k_e_var, k_mod_var, k_sp_var, inner_strength and gamma_var.

        :return: The computed strength of the inner glass
        """
        glass_strength = self.k_a_var * self.k_e_var * self.k_mod_var * self.k_sp_var * self.inner_strength / self.gamma_var

        return glass_strength

    def compute_z1(self):
        """
        This method calculates the value of z1 based on the slider width and height. The value of z1
        is used in the calculation of the chi value.

        :return: The computed value of z1
        """

        _exp = (-1.123 * ((self.sliderHeight / self.sliderWidth) - 1) ** 1.097)

        if isinstance(_exp, complex):
            return 181.8 * (self.sliderWidth / self.sliderHeight) ** 2 * (0.00406 + 0.00896 * (1 - cmath.exp(_exp)))
        else:
            return 181.8 * (self.sliderWidth / self.sliderHeight) ** 2 * (0.00406 + 0.00896 * (1 - math.exp(_exp)))

    def compute_chi(self):
        """
        This method calculates the value of chi, the shape factor of the characteristic length of the pane,
        based on the slider width, height and the computed value of z1.
        The value of chi is used in the calculation of the a_ value.

        :return: The computed value of chi
        """

        z1 = self.compute_z1()
        _exp = (-6.8 * (self.sliderWidth / self.sliderHeight) ** 1.33)
        # print(f'chi = {(z1 / 16) * (0.4198 + 0.22 * math.exp(_exp)) * (self.sliderHeight / self.sliderWidth) ** 2}')
        return (z1 / 16) * (0.4198 + 0.22 * math.exp(_exp)) * (self.sliderHeight / self.sliderWidth) ** 2

    def compute_a_(self):
        """
        This method calculates the value of a_ the characteristic length for the cavity of the rectangular glass pane,
        based on the thicknesses of the glass, the width of the gap and the computed value of chi. The value of a_ is used in the calculation of the insulation factor.

        :return: The computed value of a_
        """

        chi = self.compute_chi()
        # print(f'a_ = {28.9 * ((self.width_gap * (self.outer_thickness ** 3) * (self.inner_thickness ** 3)) / (((self.outer_thickness ** 3) + (self.inner_thickness ** 3)) * chi)) ** 0.25}')
        return 28.9 * ((self.width_gap * (self.outer_thickness ** 3) * (self.inner_thickness ** 3)) / (((self.outer_thickness ** 3) + (self.inner_thickness ** 3)) * chi)) ** 0.25

    def compute_insulation_factor(self):
        """
        This method calculates the insulation factor based on the slider height and the computed value of a_.
        The insulation factor is used in the calculation of the P_E value.

        :return: The computed insulation factor
        """

        a_ = self.compute_a_()

        return 1 / (1 + (self.sliderHeight / a_) ** 4)

    def compute_p_factor(self):
        """
        This method calculates the value of P_E based on the isolation factor, the thicknesses of the glass,
        and the external load. The P_E value is used in the calculation of the SLS and ULS values and
        divides the external load over the inner and outer pane.

        :return: The computed value of P_E
        """
        insulation_factor = self.compute_insulation_factor()
        # print(f'p_factor = {(1 - insulation_factor) * ((self.inner_thickness ** 3) / ((self.outer_thickness ** 3) + (self.inner_thickness ** 3)))}')
        return (1 - insulation_factor) * ((self.inner_thickness ** 3) / ((self.outer_thickness ** 3) + (self.inner_thickness ** 3)))

    def compute_U_dia(self):
        """
        This method calculates the value of U_dia, the limit of the deflection of the diagonal of the pane,
        based on the slider width and height. The U_dia value is used in the calculation of the SLS values.

        :return: The computed value of U_dia
        """

        return (((self.sliderHeight ** 2) + (self.sliderWidth ** 2)) ** 0.5) / 65

    def compute_SLS_outer(self):
        """
        This method calculates the SLS, Serviceability Limit State, value for the outer glass based on various
        parameters including alpha, p_wind, slider width, E, outer_thickness and the computed value of U_dia.

        :return: The computed SLS value for the outer pane.
        """
        delta_outer = self.alpha * (self.p_wind - self.compute_p_factor() * self.p_wind) * (self.sliderWidth ** 4) / (self.E * (self.outer_thickness ** 3))

        u_dia = self.compute_U_dia()
        self.Outer_SLS_var.set("{:.3f}".format(delta_outer / u_dia))

    def compute_SLS_inner(self):
        """
        This method calculates the SLS, Serviceability Limit State, value for the inner pane based on various
        parameters including alpha, p_wind, slider width, E, outer_thickness and the computed value of U_dia.

        :return: The computed SLS value for the inner pane.
        """
        delta_inner = self.alpha * (self.compute_p_factor() * self.p_wind) * (self.sliderWidth ** 4) / (self.E * (self.inner_thickness ** 3))

        u_dia = self.compute_U_dia()
        self.Inner_SLS_var.set("{:.3f}".format(delta_inner / u_dia))

    def compute_ULS_outer(self):
        """
        This method calculates the ULS , Ulimate Limit State, value for the outer pane based on various parameters including
        beta, p_wind, slider height, outer_thickness and the computed values of P_E and outer glass strength.

        :return: The computed ULS value for the outer pane
        """
        sigma_outer = self.beta * (self.p_wind - self.compute_p_factor() * self.p_wind) * \
                      (self.sliderHeight ** 2) / (self.outer_thickness ** 2)

        glass_strength = self.compute_outer_glass_strength()

        self.Outer_ULS_var.set("{:.2f}".format(sigma_outer / glass_strength))
        self.ULS_outer_indicator.text = self.Outer_ULS_var.get()
        self.ULS_outer_indicator.update_background()

    def compute_ULS_inner(self):
        """
        This method calculates the ULS , Ulimate Limit State, value for the inner pane based on various parameters including
        beta, p_wind, slider height, outer_thickness and the computed values of P_E and outer glass strength.

        :return: The computed ULS value for the inner pane
        """
        sigma_inner = self.beta * (self.compute_p_factor() * self.p_wind) * \
                      (self.sliderHeight ** 2) / (self.outer_thickness ** 2)
        # print(f'sigma inner = {sigma_inner} N/mm^2')
        glass_strength = self.compute_inner_glass_strength()
        # print(f'inner glass strength = {glass_strength} N/mm^2')
        self.Inner_ULS_var.set("{:.2f}".format(sigma_inner / glass_strength))

    def update_values(self):
        """
        This method updates the SLS and ULS values by calling the respective compute methods. If any exception
        occurs during the computation, it calls the none_check method to identify the missing attributes.
        """
        try:
            self.compute_SLS_inner()
            self.compute_SLS_outer()
            self.compute_ULS_inner()
            self.compute_ULS_outer()
        except Exception as e:
            flag = self.none_check()
    def none_check(self):
        """
        This method checks if any of the attributes used in the calculations are missing (None or empty).
        If any missing attributes are found, it sets the ULS and SLS values to an empty string.

        :return: A string containing the names of the missing attriutes, or True if no missing attributes are found.
        """
        attributes = [
            'p_wind', 'sliderHeight', 'sliderWidth', 'alpha', 'beta', 'outer_strength',
            'outer_thickness', 'outer_height_pane', 'outer_width_pane', 'inner_strength',
            'inner_thickness', 'inner_height_pane', 'inner_width_pane', 'width_gap',
            'k_a_var', 'k_e_var', 'k_mod_var', 'k_sp_var', 'gamma_var', 'E'
        ]

        missing_attributes = []

        for attr in attributes:
            if getattr(self, attr) is None or len(str(getattr(self, attr)).strip()) == 0 or float(
                    str(getattr(self, attr))) == 0:
                missing_attributes.append(attr)

        if missing_attributes:
            self.Outer_ULS_var.set('')
            self.Outer_SLS_var.set('')
            self.Inner_ULS_var.set('')
            self.Inner_SLS_var.set('')

        return ', '.join(missing_attributes) if missing_attributes else True



class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x300")

        self.label = customtkinter.CTkLabel(self, text="ToplevelWindow")
        self.label.pack(padx=20, pady=20)


class App(customtkinter.CTk):
    """
     Main application class for the Parametric tool.

    This class inherits from the customtkinter.CTk class and represents the main application window.
    It contains several frames for different functionalities including UnityCheck, DimensionWindow,
    WindForce and WindowParameters. It also includes a help button that opens a messagebox with help text.
    """
    def __init__(self):
        """
        Initialize the App instance.

        This method initializes the App instance and sets up the grid configuration. It creates instances
        of UnityCheck, DimensionWindow, WindForce and WindowParameters frames and places them on the grid.
        It also creates a help button with an associated command to open a messagebox with help text.
        """
        super().__init__()

        self.title("ReGlass")
        # self.iconbitmap('icon.ico')
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=1)

        self.unity_check_frame = UnityCheck(self)
        self.unity_check_frame.grid(row=1, column=4, padx=(
            10, 0), pady=(10, 0), ipadx=90, sticky='s')

        self.dimensions_window_frame = DimensionWindow(
            self, df_RP, self.unity_check_frame)
        self.dimensions_window_frame.grid(
            row=0, rowspan=2, column=0, columnspan=2, padx=(10, 0), pady=(20, 0),
            sticky="n")

        self.windforce_frame = WindForce(self, self.unity_check_frame)
        self.windforce_frame.grid(
            row=0, column=4, padx=(10, 0), pady=(0, 0), sticky="n")

        self.window_parameters_frame = WindowParameters(
            self, self.dimensions_window_frame, self.unity_check_frame)
        self.window_parameters_frame.grid(
            row=0, rowspan=2, column=3, padx=10, pady=20, sticky="nsew")


if __name__ == "__main__":
    app = App()
    app.state('zoomed')
    app.mainloop()
