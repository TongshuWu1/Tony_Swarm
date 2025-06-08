import tkinter as tk
from tkinter import messagebox, filedialog
import os
import region_detection as Agent_reduction
import colorsys
import adaptive_layout


class Knot_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Reduction - Path Optimization")
        self.root.geometry("1000x700")
        self.canvas_path_items = []
        self.animation_running = False
        self.animation_job = None

        self.last_scale = 1.0
        self.base_cell_size = 40
        self.zoom_scale = 1.0
        self.base_font_size = 10

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.my_canvas = tk.Canvas(main_frame)
        self.my_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        self.my_scrollbar_y = tk.Scrollbar(
            main_frame, orient=tk.VERTICAL, command=self.my_canvas.yview
        )
        self.my_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        self.my_scrollbar_x = tk.Scrollbar(
            self.root, orient=tk.HORIZONTAL, command=self.my_canvas.xview
        )
        self.my_scrollbar_x.pack(fill=tk.X, side=tk.BOTTOM)

        # Attach scrollbars to canvas
        self.my_canvas.configure(
            yscrollcommand=self.my_scrollbar_y.set,
            xscrollcommand=self.my_scrollbar_x.set,
        )

        self.my_canvas.bind(
            "<Configure>",
            lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all")),
        )
        self.my_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Control-MouseWheel>", self._on_ctrl_mousewheel)

        self.second_frame = tk.Frame(self.my_canvas)
        self.my_canvas.create_window((0, 0), window=self.second_frame, anchor="nw")

        self.label_title = tk.Label(
            self.second_frame,
            text="Agent Reduction",
            font=("Arial", 18),
            fg="black",
            bg="lightgray",
        )
        self.label_title.pack(pady=10, fill=tk.X)

        zoom_button_frame = tk.Frame(self.second_frame)
        zoom_button_frame.pack(pady=5, anchor="w")

        self.zoom_in_btn = tk.Button(
            zoom_button_frame, text="+", width=3, command=self.zoom_in
        )
        self.zoom_in_btn.pack(side=tk.LEFT, padx=(10, 2))

        self.zoom_out_btn = tk.Button(
            zoom_button_frame, text="−", width=3, command=self.zoom_out
        )
        self.zoom_out_btn.pack(side=tk.LEFT)

        self.matrix_name_label = tk.Label(self.second_frame, text="Matrix Name:")
        self.matrix_name_label.pack()
        self.matrix_name_entry = tk.Entry(self.second_frame, width=50)
        self.matrix_name_entry.pack()

        self.matrix_label = tk.Label(
            self.second_frame, text="Enter Matrix (comma separated):"
        )
        self.matrix_label.pack()

        self.matrix_frame = tk.Frame(self.second_frame)
        self.matrix_frame.pack()

        self.matrix_text = tk.Text(self.matrix_frame, height=8, width=50)
        self.matrix_text.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(self.matrix_frame)
        self.button_frame.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(
            self.button_frame, text="Save Matrix", command=self.save_matrix
        )
        self.save_button.pack(pady=5)

        self.load_button = tk.Button(
            self.button_frame, text="Load Matrix", command=self.load_matrix
        )
        self.load_button.pack(pady=5)

        self.clear_button = tk.Button(
            self.button_frame, text="Clear", command=self.clear_all
        )
        self.clear_button.pack(pady=5)

        self.notes_label = tk.Label(self.second_frame, text="Notes:")
        self.notes_label.pack()
        self.notes_text = tk.Text(self.second_frame, height=4, width=50)
        self.notes_text.pack()

        self.entry_exit_frame = tk.Frame(self.second_frame)
        self.entry_exit_frame.pack(pady=5)

        tk.Label(self.entry_exit_frame, text="Entry (row, col):").pack(side=tk.LEFT)
        self.entry_point = tk.Entry(self.entry_exit_frame, width=10)
        self.entry_point.pack(side=tk.LEFT, padx=5)

        tk.Label(self.entry_exit_frame, text="Exit (row, col):").pack(side=tk.LEFT)
        self.exit_point = tk.Entry(self.entry_exit_frame, width=10)
        self.exit_point.pack(side=tk.LEFT, padx=5)

        self.run_button = tk.Button(
            self.second_frame, text="Run Agent Reduction", command=self.run_algorithm
        )
        self.run_button.pack(pady=10)

        self.show_matrix_overlay = tk.BooleanVar()
        self.matrix_overlay_checkbox = tk.Checkbutton(
            self.second_frame,
            text="Show Matrix Overlay",
            variable=self.show_matrix_overlay,
            command=self.run_algorithm,
        )
        self.matrix_overlay_checkbox.pack(pady=5)

        self.use_cartesian_layout = tk.BooleanVar()
        self.cartesian_toggle_checkbox = tk.Checkbutton(
            self.second_frame,
            text="Show Cartesian Layout",
            variable=self.use_cartesian_layout,
            command=self.run_algorithm,
        )
        self.cartesian_toggle_checkbox.pack(pady=5)

        self.use_adaptive_layout = tk.BooleanVar()
        self.adaptive_toggle_checkbox = tk.Checkbutton(
            self.second_frame,
            text="Show Adaptive Physical Layout",
            variable=self.use_adaptive_layout,
            command=self.run_algorithm,
        )
        self.adaptive_toggle_checkbox.pack(pady=5)

        self.result_frame = tk.Frame(self.second_frame)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.result_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.output_frame = tk.Frame(self.result_frame)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.path_title_label = tk.Label(
            self.output_frame, text="Path from Entry to Exit:"
        )
        self.path_title_label.pack()

        self.path_text_scroll = tk.Scrollbar(self.output_frame)
        self.path_text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.path_text = tk.Text(
            self.output_frame,
            height=20,
            width=40,
            yscrollcommand=self.path_text_scroll.set,
        )
        self.path_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.path_text_scroll.config(command=self.path_text.yview)

        self.original_points_label = tk.Label(
            self.output_frame, text="Total number of original points: "
        )
        self.original_points_label.pack()

        self.agents_needed_label = tk.Label(
            self.output_frame, text="Total number of agents needed after reduction: "
        )
        self.agents_needed_label.pack()

        self.crossing_number_label = tk.Label(
            self.output_frame, text="Total number of crossings: "
        )
        self.crossing_number_label.pack()

    def zoom_in(self):
        self.zoom_scale = min(3.0, self.zoom_scale + 0.1)
        self.apply_full_zoom()

    def zoom_out(self):
        self.zoom_scale = max(0.5, self.zoom_scale - 0.1)
        self.apply_full_zoom()

    def _on_ctrl_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def set_zoom(self, scale):
        scale = max(0.5, min(scale, 3.0))
        if abs(scale - self.last_scale) < 0.01:
            return

        # Calculate relative scale factor
        factor = scale / self.last_scale
        self.last_scale = scale

        # Scale all canvas elements around origin
        self.canvas.scale("all", 0, 0, factor, factor)
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.my_canvas.update_idletasks()
        self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all"))

    def scale_widgets(self, scale):
        font_base = int(10 * scale)
        big_font = ("Arial", max(12, int(14 * scale)))
        small_font = ("Arial", max(8, font_base - 2))

        self.label_title.config(font=big_font)
        self.matrix_label.config(font=font_base)
        self.matrix_name_label.config(font=font_base)
        self.notes_label.config(font=font_base)
        self.path_title_label.config(font=font_base)
        self.original_points_label.config(font=font_base)
        self.agents_needed_label.config(font=font_base)
        self.crossing_number_label.config(font=font_base)

        self.matrix_text.config(font=font_base)
        self.notes_text.config(font=font_base)
        self.path_text.config(font=font_base)
        self.entry_point.config(font=font_base)
        self.exit_point.config(font=font_base)
        self.save_button.config(font=font_base)
        self.load_button.config(font=font_base)
        self.clear_button.config(font=font_base)
        self.run_button.config(font=font_base)
        self.matrix_overlay_checkbox.config(font=font_base)

        self.cell_size = int(self.base_cell_size * scale)
        self.run_algorithm()

    def apply_full_zoom(self):
        scale = self.zoom_scale
        font_size = int(self.base_font_size * scale)

        font_normal = ("Arial", font_size)
        font_title = ("Arial", int(font_size * 1.8))

        widgets = [
            self.label_title,
            self.matrix_label,
            self.matrix_name_label,
            self.notes_label,
            self.path_title_label,
            self.original_points_label,
            self.agents_needed_label,
            self.crossing_number_label,
            self.save_button,
            self.load_button,
            self.clear_button,
            self.run_button,
            self.matrix_overlay_checkbox,
        ]

        for widget in widgets:
            widget.config(font=font_normal)

        self.label_title.config(font=font_title)

        self.matrix_text.config(
            font=font_normal, height=int(8 * scale), width=int(50 * scale)
        )
        self.notes_text.config(
            font=font_normal, height=int(4 * scale), width=int(50 * scale)
        )
        self.path_text.config(
            font=font_normal, height=int(20 * scale), width=int(40 * scale)
        )
        self.entry_point.config(font=font_normal, width=int(10 * scale))
        self.exit_point.config(font=font_normal, width=int(10 * scale))

        self.cell_size = int(40 * scale)

        self.run_algorithm()  # redraw canvas/grid with new size
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.my_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

    def clear_all(self):
        self.matrix_name_entry.delete(0, tk.END)
        self.matrix_text.delete("1.0", tk.END)
        self.notes_text.delete("1.0", tk.END)
        self.entry_point.delete(0, tk.END)
        self.exit_point.delete(0, tk.END)
        self.path_text.delete("1.0", tk.END)
        self.original_points_label.config(text="Total number of original points: ")
        self.agents_needed_label.config(
            text="Total number of agents needed after reduction: "
        )
        self.crossing_number_label.config(text="Total number of crossings: ")
        self.canvas.delete("all")

        # 🛑 Stop animation
        self.animation_running = False
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None

    def save_matrix(self):
        try:
            matrix_name = self.matrix_name_entry.get().strip()
            if not matrix_name:
                messagebox.showerror("Error", "Matrix name is empty")
                return

            matrix_str = self.matrix_text.get("1.0", tk.END).strip()
            if not matrix_str:
                messagebox.showerror("Error", "Matrix is empty")
                return

            notes_str = self.notes_text.get("1.0", tk.END).strip()

            entry = self.entry_point.get().strip() or "None"
            exit_ = self.exit_point.get().strip() or "None"

            file_name = filedialog.asksaveasfilename(
                initialdir="matrixes",
                initialfile=matrix_name,
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
            )
            if file_name:
                if os.path.exists(file_name):
                    replace = messagebox.askyesno(
                        "Replace File",
                        "File already exists. Do you want to replace it?",
                    )
                    if not replace:
                        return

                with open(file_name, "w") as file:
                    file.write(f"Entry: {entry}\n")
                    file.write(f"Exit: {exit_}\n")
                    file.write(f"Notes: {notes_str}\n")
                    file.write(matrix_str)
                self.matrix_name_entry.delete(0, tk.END)
                self.matrix_name_entry.insert(
                    0, os.path.basename(file_name).split(".")[0]
                )
                messagebox.showinfo("Success", "Matrix saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save matrix: {e}")

    def load_matrix(self):
        try:
            file_path = filedialog.askopenfilename(
                initialdir="matrixes", filetypes=[("Text files", "*.txt")]
            )
            if file_path:
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    entry = lines[0].strip().split(": ")[1]
                    exit_ = lines[1].strip().split(": ")[1]

                    # Handle optional or empty Notes line
                    notes = ""
                    matrix_start_line = 2
                    if len(lines) > 2 and lines[2].startswith("Notes:"):
                        parts = lines[2].strip().split(":", 1)
                        notes = parts[1].strip() if len(parts) > 1 else ""
                        matrix_start_line = 3

                    matrix_str = "".join(lines[matrix_start_line:])

                self.matrix_text.delete("1.0", tk.END)
                self.matrix_text.insert(tk.END, matrix_str)
                self.notes_text.delete("1.0", tk.END)
                self.notes_text.insert(tk.END, notes)
                self.entry_point.delete(0, tk.END)
                if entry != "None":
                    self.entry_point.insert(0, entry)
                self.exit_point.delete(0, tk.END)
                if exit_ != "None":
                    self.exit_point.insert(0, exit_)
                self.matrix_name_entry.delete(0, tk.END)
                self.matrix_name_entry.insert(
                    0, os.path.basename(file_path).split(".")[0]
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load matrix: {e}")

    def run_algorithm(self):
        try:
            matrix_str = self.matrix_text.get("1.0", tk.END).strip()
            if not matrix_str:
                return  # no matrix to process

            matrix_lines = matrix_str.split("\n")
            matrix = [
                list(map(int, line.split(","))) for line in matrix_lines if line.strip()
            ]

            entry_text = self.entry_point.get().strip()
            exit_text = self.exit_point.get().strip()
            if (
                not entry_text
                or not exit_text
                or "," not in entry_text
                or "," not in exit_text
            ):
                return  # missing or invalid entry/exit input

            entry = tuple(map(int, entry_text.split(",")))
            exit_ = tuple(map(int, exit_text.split(",")))

            path, head, crossNumber, loop_map, agent_registry = (
                Agent_reduction.compute_agent_reduction(matrix, entry, exit_)
            )
            self.loop_map = loop_map

            path_list = []
            current_node = head
            while current_node:
                r, c = current_node.data
                pt_type = current_node.point_identifier
                path_list.append((r, c, pt_type))
                current_node = current_node.next

            self.crossing_points = set(
                (r, c)
                for r in range(len(matrix))
                for c in range(len(matrix[0]))
                if matrix[r][c] == 3
            )

            self.loop_colors = {}
            num_loops = len(loop_map)
            for i, loop_id in enumerate(sorted(loop_map.keys())):
                hue = (i + 1) / (num_loops + 1)
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color = "#%02x%02x%02x" % tuple(int(x * 255) for x in rgb)
                self.loop_colors[loop_id] = color

            sorted_agents = sorted(agent_registry.items(), key=lambda x: x[0])
            agent_points = [coord for _, coord in sorted_agents]

            # 🟢 Choose the layout mode
            if self.use_adaptive_layout.get():
                layout_segments = adaptive_layout.adaptive_physical_layout(
                    path_list, matrix, agent_points
                )
                adaptive_layout.draw_adaptive_layout(self.canvas, layout_segments)
            elif self.use_cartesian_layout.get():
                self.draw_cartesian_path(path_list, loop_map)
            else:
                self.draw_grid(matrix, path_list, loop_map)

            # 🟡 Update path display
            self.path_text.delete("1.0", tk.END)
            formatted_path = "\n".join([f"{point}" for point in path_list])
            self.path_text.insert(tk.END, formatted_path + "\n")

            turning_points = sum(
                1
                for i in range(1, len(path_list) - 1)
                if (
                    path_list[i - 1][0] - path_list[i][0],
                    path_list[i - 1][1] - path_list[i][1],
                )
                != (
                    path_list[i][0] - path_list[i + 1][0],
                    path_list[i][1] - path_list[i + 1][1],
                )
            )

            agents_needed = sum(1 for point in path_list if point[2] == "agent")

            self.original_points_label.config(
                text=f"Total number of turning points: {turning_points}"
            )
            self.agents_needed_label.config(
                text=f"Total number of agents needed after reduction: {agents_needed}"
            )
            self.crossing_number_label.config(
                text=f"Total number of crossings: {crossNumber}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def draw_grid(self, matrix, path, loop_map):
        self.canvas.delete("all")
        self.canvas_path_items = []

        rows, cols = len(matrix), len(matrix[0])
        self.cell_size = max(30, min(50, 800 // max(rows, cols)))
        canvas_width = cols * self.cell_size
        canvas_height = rows * self.cell_size
        self.canvas.config(width=canvas_width, height=canvas_height)

        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = (c + 1) * self.cell_size, (r + 1) * self.cell_size
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="gray", fill="white"
                )
                self.canvas.create_text(
                    x2 - 3,
                    y2 - 3,
                    anchor="se",
                    text=f"({r},{c})",
                    font=("Arial", 7),
                    fill="gray",
                )
                if self.show_matrix_overlay.get():
                    self.canvas.create_text(
                        x1 + self.cell_size // 2,
                        y1 + self.cell_size // 2,
                        text=str(matrix[r][c]),
                        font=("Arial", 10),
                        tags="overlay",
                    )

        for r, c, pt_type in path:
            if pt_type == "agent":
                x = c * self.cell_size + self.cell_size // 2
                y = r * self.cell_size + self.cell_size // 2
                radius = 6
                self.canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius, fill="gold"
                )

        if path:
            entry_r, entry_c = path[0][:2]
            exit_r, exit_c = path[-1][:2]
            self._draw_marker(entry_r, entry_c, "green")
            self._draw_marker(exit_r, exit_c, "red")

        for i in range(len(path) - 1):
            r1, c1, _ = path[i]
            r2, c2, _ = path[i + 1]
            x1 = c1 * self.cell_size + self.cell_size // 2
            y1 = r1 * self.cell_size + self.cell_size // 2
            x2 = c2 * self.cell_size + self.cell_size // 2
            y2 = r2 * self.cell_size + self.cell_size // 2

            color = "red"
            edge = tuple(sorted([(r1, c1), (r2, c2)]))
            for loop_id, loop_info in loop_map.items():
                loop_path = loop_info["path"]
                loop_edges = set(
                    tuple(sorted([loop_path[j], loop_path[(j + 1) % len(loop_path)]]))
                    for j in range(len(loop_path))
                )
                if edge in loop_edges:
                    color = self.loop_colors.get(loop_id, "red")
                    break

            if r1 == r2:
                row = r1
                col_start = min(c1, c2)
                col_end = max(c1, c2)
                cx_last = 0
                for col in range(col_start, col_end):
                    y = row * self.cell_size + self.cell_size // 2
                    cx = col * self.cell_size + self.cell_size // 2
                    cx_next = (col + 1) * self.cell_size + self.cell_size // 2
                    if cx_last == 0:
                        cx_last = cx

                    if matrix[row][col] == 3:
                        center_x = col * self.cell_size + self.cell_size // 2
                        y = row * self.cell_size + self.cell_size // 2
                        gap = 30
                        gap_start = center_x - gap / 2
                        gap_end = center_x + gap / 2

                        print(
                            "Drawing horizontal line with gap at row:",
                            row,
                            "col:",
                            col,
                            "gap_start:",
                            gap_start,
                            "gap_end:",
                            gap_end,
                        )

                        # Draw left half line only up to the gap_start
                        self.canvas.create_line(
                            cx_last, y, gap_start, y, fill="black", width=4
                        )
                        self.canvas.create_line(
                            cx_last, y, gap_start, y, fill=color, width=2
                        )

                        # Draw right half line starting after the gap_end
                        self.canvas.create_line(
                            gap_end, y, cx_next, y, fill="black", width=4
                        )
                        self.canvas.create_line(
                            gap_end, y, cx_next, y, fill=color, width=2
                        )

                    else:
                        self.canvas.create_line(
                            cx_last, y, cx_next, y, fill="black", width=4
                        )
                        self.canvas.create_line(
                            cx_last, y, cx_next, y, fill=color, width=2
                        )

                    cx_last = cx_next

            elif c1 == c2:
                self.canvas.create_line(
                    x1, y1, x2, y2, fill="black", width=4, arrow=tk.LAST
                )
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all"))

    def _draw_marker(self, r, c, color):
        x = c * self.cell_size + self.cell_size // 2
        y = r * self.cell_size + self.cell_size // 2
        radius = 5
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius, fill=color
        )

    def draw_cartesian_path(self, path, loop_map):
        self.canvas.delete("all")
        self.canvas_path_items = []

        if not path:
            return

        scale = 40  # pixels per unit
        margin = 50
        coords = [(c * scale + margin, r * scale + margin) for r, c, _ in path]

        for (x, y), (_, _, pt_type) in zip(coords, path):
            if pt_type == "agent":
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="gold")

        # Draw start/end markers
        self._draw_cartesian_marker(*coords[0], "green")
        self._draw_cartesian_marker(*coords[-1], "red")

        # Draw lines between points
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            color = "red"
            edge = tuple(
                sorted([(path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1])])
            )
            for loop_id, loop_info in loop_map.items():
                loop_path = loop_info["path"]
                loop_edges = set(
                    tuple(sorted([loop_path[j], loop_path[(j + 1) % len(loop_path)]]))
                    for j in range(len(loop_path))
                )
                if edge in loop_edges:
                    color = self.loop_colors.get(loop_id, "red")
                    break
            self.canvas.create_line(x1, y1, x2, y2, fill="black", width=4)
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all"))

    def _draw_cartesian_marker(self, x, y, color):
        radius = 5
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius, fill=color
        )

    def split_into_sections_by_agents(self, path_list, agent_points):
        sections = []
        current_section = []

        for point in path_list:
            current_section.append(point)
            if point[:2] in agent_points and len(current_section) > 1:
                sections.append(current_section)
                current_section = [point]  # Start new section with this agent

        if len(current_section) > 1:
            sections.append(current_section)

        return sections

    def check_and_log_crossings_by_section(self, sections):
        def segments_cross(p1, p2, q1, q2):
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

            return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (
                ccw(p1, p2, q1) != ccw(p1, p2, q2)
            )

        log_lines = ["\n\U0001F50D Section-wise Crossing Check:"]
        for idx, section in enumerate(sections):
            for i in range(len(section) - 1):
                a1 = section[i][:2]
                a2 = section[i + 1][:2]
                for sidx in range(idx):
                    prev_section = sections[sidx]
                    for j in range(len(prev_section) - 1):
                        b1 = prev_section[j][:2]
                        b2 = prev_section[j + 1][:2]
                        if segments_cross(a1, a2, b1, b2):
                            log_lines.append(
                                f"  ⚠️ Section {idx + 1} [{a1} → {a2}] crosses section {sidx + 1} [{b1} → {b2}]"
                            )

        if len(log_lines) == 1:
            log_lines.append("  ✅ No section-wise crossings detected.")

        self.path_text.insert(tk.END, "\n".join(log_lines) + "\n")
        self.path_text.see(tk.END)
        self.path_text.update_idletasks()

    def segments_cross(self, p1, p2, q1, q2, agent_points):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        shared = set((p1, p2)) & set((q1, q2))
        if shared:
            # If they connect at a shared endpoint and that point is an agent, don't consider it a cross
            if any(pt in agent_points for pt in shared):
                return False

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (
            ccw(p1, p2, q1) != ccw(p1, p2, q2)
        )

    def get_segment_intersection(self, p1, p2, q1, q2):
        # Solve line-line intersection algebraically
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        # Clamp to grid points for simplicity
        return (round(px), round(py))


if __name__ == "__main__":

    root = tk.Tk()
    app = Knot_GUI(root)
    root.mainloop()
