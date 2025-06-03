import tkinter as tk
from tkinter import messagebox, filedialog
import os
import cartesian_reduction as Agent_reduction
import colorsys
import math

class cartesian_GUI:
    def __init__(self, root):


        self.root = root
        self.root.title("Agent Reduction - Path Optimization")
        self.root.geometry("1000x700")
        self.canvas_path_items = []
        self.animation_running = False
        self.animation_job = None


        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.animation_running = False
        self.animation_job = None

        self.my_canvas = tk.Canvas(main_frame)
        self.my_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.my_scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.my_canvas.yview)
        self.my_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.my_canvas.configure(yscrollcommand=self.my_scrollbar.set)
        self.my_canvas.bind("<Configure>", lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all")))
        self.my_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.second_frame = tk.Frame(self.my_canvas)
        self.my_canvas.create_window((0, 0), window=self.second_frame, anchor="nw")

        self.label_title = tk.Label(self.second_frame, text="Agent Reduction", font=("Arial", 18), fg="black", bg="lightgray")
        self.label_title.pack(pady=10, fill=tk.X)

        self.matrix_name_label = tk.Label(self.second_frame, text="Matrix Name:")
        self.matrix_name_label.pack()
        self.matrix_name_entry = tk.Entry(self.second_frame, width=50)
        self.matrix_name_entry.pack()

        self.matrix_label = tk.Label(self.second_frame, text="Enter Matrix (comma separated):")
        self.matrix_label.pack()

        self.matrix_frame = tk.Frame(self.second_frame)
        self.matrix_frame.pack()

        self.matrix_text = tk.Text(self.matrix_frame, height=8, width=50)
        self.matrix_text.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(self.matrix_frame)
        self.button_frame.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(self.button_frame, text="Save Matrix", command=self.save_matrix)
        self.save_button.pack(pady=5)

        self.load_button = tk.Button(self.button_frame, text="Load Matrix", command=self.load_matrix)
        self.load_button.pack(pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_all)
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

        self.run_button = tk.Button(self.second_frame, text="Run Agent Reduction", command=self.run_algorithm)
        self.run_button.pack(pady=10)

        self.result_frame = tk.Frame(self.second_frame)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.result_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.output_frame = tk.Frame(self.result_frame)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.path_title_label = tk.Label(self.output_frame, text="Path from Entry to Exit:")
        self.path_title_label.pack()

        self.show_matrix_overlay = tk.BooleanVar()
        self.matrix_overlay_checkbox = tk.Checkbutton(self.second_frame, text="Show Matrix Overlay",
                                                      variable=self.show_matrix_overlay, command=self.run_algorithm)
        self.matrix_overlay_checkbox.pack(pady=5)

        self.path_text = tk.Text(self.output_frame, height=20, width=40)
        self.path_text.pack(fill=tk.BOTH, expand=True)

        self.original_points_label = tk.Label(self.output_frame, text="Total number of original points: ")
        self.original_points_label.pack()

        self.agents_needed_label = tk.Label(self.output_frame, text="Total number of agents needed after reduction: ")
        self.agents_needed_label.pack()

        self.crossing_number_label = tk.Label(self.output_frame, text="Total number of crossings: ")
        self.crossing_number_label.pack()


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
        self.agents_needed_label.config(text="Total number of agents needed after reduction: ")
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

            file_name = filedialog.asksaveasfilename(initialdir='matrixes', initialfile=matrix_name, defaultextension=".txt",
                                                     filetypes=[("Text files", "*.txt")])
            if file_name:
                if os.path.exists(file_name):
                    replace = messagebox.askyesno("Replace File", "File already exists. Do you want to replace it?")
                    if not replace:
                        return

                with open(file_name, 'w') as file:
                    file.write(f"Entry: {entry}\n")
                    file.write(f"Exit: {exit_}\n")
                    file.write(f"Notes: {notes_str}\n")
                    file.write(matrix_str)
                self.matrix_name_entry.delete(0, tk.END)
                self.matrix_name_entry.insert(0, os.path.basename(file_name).split('.')[0])
                messagebox.showinfo("Success", "Matrix saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save matrix: {e}")


    def load_matrix(self):
        try:
            file_path = filedialog.askopenfilename(initialdir='matrixes', filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
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
                self.matrix_name_entry.insert(0, os.path.basename(file_path).split('.')[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load matrix: {e}")

    import math

    def run_algorithm(self):
        try:
            matrix_str = self.matrix_text.get("1.0", tk.END).strip()
            matrix_lines = matrix_str.split("\n")
            matrix = [list(map(int, line.split(','))) for line in matrix_lines if line.strip()]

            entry = tuple(map(int, self.entry_point.get().strip().split(',')))
            exit_ = tuple(map(int, self.exit_point.get().strip().split(',')))

            path, head, crossNumber, loop_map = Agent_reduction.compute_agent_reduction(matrix, entry, exit_)
            self.loop_map = loop_map

            path_list = []
            current_node = head
            while current_node:
                r, c = current_node.data
                pt_type = current_node.point_identifier
                path_list.append((r, c, pt_type))
                current_node = current_node.next

            self.crossing_points = set(
                (r, c) for r in range(len(matrix)) for c in range(len(matrix[0])) if matrix[r][c] == 3
            )

            self.loop_colors = {}
            num_loops = len(loop_map)
            for i, loop_id in enumerate(sorted(loop_map.keys())):
                hue = (i + 1) / (num_loops + 1)
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color = '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
                self.loop_colors[loop_id] = color

            self.draw_grid(matrix, path_list, loop_map)

            self.path_text.delete("1.0", tk.END)
            formatted_path = "\n".join([f"{point}" for point in path_list])
            self.path_text.insert(tk.END, formatted_path)

            turning_points = sum(
                1 for i in range(1, len(path_list) - 1)
                if (path_list[i - 1][0] - path_list[i][0], path_list[i - 1][1] - path_list[i][1]) !=
                (path_list[i][0] - path_list[i + 1][0], path_list[i][1] - path_list[i + 1][1])
            )

            agents_needed = sum(1 for point in path_list if point[2] == "agent")

            self.original_points_label.config(text=f"Total number of turning points: {turning_points}")
            self.agents_needed_label.config(text=f"Total number of agents needed after reduction: {agents_needed}")
            self.crossing_number_label.config(text=f"Total number of crossings: {crossNumber}")

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
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", fill="white")
                self.canvas.create_text(x2 - 3, y2 - 3, anchor="se", text=f"({r},{c})", font=("Arial", 7), fill="gray")
                if self.show_matrix_overlay.get():
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2,
                                            text=str(matrix[r][c]), font=("Arial", 10), tags="overlay")

        for r, c, pt_type in path:
            if pt_type == "agent":
                x = c * self.cell_size + self.cell_size // 2
                y = r * self.cell_size + self.cell_size // 2
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="blue")

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
                    tuple(sorted([loop_path[j], loop_path[(j + 1) % len(loop_path)]])) for j in range(len(loop_path))
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
                        gap_start = center_x - gap/2
                        gap_end = center_x + gap/2

                        print('Drawing horizontal line with gap at row:', row, 'col:', col, 'gap_start:', gap_start, 'gap_end:', gap_end)

                        # Draw left half line only up to the gap_start
                        self.canvas.create_line(cx_last, y, gap_start, y, fill="black", width=4 )
                        self.canvas.create_line(cx_last, y, gap_start, y, fill=color, width=2)

                        # Draw right half line starting after the gap_end
                        self.canvas.create_line(gap_end, y, cx_next, y, fill="black", width=4)
                        self.canvas.create_line(gap_end, y, cx_next, y, fill=color, width=2)


                    else:
                        self.canvas.create_line(cx_last, y, cx_next, y, fill="black", width=4)
                        self.canvas.create_line(cx_last, y, cx_next, y, fill=color, width=2)

                    cx_last = cx_next

            elif c1 == c2:
                self.canvas.create_line(x1, y1, x2, y2, fill="black", width=4,arrow=tk.LAST)
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

    def _draw_marker(self, r, c, color):
        x = c * self.cell_size + self.cell_size // 2
        y = r * self.cell_size + self.cell_size // 2
        radius = 5
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)


if __name__ == "__main__":


    root = tk.Tk()
    app = cartesian_GUI(root)
    root.mainloop()