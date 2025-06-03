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
        self.root.state("zoomed")
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

    def offset_point(x1, y1, x2, y2, ratio=0.8):
        """
        Returns a point offset along the segment from (x1, y1) to (x2, y2).
        `ratio` controls how far along the line the offset is.
        """
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return x1, y1
        offset_x = x1 + dx * ratio
        offset_y = y1 + dy * ratio
        return offset_x, offset_y

    offset_point(0, 0, 10, 0)  # Example test: offset along a horizontal line

    def run_algorithm(self):
        try:
            matrix_str = self.matrix_text.get("1.0", tk.END).strip()
            matrix_lines = matrix_str.split("\n")
            matrix = [list(map(int, line.split(','))) for line in matrix_lines if line.strip()]

            entry = tuple(map(int, self.entry_point.get().strip().split(',')))
            exit_ = tuple(map(int, self.exit_point.get().strip().split(',')))

            path, head, crossNumber, loop_map = Agent_reduction.compute_agent_reduction(matrix, entry, exit_)
            self.loop_map = loop_map

            self.loop_colors = {}
            num_loops = len(loop_map)
            for i, loop_id in enumerate(sorted(loop_map.keys())):
                hue = (i + 1) / (num_loops + 1)
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                color = '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
                self.loop_colors[loop_id] = color

            path_list = []
            current_node = head
            while current_node:
                r, c = current_node.data
                pt_type = current_node.point_identifier
                path_list.append((r, c, pt_type))
                current_node = current_node.next

            self.draw_grid(matrix, path_list, loop_map)

            self.path_text.delete("1.0", tk.END)
            formatted_path = "\n".join([f"{point}" for point in path_list])
            self.path_text.insert(tk.END, formatted_path)

            turning_points = sum(
                1 for i in range(1, len(path_list) - 1)
                if (path_list[i - 1][0] - path_list[i][0], path_list[i - 1][1] - path_list[i][1]) != (
                path_list[i][0] - path_list[i + 1][0], path_list[i][1] - path_list[i + 1][1])
            )

            agents_needed = sum(1 for point in path_list if point[2] == "agent")

            self.original_points_label.config(text=f"Total number of turning points: {turning_points}")
            self.agents_needed_label.config(text=f"Total number of agents needed after reduction: {agents_needed}")
            self.crossing_number_label.config(text=f"Total number of crossings: {crossNumber}")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
    def draw_grid(self, matrix, path, loop_map):
        self.canvas.delete("all")
        self.path_for_animation = path
        self.canvas_path_items = []

        rows, cols = len(matrix), len(matrix[0])
        self.cell_size = max(30, min(50, 800 // max(rows, cols)))
        canvas_width = cols * self.cell_size
        canvas_height = rows * self.cell_size
        self.canvas.config(width=canvas_width, height=canvas_height)

        # Draw grid and coordinates
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = (c + 1) * self.cell_size, (r + 1) * self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", fill="white")
                self.canvas.create_text(x2 - 3, y2 - 3, anchor="se", text=f"({r},{c})", font=("Arial", 7), fill="gray")
                if self.show_matrix_overlay.get():
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2,
                                            text=str(matrix[r][c]), font=("Arial", 10), tags="overlay")

        # Assign colors to loops
        self.loop_colors = {}
        self.loop_path_set = set()
        num_loops = len(loop_map)
        for i, loop_id in enumerate(sorted(loop_map.keys())):
            hue = (i + 1) / (num_loops + 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            color = '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
            self.loop_colors[loop_id] = color
            print(f"🖌️ Assigning color {color} to Loop ID #{loop_id}")

            path_pts = loop_map[loop_id]["path"]
            for j in range(len(path_pts)):
                pt1 = path_pts[j]
                pt2 = path_pts[(j + 1) % len(path_pts)]
                self.loop_path_set.add(tuple(sorted([pt1, pt2])))

        # Draw agents (blue dots)
        for r, c, pt_type in path:
            if pt_type == "agent":
                x = c * self.cell_size + self.cell_size // 2
                y = r * self.cell_size + self.cell_size // 2
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="blue")

        # Entry and exit points
        if path:
            entry_r, entry_c = path[0][:2]
            exit_r, exit_c = path[-1][:2]
            self._draw_marker(entry_r, entry_c, "green")
            self._draw_marker(exit_r, exit_c, "red")

        if self.show_matrix_overlay.get():
            self.canvas.tag_raise("overlay")

        # Begin animation
        self.animation_running = True
        self.animate_full_path(index=0)

    def _draw_marker(self, r, c, color):
        x = c * self.cell_size + self.cell_size // 2
        y = r * self.cell_size + self.cell_size // 2
        radius = 5
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

    def animate_full_path(self, index=0):
        if not self.animation_running:
            return  # Stop if animation is disabled (e.g., on clear)

        if index == 0:
            for item in getattr(self, "canvas_path_items", []):
                self.canvas.delete(item)
            self.canvas_path_items = []

        if index >= len(self.path_for_animation) - 1:
            self.animation_job = self.root.after(1600, lambda: self.animate_full_path(0))
            return

        self.draw_next_segment(index)
        self.animation_job = self.root.after(100, lambda: self.animate_full_path(index + 1))

    def draw_next_segment(self, index):
        r1, c1, _ = self.path_for_animation[index]
        r2, c2, _ = self.path_for_animation[index + 1]
        x1, y1 = c1 * self.cell_size + self.cell_size // 2, r1 * self.cell_size + self.cell_size // 2
        x2, y2 = c2 * self.cell_size + self.cell_size // 2, r2 * self.cell_size + self.cell_size // 2

        edge = tuple(sorted([(r1, c1), (r2, c2)]))
        color = "red"

        # Check if this edge belongs to a loop
        for loop_id, loop_info in self.loop_map.items():
            loop_path = loop_info["path"]
            loop_edges = set(
                tuple(sorted([loop_path[i], loop_path[(i + 1) % len(loop_path)]])) for i in range(len(loop_path)))
            if edge in loop_edges:
                color = self.loop_colors.get(loop_id, "red")
                break

        # Arrow only at turning points
        arrow_option = None
        if index > 0 and index < len(self.path_for_animation) - 1:
            r0, c0, _ = self.path_for_animation[index - 1]
            dr1, dc1 = r1 - r0, c1 - c0
            dr2, dc2 = r2 - r1, c2 - c1
            if (dr1, dc1) != (dr2, dc2):
                arrow_option = tk.LAST

        # Determine orientation
        is_horizontal = r1 == r2
        is_vertical = c1 == c2
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        gap_size = 6

        # Track drawn edges if not already initialized
        if not hasattr(self, "drawn_edges"):
            self.drawn_edges = []

        # Check for crossing
        crossed = False
        for prev in self.drawn_edges:
            (px1, py1), (px2, py2), porient, _ = prev

            if is_vertical and porient == "horizontal":
                if min(px1, px2) < x1 < max(px1, px2) and min(y1, y2) < py1 < max(y1, y2):
                    # Cross detected — split horizontal
                    self.canvas.delete(prev[3])  # delete previous line
                    gap = gap_size
                    self.canvas_path_items.append(
                        self.canvas.create_line(px1, py1, mid_x - gap, py1, fill="black", width=8)
                    )
                    self.canvas_path_items.append(
                        self.canvas.create_line(mid_x + gap, py1, px2, py2, fill="black", width=8)
                    )
                    crossed = True
                    break

        # Draw thick black line (backbone)
        if is_horizontal or is_vertical:
            line_back = self.canvas.create_line(x1, y1, x2, y2, fill="black", width=6, arrow=arrow_option)
            self.canvas_path_items.append(line_back)

            # Draw thinner colored overlay — no arrow
            line_front = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=1)
            self.canvas_path_items.append(line_front)

            # Track new segment
            orientation = "horizontal" if is_horizontal else "vertical"
            self.drawn_edges.append(((x1, y1), (x2, y2), orientation, line_back))


# ============ Run GUI ============
if __name__ == "__main__":


    root = tk.Tk()
    app = cartesian_GUI(root)
    root.mainloop()