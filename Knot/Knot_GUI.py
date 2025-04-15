import tkinter as tk
from tkinter import messagebox, filedialog
import os
import region_detection as Agent_reduction


class AgentReductionGUI:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x1000")
        self.root.title("Agent Reduction - Path Optimization")

        # ============ 1) Main Scrollable Frame Setup ============
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for scrolling
        self.my_canvas = tk.Canvas(main_frame)
        self.my_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        self.my_scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.my_canvas.yview)
        self.my_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        self.my_canvas.configure(yscrollcommand=self.my_scrollbar.set)
        self.my_canvas.bind(
            "<Configure>",
            lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all"))
        )

        # Bind mouse wheel to scroll
        self.my_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Frame inside the canvas (this is our "scrollable" frame)
        self.second_frame = tk.Frame(self.my_canvas)
        self.my_canvas.create_window((0, 0), window=self.second_frame, anchor="nw")

        # ============ 2) Widgets inside second_frame ============

        # Title
        self.label_title = tk.Label(
            self.second_frame, text="Agent Reduction",
            font=("Arial", 18), fg="black", bg="lightgray"
        )
        self.label_title.pack(pady=10, fill=tk.X)

        # Matrix Name Input
        self.matrix_name_label = tk.Label(self.second_frame, text="Matrix Name:")
        self.matrix_name_label.pack()
        self.matrix_name_entry = tk.Entry(self.second_frame, width=50)
        self.matrix_name_entry.pack()

        # Matrix Input Text Area
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

        # Notes Input Text Area
        self.notes_label = tk.Label(self.second_frame, text="Notes:")
        self.notes_label.pack()

        self.notes_text = tk.Text(self.second_frame, height=4, width=50)
        self.notes_text.pack()

        # Frame for Entry and Exit Points
        self.entry_exit_frame = tk.Frame(self.second_frame)
        self.entry_exit_frame.pack(pady=5)

        tk.Label(self.entry_exit_frame, text="Entry (row, col):").pack(side=tk.LEFT)
        self.entry_point = tk.Entry(self.entry_exit_frame, width=10)
        self.entry_point.pack(side=tk.LEFT, padx=5)

        tk.Label(self.entry_exit_frame, text="Exit (row, col):").pack(side=tk.LEFT)
        self.exit_point = tk.Entry(self.entry_exit_frame, width=10)
        self.exit_point.pack(side=tk.LEFT, padx=5)

        # Run Button
        self.run_button = tk.Button(self.second_frame, text="Run Agent Reduction", command=self.run_algorithm)
        self.run_button.pack(pady=10)

        # Frame to hold canvas and path output side by side
        self.result_frame = tk.Frame(self.second_frame)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Canvas for Path Visualization
        self.canvas = tk.Canvas(self.result_frame, width=400, height=400, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10)

        # Frame for the right-side output (Path)
        self.output_frame = tk.Frame(self.result_frame)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Path Output Label
        self.path_title_label = tk.Label(self.output_frame, text="Path from Entry to Exit:")
        self.path_title_label.pack()

        # Checkbox for toggling matrix overlay
        self.show_matrix_overlay = tk.BooleanVar()
        self.matrix_overlay_checkbox = tk.Checkbutton(self.second_frame, text="Show Matrix Overlay",
                                                      variable=self.show_matrix_overlay, command=self.run_algorithm)
        self.matrix_overlay_checkbox.pack(pady=5)

        # Path Text Area
        self.path_text = tk.Text(self.output_frame, height=20, width=40)
        self.path_text.pack(fill=tk.BOTH, expand=True)

        # Labels for summary info
        self.original_points_label = tk.Label(self.output_frame, text="Total number of original points: ")
        self.original_points_label.pack()

        self.agents_needed_label = tk.Label(self.output_frame, text="Total number of agents needed after reduction: ")
        self.agents_needed_label.pack()

        # Label for crossing number
        self.crossing_number_label = tk.Label(self.output_frame, text="Total number of crossings: ")
        self.crossing_number_label.pack()

    def _on_mousewheel(self, event):
        self.my_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

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
                self.matrix_name_entry.insert(0, file_name.split('/')[-1].split('.')[0])
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
                    notes = lines[2].strip().split(": ")[1] if len(lines) > 2 else ""
                    matrix_str = "".join(lines[3:])

                self.matrix_text.delete("1.0", tk.END)
                self.matrix_text.insert(tk.END, matrix_str)
                self.notes_text.delete("1.0", tk.END)
                self.notes_text.insert(tk.END, notes)
                self.entry_point.delete(0, tk.END)
                self.entry_point.insert(0, entry)
                self.exit_point.delete(0, tk.END)
                self.exit_point.insert(0, exit_)
                self.matrix_name_entry.delete(0, tk.END)
                self.matrix_name_entry.insert(0, file_path.split('/')[-1].split('.')[0])
                messagebox.showinfo("Success", "Matrix loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load matrix: {e}")

    def run_algorithm(self):
        """
        Reads the matrix, entry, and exit from GUI.
        Runs the agent reduction algorithm from Agent_reduction.
        Shows the results in the path_text widget and draws the path on self.canvas.
        """
        try:
            # Read matrix input
            matrix_str = self.matrix_text.get("1.0", tk.END).strip()
            matrix_lines = matrix_str.split("\n")
            matrix = [list(map(int, line.split(','))) for line in matrix_lines if line.strip()]

            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0

            # Read entry and exit points
            entry = tuple(map(int, self.entry_point.get().split(',')))
            exit_ = tuple(map(int, self.exit_point.get().split(',')))

            # Validate coordinates
            if not (0 <= entry[0] < rows and 0 <= entry[1] < cols):
                messagebox.showerror("Error", "Invalid Entry Point")
                return
            if not (0 <= exit_[0] < rows and 0 <= exit_[1] < cols):
                messagebox.showerror("Error", "Invalid Exit Point")
                return

            # Check if entry and exit points are valid (either 1 or -1)
            if matrix[entry[0]][entry[1]] not in (1, -1):
                messagebox.showerror("Error", "Entry point must be either 1 or -1")
                return
            if matrix[exit_[0]][exit_[1]] not in (1, -1):
                messagebox.showerror("Error", "Exit point must be either 1 or -1")
                return

            # Run the actual (updated) agent reduction algorithm
            path, head, crossNumber = Agent_reduction.compute_agent_reduction(matrix, entry, exit_)

            # Convert linked list to a list of (row, col, 'agent' or 'path')
            path_list = []
            current_node = head
            while current_node:
                r, c = current_node.data
                pt_type = current_node.point_identifier
                path_list.append((r, c, pt_type))
                current_node = current_node.next

            # Reverse path_list if needed
            path_list.reverse()

            # Draw Path on the canvas
            self.draw_grid(matrix, path_list)

            # Show results
            self.path_text.delete("1.0", tk.END)
            formatted_path = "\n".join([f"{point}" for point in path_list])
            self.path_text.insert(tk.END, formatted_path)

            original_points = len(path_list)
            agents_needed = sum(1 for point in path_list if point[2] == "agent")

            self.original_points_label.config(text=f"Total number of original points: {original_points}")
            self.agents_needed_label.config(text=f"Total number of agents needed after reduction: {agents_needed}")
            self.crossing_number_label.config(text=f"Total number of crossings: {crossNumber}")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def draw_grid(self, matrix, path):
        """
        Draw the matrix with the path on the canvas.
        `path` is a list of (row, col, type).
        """
        self.canvas.delete("all")  # Clear previous drawings

        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0

        # Adjust cell size
        cell_size = min(400 // max(rows, 1), 400 // max(cols, 1)) if rows and cols else 40

        # Draw the grid
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = (c + 1) * cell_size, (r + 1) * cell_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")
                self.canvas.create_text(x1 + 5, y1 + 5, anchor=tk.NW, text=f"({r},{c})", font=("Arial", 8))

                if self.show_matrix_overlay.get():
                    self.canvas.create_text(x1 + cell_size // 2, y1 + cell_size // 2, text=f"{matrix[r][c]}",
                                            font=("Arial", 16), fill="black", tags="overlay")

        # Draw the path
        for i in range(len(path) - 1):
            (r1, c1, t1) = path[i]
            (r2, c2, t2) = path[i + 1]

            x1, y1 = c1 * cell_size + cell_size // 2, r1 * cell_size + cell_size // 2
            x2, y2 = c2 * cell_size + cell_size // 2, r2 * cell_size + cell_size // 2

            # Determine cross type
            if matrix[r1][c1] == -1 and matrix[r2][c2] == 1:
                cross_type = 'over'
            elif matrix[r1][c1] == 1 and matrix[r2][c2] == -1:
                cross_type = 'under'
            else:
                cross_type = None

            if cross_type == 'over':
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2, arrow=tk.LAST)
            elif cross_type == 'under':
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2, dash=(4, 2), arrow=tk.LAST)
            else:
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2, arrow=tk.LAST)

            if t1 == "agent":
                radius = 3
                self.canvas.create_oval(x1 - radius, y1 - radius, x1 + radius, y1 + radius, fill="blue")

        # Bring overlay text to front
        if self.show_matrix_overlay.get():
            self.canvas.tag_raise("overlay")



# ============ Run GUI ============
if __name__ == "__main__":
    root = tk.Tk()
    app = AgentReductionGUI(root)
    root.mainloop()