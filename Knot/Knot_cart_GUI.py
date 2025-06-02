# KnotCartGUI.py
import tkinter as tk
from tkinter import filedialog
import json

class KnotCartGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Knot Cartesian Visualizer")
        self.root.geometry("1200x800")

        # Canvas and controls
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Cartesian Path", command=self.load_cartesian_json)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def load_cartesian_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        with open(file_path, 'r') as f:
            data = json.load(f)
            self.visualize_path(data)

    def visualize_path(self, data):
        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # Extract scaling factors
        all_x = [pt[0] for pt in data["path"]]
        all_y = [pt[1] for pt in data["path"]]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        x_margin, y_margin = 50, 50
        scale_x = (width - 2 * x_margin) / (max_x - min_x + 1e-5)
        scale_y = (height - 2 * y_margin) / (max_y - min_y + 1e-5)

        def to_canvas_coords(x, y):
            cx = x_margin + (x - min_x) * scale_x
            cy = height - (y_margin + (y - min_y) * scale_y)
            return cx, cy

        path = data["path"]
        crossings = set(tuple(c) for c in data.get("crossings", []))

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            x1, y1 = to_canvas_coords(*p1)
            x2, y2 = to_canvas_coords(*p2)

            # Draw line
            if tuple(p1) in crossings or tuple(p2) in crossings:
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2, dash=(4, 2))
            else:
                self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

            # Draw points
            r = 4
            self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="black")
        # Draw last point
        x_last, y_last = to_canvas_coords(*path[-1])
        self.canvas.create_oval(x_last - 4, y_last - 4, x_last + 4, y_last + 4, fill="black")

        # Entry and exit
        x0, y0 = to_canvas_coords(*path[0])
        xN, yN = to_canvas_coords(*path[-1])
        self.canvas.create_oval(x0 - 6, y0 - 6, x0 + 6, y0 + 6, fill="green")
        self.canvas.create_oval(xN - 6, yN - 6, xN + 6, yN + 6, fill="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = KnotCartGUI(root)
    root.mainloop()