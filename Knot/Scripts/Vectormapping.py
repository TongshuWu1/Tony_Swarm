import tkinter as tk
from shapely.geometry import Point, Polygon
import numpy as np

# Define square ABCD
A = (100, 300)
B = (300, 300)
C = (300, 100)
D = (100, 100)
square = Polygon([A, B, C, D])

# Define flattened triangle A'-C'-D'
A_flat = (100, 300)
C_flat = (300, 300)
D_flat = (200, 100)
triangle = Polygon([A_flat, C_flat, D_flat])

# Vector basis from square
AB = np.array(B) - np.array(A)
AD = np.array(D) - np.array(A)
M = np.column_stack((AB, AD))

# Flattened triangle edges
AC_flat = np.array(C_flat) - np.array(A_flat)
AD_flat = np.array(D_flat) - np.array(A_flat)

# GUI Setup
root = tk.Tk()
root.title("Correct Square-to-Triangle Mapping")

canvas = tk.Canvas(root, width=800, height=400, bg='white')
canvas.pack()

# Draw square on left
square_id = canvas.create_polygon([*A, *B, *C, *D], outline='blue', fill='', width=2)

# Draw triangle on right (offset)
triangle_offset = 400
A2 = (A_flat[0] + triangle_offset, A_flat[1])
C2 = (C_flat[0] + triangle_offset, C_flat[1])
D2 = (D_flat[0] + triangle_offset, D_flat[1])
triangle_id = canvas.create_polygon([*A2, *C2, *D2], outline='green', fill='', width=2)

# Store drawn points for reset
drawn_points = []

# Correct Square-to-Triangle mapping function
def map_square_to_triangle(u, v):
    base = np.array(A_flat) + u * (np.array(C_flat) - np.array(A_flat))
    height_vec = (np.array(D_flat) - np.array(A_flat)) * (1 - u)
    return base + v * height_vec

# Click event handler
def on_click(event):
    ex, ey = event.x, event.y
    E = np.array([ex, ey])

    if not square.contains(Point(E)):
        print("Click is outside the square.")
        return

    # Compute (u, v) in square space
    u, v = np.linalg.inv(M) @ (E - np.array(A))

    if not (0 <= u <= 1 and 0 <= v <= 1):
        print("Invalid square space coordinates.")
        return

    # Map to triangle space (correct warp)
    E_flat = map_square_to_triangle(u, v)
    E_flat_display = E_flat + np.array([triangle_offset, 0])

    # Draw points in both panels
    orig = canvas.create_oval(ex-3, ey-3, ex+3, ey+3, fill='red')
    mapped = canvas.create_oval(E_flat_display[0]-3, E_flat_display[1]-3, E_flat_display[0]+3, E_flat_display[1]+3, fill='red')
    drawn_points.extend([orig, mapped])

    print(f"E = ({ex:.1f}, {ey:.1f}) → E' = ({E_flat_display[0]:.1f}, {E_flat_display[1]:.1f})")

# Reset button
def reset_canvas():
    for item in drawn_points:
        canvas.delete(item)
    drawn_points.clear()
    print("Canvas reset.")

# Add reset button
reset_btn = tk.Button(root, text="Reset", command=reset_canvas)
reset_btn.pack(pady=10)

# Bind click
canvas.bind("<Button-1>", on_click)

# Run GUI
root.mainloop()
