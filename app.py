import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, filedialog

# Function to open file dialog for image selection
def select_image():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )
    return file_path

# Step 1: Load the Image
image_path = select_image()
if not image_path:
    print("No image selected. Exiting.")
else:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
    else:
        # Step 2: Define the Kernel
        # Example: Sobel kernel for edge detection
        sobel_kernel_x = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])

        sobel_kernel_y = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])

        # Step 3: Apply the Kernel using Convolution
        # Apply Sobel kernel in x and y directions
        edges_x = cv2.filter2D(image, -1, sobel_kernel_x)
        edges_y = cv2.filter2D(image, -1, sobel_kernel_y)

        # Combine the edges
        edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

        # Step 4: Enhance the Edges
        # Example: Use Canny edge detector for enhancement
        enhanced_edges = cv2.Canny(edges, 100, 200)

        # Step 5: Save/Display the Results
        cv2.imwrite('edges.jpg', edges)
        cv2.imwrite('enhanced_edges.jpg', enhanced_edges)

        # Display the results using matplotlib
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title('Edge Detection')
        plt.imshow(edges, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title('Enhanced Edges')
        plt.imshow(enhanced_edges, cmap='gray')

        plt.show()
