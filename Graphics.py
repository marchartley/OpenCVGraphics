import os.path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Cv2Image = np.ndarray
Size = Tuple[int, int]
class Graphic:
    """
    A versatile image manipulation class that supports various image operations using the OpenCV library. It provides functionality to handle and manipulate images including resizing, cropping, drawing shapes, text, and applying a variety of filters and transformations to adjust the visual appearance of images.

    Attributes:
        image (np.ndarray): The current working image that is being manipulated.
        _initial_image (np.ndarray): The original image stored for reset purposes.

    Methods:
        Various methods for image manipulation such as drawing, transforming, and applying effects are provided, which facilitate complex graphic operations suited for GUIs, games, or image processing applications.
    """

    def __init__(self, image: Union[Cv2Image, Size, 'Graphic'] = (0, 0)):
        """
        Initializes a new Graphic object with either an image or a size.
        :param image: A numpy array representing an image or a tuple for size (width, height)
        """
        if isinstance(image, Graphic):
            self.image = image.image
            self._initial_image = image._initial_image
            return
        self.image = None
        self._initial_image = None
        self.reset_image(image)

    def width(self):
        """
        Returns the width of the image.
        :return: The width of the image as an integer
        """
        return self.image.shape[1]

    def height(self):
        """
        Returns the height of the image.
        :return: The height of the image as an integer
        """
        return self.image.shape[0]

    def reset_image(self, image: Union[Cv2Image, Size] = None):
        """
        Resets the image to a given image or reverts to the initial image if none is provided.
        :param image: The new image or dimensions to reset to (default None)
        :return: Self for chaining
        """
        if image is not None:
            if isinstance(image, Cv2Image):
                self._initial_image = image.copy()
            else:
                self._initial_image = image
        else:
            if isinstance(image, Cv2Image):
                image = self._initial_image.copy()
            else:
                image = self._initial_image

        if isinstance(image, Cv2Image):
            if image.shape[2] == 3:
                self.image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2BGRA)
                self.image[:,:,3] = 255
            else:
                self.image = image.copy()
        else:
            self.image = np.zeros((image[1], image[0], 4), dtype=np.uint8) # Consider that "image" is in fact a Tuple(width, height)
            self.image[:,:,3] = 0
        return self

    def add_mask(self, mask_image: Cv2Image, use_values_between_0_and_1: bool = False):
        """
        Adds a mask to the image, altering its transparency.
        :param mask_image: A mask image array
        :param use_values_between_0_and_1: If true, mask values should be between 0 and 1, otherwise values should be between 0 and 255 (default False)
        """
        if len(mask_image.shape) == 3:
            mask_image = np.reshape(mask_image, (mask_image.shape[0], mask_image.shape[1]))
        if use_values_between_0_and_1:
            mask_image = np.clip(mask_image * 255, 0, 255).astype(np.uint8)
        self.image[:,:,3] = (255 - mask_image)
        return self

    def _blend_with_alpha(self, position, color, alpha):
        """
        Blends a color into the image at the specified position with alpha transparency.
        :param position: Pixel position for blending
        :param color: Color value to blend
        :param alpha: Transparency level (default 1.0)
        """
        overlay = self.image.copy()
        overlay[position] = color
        cv2.addWeighted(src1=overlay, alpha=alpha, src2=self.image, beta=1 - alpha, gamma=0, dst=self.image)

    def get_image(self):
        """
        Returns the current image.
        :return: The current image as a numpy array
        """
        return self.image

    def fill(self, color, alpha=1.0):
        """
        Fills the entire image with a specified color and transparency.
        :param color: Color to fill the image with
        :param alpha: Transparency level (default 1.0)
        :return: Self for chaining
        """
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        temp_image[:, :] = color
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self

    def draw_rectangle(self, start_point, end_point, color, thickness=2, alpha=1.0):
        """
        Draws a rectangle on the image.
        :param start_point: Top-left corner of the rectangle
        :param end_point: Bottom-right corner of the rectangle
        :param color: Rectangle color
        :param thickness: Line thickness (default 2)
        :param alpha: Transparency level (default 1.0)
        :return: Self for chaining
        """
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        cv2.rectangle(temp_image, start_point, end_point, color, thickness)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self

    def draw_circle(self, center, radius, color, thickness=2, alpha =1.0):
        """
        Draws a circle on the image.
        :param center: Center of the circle
        :param radius: Radius of the circle
        :param color: Circle color
        :param thickness: Line thickness (default 2)
        :param alpha: Transparency level (default 1.0)
        :return: Self for chaining
        """
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        cv2.circle(temp_image, center, radius, color, thickness)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self

    def draw_text(self, text, position, font_path, font_size, color, alpha = 1.0, center = False, box_size = (0, 0)):
        """
        Draws text on the image.
        :param text: Text to draw
        :param position: Position to start the text
        :param font_path: Path to the font file
        :param font_size: Size of the font
        :param color: Text color
        :param alpha: Transparency level (default 1.0)
        :param center: Whether to center text (default False)
        :param box_size: Box size for centering text (default (0, 0))
        :return: Self for chaining
        """
        # Convert BGR (OpenCV) image to RGB (Pillow)
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        rgb_image = cv2.cvtColor(temp_image, cv2.COLOR_BGRA2RGBA)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        if not font_path.endswith(".ttf"):
            font_path += ".ttf"
        if not os.path.exists(font_path):
            font_path = os.path.dirname(__file__) + "/Fonts/" + font_path
        font = ImageFont.truetype(font_path, font_size)
        if center:
            _, _, w, h = draw.textbbox((0, 0), text, font=font)
            position = (int(position[0] + (box_size[0]/2 - w/2)), int(position[1] + (box_size[1]/2 - h/2)))
        draw.text(position, text, font=font, fill=color)

        temp_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self


    def draw_arrow(self, start_point, end_point, color, thickness=2, tip_length=0.1, alpha = 1.0):
        """
        Draws an arrow on the image.
        :param start_point: Starting point of the arrow
        :param end_point: Ending point of the arrow
        :param color: Arrow color
        :param thickness: Line thickness (default 2)
        :param tip_length: Length of the arrow tip (default 0.1)
        :param alpha: Transparency level (default 1.0)
        :return: Self for chaining
        """
        temp_image = self.image.copy()
        if len(color) == 3: color = (*color, 255)
        cv2.arrowedLine(temp_image, start_point, end_point, color, thickness, tipLength=tip_length)
        cv2.addWeighted(temp_image, alpha, self.image, 1 - alpha, 0, self.image)
        return self


    def resize(self, new_size, interpolation=cv2.INTER_LINEAR):
        """
        Resizes the image to a new size.
        :param new_size: New dimensions (width, height)
        :param interpolation: Interpolation method (default cv2.INTER_LINEAR)
        :return: Self for chaining
        """
        self.image = cv2.resize(self.image, new_size, interpolation=interpolation)
        return self

    def crop(self, start_point, end_point):
        """
        Crops the image to a specified rectangle.
        :param start_point: Top-left corner of the crop rectangle
        :param end_point: Bottom-right corner of the crop rectangle
        :return: Self for chaining
        """
        self.image = self.image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        return self

    def rotate(self, angle, center=None, scale=1.0):
        """
        Rotates the image around a specified center and angle.
        :param angle: Rotation angle in degrees
        :param center: Center point for rotation (default None means center of image)
        :param scale: Scale factor (default 1.0)
        :return: Self for chaining
        """
        (h, w) = self.image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image = cv2.warpAffine(self.image, M, (w, h))
        return self

    def get_box(self):
        """
        Returns the corners of the image (top-left, top-right, bottom-right, bottom-left)
        :return: List of corner coordinates as a numpy array
        """
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([self.width(), self.height()])

    def apply_perspective_transform(self, src_points, dst_points, size, interpolation = None):
        """
        Applies a perspective transform to the image.
        :param src_points: Source points for the transform
        :param dst_points: Destination points for the transform
        :param size: Size of the output image
        :param interpolation: Interpolation method (default None)
        :return: Self for chaining
        """
        matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype='float32'),
                                             np.array(dst_points, dtype='float32'))
        self.image = cv2.warpPerspective(self.image, matrix, size, flags=interpolation)
        return self

    def adjust_brightness(self, value):
        """
        Adjusts the brightness of the image.
        :param value: Value to add to the brightness
        :return: Self for chaining
        """
        self.image = cv2.add(self.image, np.array([value, value, value]))
        return self

    def adjust_contrast(self, factor):
        """
        Adjusts the contrast of the image.
        :param factor: Multiplicative factor for contrast adjustment
        :return: Self for chaining
        """
        self.image = cv2.multiply(self.image, np.array([factor, factor, factor]))
        return self

    def adjust_saturation(self, factor):
        """
        Adjusts the saturation of the image.
        :param factor: Multiplicative factor for saturation adjustment
        :return: Self for chaining
        """
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = cv2.multiply(hsv_image[..., 1], np.array([factor]))
        self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return self

    def convert_color_space(self, conversion_code):
        """
        Converts the image to a different color space.
        :param conversion_code: OpenCV color conversion code
        :return: Self for chaining
        """
        self.image = cv2.cvtColor(self.image, conversion_code)
        return self

    def convert_to_grayscale(self):
        """
        Converts the image to a grayscale version
        :return: Self for chaining
        """
        self.image = cv2.cvtColor(cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2BGRA)
        return self


    def apply_blur(self, kernel_size=(5, 5)):
        """
        Applies a Gaussian blur to the image.
        :param kernel_size: Size of the blur kernel (default (5, 5))
        :return: Self for chaining
        """
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return self

    def apply_sharpen(self):
        """
        Applies a sharpening effect to the image using a predefined kernel.
        :return: Self for chaining
        """
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    def apply_edge_detection(self):
        """
        Applies an edge detection filter to the image (Canny).
        :return: Self for chaining
        """
        self.image = cv2.cvtColor(cv2.Canny(self.image, 100, 200), cv2.COLOR_GRAY2BGRA)
        return self

    def apply_custom_filter(self, kernel: np.ndarray):
        """
        Applies a custom convolution filter to the image.
        :param kernel: The convolution kernel as a numpy array
        :return: Self for chaining
        """
        self.image = cv2.filter2D(self.image, -1, kernel)
        return self

    def apply_sketch_effect(self):
        """
        Applies a sketch effect to the image by inverting and blurring.
        :return: Self for chaining
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        inverted_img = cv2.bitwise_not(gray_image)
        blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
        inverted_blur = cv2.bitwise_not(blurred_img)
        self.image = cv2.cvtColor(cv2.divide(gray_image, inverted_blur, scale=256.0), cv2.COLOR_GRAY2BGRA)
        return self

    def apply_cartoon_effect(self):
        """
        Applies a cartoon effect by combining edge detection and bilateral filtering.
        :return: Self for chaining
        """
        alpha = self.image[:,:,3]
        color = cv2.bilateralFilter(cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR), d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=9, C=2)
        self.image = cv2.cvtColor(cv2.bitwise_and(color, color, mask=edges), cv2.COLOR_BGR2BGRA)
        self.image[:,:,3] = alpha
        return self

    def apply_painting_effect(self):
        """
        Applies a painting effect using stylization.
        :return: Self for chaining
        """
        alpha = self.image[:,:,3]
        self.image = cv2.cvtColor(cv2.stylization(cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR), sigma_s=60, sigma_r=0.6), cv2.COLOR_BGR2BGRA)
        self.image[:,:,3] = alpha
        return self



class SceneRender:
    """
    A class designed to manage and composite multiple Graphic objects into a single rendered scene. This is useful for creating complex visual compositions where multiple layers (images) need to be managed and rendered onto a single canvas. Each layer can have individual properties like position and transparency which are considered during the compositing process.

    Attributes:
        base_size (Tuple[int, int]): The width and height of the base canvas where the layers will be composited.
        layers (List[Tuple[Graphic, Tuple[int, int], float]]): A list storing layers as tuples containing a Graphic object, its position, and its transparency.

    Methods:
        add_layer: Adds a new layer to the scene.
        get_image: Composites all the layers and returns the final image.
        clear: Clears all layers from the scene.
    """

    def __init__(self, base_size):
        """
        Initializes a new SceneRender object with a base canvas size.
        :param base_size: Tuple indicating the width and height of the base canvas
        """
        self.base_size = base_size
        self.layers: List[Tuple[Graphic, Tuple[int, int], float]] = []  # Stores layers as tuples of (image, position, alpha)

    def add_layer(self, image, position=(0, 0), alpha = 1.0):
        """
        Adds a graphic layer to the scene.
        :param image: A Graphic object or an image to be wrapped as a Graphic
        :param position: Position of the layer (default (0, 0))
        :param alpha: Transparency of the layer (default 1.0)
        """
        self.layers.append((image if isinstance(image, Graphic) else Graphic(image), position, alpha))

    def get_image(self):
        """
        Composites all layers into a final rendered image.
        :return: The final composited image as a numpy array
        """
        final_image = np.zeros((self.base_size[1], self.base_size[0], 3), dtype=np.uint8)

        for imageManager, position, alpha_coef in self.layers:
            image = imageManager.get_image().copy()

            # Starting positions may be negative
            start_x = int(position[0])
            start_y = int(position[1])

            # Adjust start_x and start_y if they are negative
            if start_x < 0:
                image = image[:, -start_x:]  # Crop the image from the left
                start_x = 0
            if start_y < 0:
                image = image[-start_y:, :]  # Crop the image from the top
                start_y = 0

            if start_x >= self.base_size[0]:
                continue
            if start_y >= self.base_size[1]:
                continue

            # Determine the region of the canvas that the layer will occupy
            end_x = start_x + image.shape[1]
            end_y = start_y + image.shape[0]

            # Ensure the image does not go out of the canvas boundaries
            if end_x >= self.base_size[0]:
                image = image[:, :(self.base_size[0] - start_x)]
                end_x = self.base_size[0]
            if end_y >= self.base_size[1]:
                image = image[:(self.base_size[1] - start_y), :]
                end_y = self.base_size[1]

            # Check if the image layer has an alpha channel
            if image.shape[2] >= 4:
                alpha = alpha_coef * (image[:, :, 3] / 255.0).reshape((image.shape[0], image.shape[1], 1))
                blend_area = final_image[start_y:end_y, start_x:end_x]
                image_rgb = image[:, :, :3]
                final_image[start_y:end_y, start_x:end_x] = (image_rgb * alpha + blend_area * (1 - alpha)).astype(np.uint8)
            else:
                final_image[start_y:end_y, start_x:end_x] = image

        return final_image

    def clear(self):
        """
        Clears all layers from the scene.
        """
        self.layers.clear()



#
#
# def main():
#     # Create a blank white image
#     height, width = 400, 400
#
#     cap = cv2.VideoCapture(0)
#
#     smiley = cv2.imread("smiley.png", cv2.IMREAD_UNCHANGED)
#
#     while cap.isOpened():
#         ret, image = cap.read()
#         if not ret:
#             break
#         image = cv2.resize(image, (width, height))
#
#         layers = SceneRender((300, 300))
#
#         # Initialize the Drawing class
#         drawing = Graphic((400, 400))
#         # Initialize the Drawing class
#         transforms = Graphic(image.copy())
#
#         sprite = Graphic(smiley)
#
#         # Draw shapes and text
#         drawing.draw_rectangle((0, 0), (400, 400), (255, 255, 255), -1)
#         drawing.draw_rectangle((50, 50), (150, 150), (255, 0, 0), 5, alpha = 0.5)
#         drawing.draw_circle((200, 200), 50, (0, 255, 255), -1, alpha=0.5)
#         drawing.draw_circle((250, 200), 50, (0, 255, 0), -1, alpha=0.5)
#         drawing.draw_circle((300, 200), 50, (0, 255, 0), -1, alpha=0.5)
#         drawing.draw_text('Hello, OpenCV!', (50, 250), 'Hollster.ttf', 30, (0, 0, 0), alpha = 0.8)
#         drawing.draw_arrow((100, 200), (300, 300), (0, 0, 255), 3, 0.3)
#
#         sprite.resize((300, 300), cv2.INTER_NEAREST)
#
#
#         transforms.apply_perspective_transform(
#             [(0, 0), (width, 0), (width, height), (0, height)],  # source points
#             [(0, height // 3), (width, 0), (width, height), (0, 2 * height // 3)],  # destination points
#             (width, height)
#         ).apply_sketch_effect()
#
#         # layers.add_layer(transforms, (0, 0), 0.5)
#         layers.add_layer(drawing, (0, 0))
#         # layers.add_layer(sprite, (0, 0))
#
#         # Display the image
#         cv2.imshow('Drawing', layers.get_image())
#
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q") or key == 27:
#             break
#     cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)

    original_graphic = Graphic()
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        # Create a Graphic object for the original image
        original_graphic.reset_image(img)
        original_graphic.resize((100, 100))

        # Create various transformations and effects
        gray_graphic = Graphic(original_graphic).convert_to_grayscale()
        blurred_graphic = Graphic(original_graphic).apply_blur((5, 5)).rotate(90)
        edge_graphic = Graphic(original_graphic).apply_edge_detection()
        cartoon_graphic = Graphic(original_graphic).apply_cartoon_effect()
        painting_graphic = Graphic(original_graphic).apply_painting_effect()
        sketch_graphic = Graphic(original_graphic).apply_sketch_effect()

        # Set the size for the scene render based on image width and number of transformations
        render_width = original_graphic.width() * 3  # Display 7 images side by side
        render_height = max(original_graphic.height(), gray_graphic.height(), blurred_graphic.height(),
                            edge_graphic.height(), cartoon_graphic.height(), painting_graphic.height(),
                            sketch_graphic.height()) * 2

        scene = SceneRender((render_width, render_height))

        # Add layers to the scene with the original and transformed images
        x_offset = 0
        y_offset = 0
        scene.add_layer(gray_graphic, (x_offset, y_offset))
        x_offset += gray_graphic.width()
        scene.add_layer(blurred_graphic, (x_offset, y_offset))
        x_offset += blurred_graphic.width()
        scene.add_layer(edge_graphic, (x_offset, y_offset))
        x_offset = 0
        y_offset += edge_graphic.height()
        scene.add_layer(cartoon_graphic, (x_offset, y_offset))
        x_offset += cartoon_graphic.width()
        scene.add_layer(painting_graphic, (x_offset, y_offset))
        x_offset += painting_graphic.width()
        scene.add_layer(sketch_graphic, (x_offset, y_offset))

        cv2.imshow("Result", scene.get_image())
        key = cv2.waitKey(30)
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
