# OpenCV-Graphics
 Small Python graphic library for educational purposes

This repository demonstrates how to create and manipulate graphics using OpenCV in Python. It includes functionality for drawing shapes, adding text, and applying effects to images captured in real-time from a webcam.

## Features

- Real-time graphics manipulation and rendering.
- Use of custom graphic elements like shapes, text, and imported images.
- Application of effects such as sketching and adding masks to images.
- Layer-based scene rendering for complex graphical outputs.

## Prerequisites

To run this application, you will need to install the following:

- `opencv-contrib-python`: For all image processing tasks.
- `numpy`: For numerical operations.

These packages can be installed via pip:

```bash
pip install opencv-contrib-python numpy
```

## Installation

Clone this repository to your local machine to get started:

```bash
git clone https://github.com/marchartley/OpenCV-Graphics.git
cd OpenCV-Graphics
```

## Usage

To run the program, execute the `main()` function in your Python environment. This will start the application which uses the default webcam as the video source.

```python
python3 main.py  
```

## Application Details

- **Graphic**: A class to encapsulate image data, allowing transformations and effects to be applied easily.
- **SceneRender**: Manages the rendering of multiple layers, such as backgrounds, images, and text, into a single output scene.
- The application window will display the resulting graphical manipulations in real-time. The rendering stack includes a background canvas, a real-time webcam feed with effects, and an overlaid image.

## Exiting the Application

- To exit the application, press the "q" key or the ESC key while the output window is active.

## Contributing

This project has no ambition to be used for more than education purposes. However, contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
