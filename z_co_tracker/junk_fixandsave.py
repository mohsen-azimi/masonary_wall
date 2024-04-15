import cv2


def fix_and_save_image(input_image_path, output_image_path, output_size=(512, 384)):
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Resize the input image to match the output size
    resized_image = cv2.resize(input_image, (output_size[1], output_size[0]))  # Swapping height and width

    # Save the resized image as PNG
    cv2.imwrite(output_image_path, resized_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"Image resized and saved to {output_image_path} successfully.")


# Usage
input_image_path = "./assets/leila_2.png"
output_image_path = "./assets/leila_2.png"
fix_and_save_image(input_image_path, output_image_path)
