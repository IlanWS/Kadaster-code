from kadaster_code.data.preprocessor import process_images


def main() -> None:
    """Process images for training."""
    input_folder = "zutphen-zonder-labels-map"
    target_folder = "zutphen-met-alleen-labels-map"

    input_array, target_array = process_images(input_folder, target_folder)
    print(f"Input array shape: {input_array.shape}")
    print(f"Target array shape: {target_array.shape}")


if __name__ == "__main__":
    main()
