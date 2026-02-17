import os

from kadaster_code.data.downloader import download_images


def main() -> None:
    """Download images from WMS service."""
    json_path = "zutphen-met-labels.json"
    output_folder = "zutphen-zonder-gebouwen-map"

    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} not found")
        return

    download_images(json_path, output_folder)


if __name__ == "__main__":
    main()
