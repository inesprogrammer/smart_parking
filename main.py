"""
main.py
-------
Command-line entry point for the Smart Parking System.

Usage examples:
  # Process all images in a folder
  python main.py --input dataset/sample_images

  # Process a single image
  python main.py --input path/to/image.jpg

  # Specify total parking spaces
  python main.py --input dataset/sample_images --spaces 20

  # Save annotated images (don't open a window)
  python main.py --input dataset/sample_images --no-display

  # Save CSV results
  python main.py --input dataset/sample_images --save-csv
"""

import argparse
import sys
import os
import cv2
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from parking_system import (
    ParkingDetector,
    load_images_from_folder,
    load_single_image,
    preprocess_image,
    save_results_to_csv,
    save_annotated_image,
    print_console_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Parking System — Detect cars, plates, and free spaces."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an image file or a folder of images."
    )
    parser.add_argument(
        "--spaces", "-s",
        type=int,
        default=10,
        help="Total number of parking spaces in the lot (default: 10)."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt, auto-downloaded)."
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save output files (default: results/)."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save detection results to a CSV file."
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save annotated images to output directory (default: True)."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open OpenCV windows to display results."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Max image width for processing (default: 1280)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    # --- Load images ---
    if input_path.is_dir():
        images = load_images_from_folder(str(input_path))
        if not images:
            logger.error("No images found. Check the input folder.")
            sys.exit(1)
    elif input_path.is_file():
        img = load_single_image(str(input_path))
        images = [(input_path.name, img)]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # --- Initialize detector ---
    detector = ParkingDetector(
        model_path=args.model,
        total_spaces=args.spaces
    )

    all_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each image ---
    for filename, image in images:
        logger.info(f"\nProcessing: {filename}")

        # Preprocess (resize if needed)
        image = preprocess_image(image, target_width=args.width)

        # Run detection
        result = detector.detect(image)
        result["filename"] = filename
        all_results.append(result)

        # Print console report
        print_console_report(result, filename)

        annotated = result["annotated_image"]

        # Save annotated image
        if args.save_images:
            save_annotated_image(
                annotated,
                filename,
                output_dir=str(output_dir / "annotated")
            )

        # Display window
        if not args.no_display:
            window_title = f"Smart Parking — {filename}"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, min(1280, annotated.shape[1]), min(720, annotated.shape[0]))
            cv2.imshow(window_title, annotated)

            print("\n  [Press any key to continue to next image, or 'q' to quit]")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord("q"):
                logger.info("Quitting early.")
                break

    # --- Save CSV ---
    if args.save_csv:
        csv_path = save_results_to_csv(
            all_results,
            output_path=str(output_dir / "parking_results.csv")
        )
        print(f"\n  ✅ CSV saved to: {csv_path}")

    # --- Final summary ---
    total_cars = sum(r["car_count"] for r in all_results)
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"  Images processed : {len(all_results)}")
    print(f"  Total cars found : {total_cars}")
    print(f"  Results saved to : {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
