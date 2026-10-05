import os
import argparse
from dataset import scan_directory_to_dataframe


def main():
    parser = argparse.ArgumentParser(description="Generate annotations.csv by scanning image folders")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to your raw images directory")
    parser.add_argument("--output-csv", type=str, default="dataset/annotations.csv", help="Where to save the CSV file")
    args = parser.parse_args()

    print(f"[+] Scanning directory: {args.input_dir}")
    df = scan_directory_to_dataframe(args.input_dir)

    if len(df) == 0:
        print("[!] No image files (.jpg, .png, .jpeg, etc.) were found.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[OK] Successfully indexed {len(df)} images.")
    print(f"[OK] Generated CSV saved to: {args.output_csv}")
    print("\nSample preview:")
    print(df.head())


if __name__ == "__main__":
    main()
