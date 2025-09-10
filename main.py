import argparse
from generate_wordcloud import generate_wordcloud

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wordcloud from all data.")
    parser.add_argument("csv_path", help="Path to CSV file with txtContent column")
    args = parser.parse_args()

    generate_wordcloud(args.csv_path)
