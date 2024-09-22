from bs4 import BeautifulSoup
import os
from pathlib import Path


def html_page_to_text(input_filepath, output_path):
    with open(input_filepath, "r") as f:
        html_page = f.read()

    soup = BeautifulSoup(html_page, "html.parser")
    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    text = "\n".join([line for line in text.split("\n") if len(line) > 1])

    with open(output_path, "w") as f:
        f.write(text)


def main():
    html_page_to_text(
        "documents/yandex_internship.htm", "documents/yandex_internship.txt"
    )


if __name__ == "__main__":
    main()
