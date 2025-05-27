import os
import re
import logging

MAIN_TEX_PATH = "main.tex"
CHAPTER_DIR = "main"


def extract_main_chapters(main_tex_path):
    """Extract filenames from \\input{} lines between \\mainmatter and \\appendix."""
    with open(main_tex_path, encoding='utf-8') as f:
        content = f.read()

    # Find the mainmatter-to-appendix block
    match = re.search(r'\\mainmatter(.*?)\\appendix', content, re.DOTALL)
    if not match:
        raise ValueError("Could not locate the mainmatter-to-appendix block in the LaTeX source.")

    input_lines = re.findall(r'\\input\{main/(.*?)\}', match.group(1))
    return input_lines


def strip_latex_commands(text):
    """Remove LaTeX markup and retain visible text including footnotes, math, and minted blocks.
        Explicitly removes TikZ environments from character count."""
    # Remove comments
    text = re.sub(r'%.*', '', text)

    # Remove TikZ environments entirely
    text = re.sub(r'\\begin\{tikzpicture}.*?\\end\{tikzpicture\}', '', text, flags=re.DOTALL)

    #remove figures and tables
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.DOTALL)

    # Keep footnote text
    text = re.sub(r'\\footnote\{([^}]*)\}', r' \1 ', text)

    # Keep inline/display math content
    text = re.sub(r'\\\[(.*?)\\\]', r' \1 ', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r' \1 ', text, flags=re.DOTALL)

    # Keep captions
    text = re.sub(r'\\caption\{([^}]*)\}', r' \1 ', text)

    # Keep text inside minted/code environments
    text = re.sub(r'\\begin\{minted}.*?\}(.*?)\\end\{minted\}', r' \1 ', text, flags=re.DOTALL)

    # Remove begin/end for non-verbatim environments
    text = re.sub(r'\\begin\{.*?\}|\\end\{.*?\}', '', text)

    # Remove LaTeX commands but preserve arguments if useful
    text = re.sub(r'\\[a-zA-Z@]+\*?(\[[^\]]*\])?\{([^}]*)\}', r' \2 ', text)
    text = re.sub(r'\\[a-zA-Z@]+', '', text)

    # Remove leftover braces and collapse whitespace
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def setup_logger(log_path="count_chars.log"):
    """Set up a logger that overwrites the log file each run."""
    logger = logging.getLogger("count_chars")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def count_characters(tex_files, base_dir, logger=None):
    total_chars = 0
    for filename in tex_files:
        path = os.path.join(base_dir, f"{filename}.tex")
        if not os.path.isfile(path):
            msg = f"⚠️ Skipping missing file: {path}"
            print(msg)
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            continue

        with open(path, encoding='utf-8') as f:
            raw = f.read()
            clean = strip_latex_commands(raw)
            char_count = len(clean)
            total_chars += char_count
            msg = f"{filename}.tex: {char_count} chars"
            if logger:
                logger.info(msg)
            else:
                print(msg)

    return total_chars


if __name__ == "__main__":
    logger = setup_logger()
    chapter_files = extract_main_chapters(MAIN_TEX_PATH)
    total = count_characters(chapter_files, CHAPTER_DIR, logger=logger)
    print("\n")
    print("==============================")
    print(f"✅ Total characters (main body): {total:,}")
    print("==============================")
    print("\n")
    logger.info(f"Total characters (main body): {total:,}")