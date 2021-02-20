def make_header(text: str) -> str:
    return (
        f"{'#' * (len(text) + 4)}\n"
        f"# {text} #\n"
        f"{'#' * (len(text) + 4)}\n"
    )