def normalize_slug(name: str) -> str:
    return (
        (
            name.replace(".tar.gz", "")
            .replace(".zip", "")
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "__")
        )
        .lower()
        .strip()
    )
