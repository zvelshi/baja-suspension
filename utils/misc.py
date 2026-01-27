def parse_time(seconds: float) -> str:
    """
    Convert seconds to a compact human-readable string (e.g. '1d 2h 4m 3s').
    """

    seconds = int(round(seconds))
    if seconds == 0:
        return "0s"

    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)