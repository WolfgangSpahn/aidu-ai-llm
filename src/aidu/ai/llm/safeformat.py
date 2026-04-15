"""
    This module contains the SafeFormat class which is a helper class to allow for
    the use of format_map with missing keys.
"""


class SafeFormat(dict):
    """This is a helper class to allow for the use of format_map with missing keys
    
        Use it like this:
        >>> safe_format = SafeFormat()
        >>> safe_format["key"] = "value"
        >>> print(safe_format["key"])          # Output: value
        >>> print(safe_format["missing_key"])  # Output: {missing_key}
    """
    def __missing__(self, key: str):
        return "{" + key + "}"

if __name__ == "__main__":
    # Create a SafeFormat object
    safe_format = SafeFormat()
    # Print the missing key
    print(safe_format["key"])
    # Print the missing key with a default value
    print(safe_format.get("key", "default_value"))