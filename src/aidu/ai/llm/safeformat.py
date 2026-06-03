# Copyright (c) 2026 Wolfgang Spahn, PHBern
# Licensed under the MIT License.
# Please follow standard academic practice when using this software in research or publications.
# See LICENSE for the full text.

"""
Helper class for safe string formatting with format_map that handles missing keys gracefully.
Returns formatted placeholder instead of raising KeyError for undefined variables.
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
