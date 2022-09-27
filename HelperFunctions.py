"""
A module for separating auxiliary functions from the main modules
"""
from typing import List


def MethodComparison(List1: List, List2: List):
    """
    List1 are available methods and List2 (input) will be checked against List1.

    Args:
        List1: list with available methods
        List2: list of input methods

    Returns:
        None

    Exceptions:
        Raised when methods dont mathc
    """
    for m in List2:
        if m not in List1:
            raise Exception(
                f"{m} is not implemented. Please check spelling of seleted methods or documentation of available methods"
            )
    return
