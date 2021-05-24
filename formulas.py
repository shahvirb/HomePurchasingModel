import numpy as np

def compound_interest_amount(p, r, n, t):
    """
    >>> '%.2f' % compound_interest_amount(100, 0.1, 1, 1)
    '110.00'
    >>> compound_interest_amount(100, 0.03875, 12, 7.5)
    133.66370154434335
    """
    return p * np.power((1 + r/n), n*t)

def payment(p, r, n, t):
    """
    >>> payment(100000, 0.07, 12, 30)
    665.3024951791823
    """
    return compound_interest_amount(p, r, n, t) * r/n / (np.power((1+r/n), n*t) - 1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
