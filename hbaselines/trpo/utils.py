import numpy as np
import os
import errno


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def flatten_lists(listoflists):
    """Flatten a python list of list.

    Parameters
    ----------
    listoflists : list of list
        the list of flatten

    Returns
    -------
    list
        flattened list
    """
    return [el for list_ in listoflists for el in list_]


def explained_variance(y_pred, y_true):
    """Compute fraction of variance that ypred explains about y.

    Returns 1 - Var[y-ypred] / Var[y]

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Parameters
    ----------
    y_pred : np.ndarray
        the prediction
    y_true : np.ndarray
        the expected value

    Returns
    -------
    float
        explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def fmt_row(width, row, header=False):
    """
    fits a list of items to at least a certain length

    :param width: (int) the minimum width of the string
    :param row: ([Any]) a list of object you wish to get the string representation
    :param header: (bool) whether or not to return the string as a header
    :return: (str) the string representation of all the elements in 'row', of length >= 'width'
    """
    out = " | ".join(fmt_item(x, width) for x in row)
    if header:
        out = out + "\n" + "-" * len(out)
    return out


def fmt_item(item, min_width):
    """
    fits items to a given string length

    :param item: (Any) the item you wish to get the string representation
    :param min_width: (int) the minimum width of the string
    :return: (str) the string representation of 'x' of length >= 'l'
    """
    if isinstance(item, np.ndarray):
        assert item.ndim == 0
        item = item.item()
    if isinstance(item, (float, np.float32, np.float64)):
        value = abs(item)
        if (value < 1e-4 or value > 1e+4) and value > 0:
            rep = "%7.2e" % item
        else:
            rep = "%7.5f" % item
    else:
        rep = str(item)
    return " " * (min_width - len(rep)) + rep


COLOR_TO_NUM = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize, bold and/or highlight a string for terminal print

    Parameters
    ----------
    string : str
        input string
    color : str
        the color, the lookup table is the dict at console_util.color2num
    bold : bool
        if the string should be bold or not
    highlight : bool
        if the string should be highlighted or not

    Returns
    -------
    str
        the stylized output string
    """
    attr = []
    num = COLOR_TO_NUM[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
