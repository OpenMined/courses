import urllib
import os
import string
import re


def create_dirs(*paths)->None:
    """
    Creates a set of directories from a tuple of directories
    :param paths: List or tuple of strings for the paths to be created if they do not exist
    :return: void
    """
    for path in paths:

        if not os.path.exists(path):
            os.mkdir(path)

def download_data(url, filename, download_path):
    """
    Downloads a file from the specified url and persist it into download_path with specific name (filename)
    """
    file = download_path +filename
    create_dirs(download_path)
    if not os.path.isfile(file):
        print('Dowloading data...\n\nurl:{}\nfilename:{}\ndowload_path:{}'.format(url,filename,download_path))
        urllib.request.urlretrieve(url, file)


def process_load_textfile(filename, download_path):
    """
    Loads a textfile replaces all newlines with spaces and all capital letters to lower.
    only stores ascii laters 'abcdefghijklmnopqrstuvwxyz'
    """
    with open(download_path + filename) as file:
        data = ''.join(file.readlines()).replace('\n', ' ').lower()
        data = ''.join([x for x in data if x in string.ascii_lowercase + ' '])
        data = re.sub(' +', ' ', data)

    return data