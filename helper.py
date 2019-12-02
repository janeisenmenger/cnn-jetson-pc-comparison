import glob

def get_all_files_in_directory(directory, extension=''):
    '''
    A function used to recursivly extract all files with the given extension from a directory.

    :param directory: The directory we want to extract the files from.
    :param extension: The extension we want to use. All files are retrieved on default.

    :returns: The files in the directory with the given extension.
    '''
    if directory[-1] == '/':
        directory = directory[:-1]
    return glob.glob(directory + '/**/*' + extension, recursive=True)