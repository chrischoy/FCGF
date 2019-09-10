import os
import re
from os import listdir
from os.path import isfile, isdir, join, splitext


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path, mode=0o755)


def sorted_alphanum(file_list_ordered):

  def convert(text):
    return int(text) if text.isdigit() else text

  def alphanum_key(key):
    return [convert(c) for c in re.split('([0-9]+)', key)]

  return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  else:
    file_list = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and splitext(f)[1] == extension
    ]
  file_list = sorted_alphanum(file_list)
  return file_list


def get_file_list_specific(path, color_depth, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  else:
    file_list = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and color_depth in f and splitext(f)[1] == extension
    ]
    file_list = sorted_alphanum(file_list)
  return file_list


def get_folder_list(path):
  folder_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
  folder_list = sorted_alphanum(folder_list)
  return folder_list
