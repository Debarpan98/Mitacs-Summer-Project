import py_wsi
import py_wsi.imagepy_toolkit as tk

file_dir = '/home/fabian/projects/def-erangauk-ab/fabian/liver-data/py_wsi/data/images/'
db_location = '/home/fabian/projects/def-erangauk-ab/fabian/liver-data/py_wsi/db/images/'
#file_dir = 'data/py_wsi/data/images/'
#db_location = 'data/py_wsi/db/images/'
xml_dir = file_dir
db_name = "patch_db"

patch_size = 256
level = 14 # magnification
overlap = 0

turtle = py_wsi.Turtle(file_dir, db_location, db_name,  storage_type='disk')
#print(turtle.retrieve_tile_dimensions('01_01_0083.svs',patch_size=256))
turtle.sample_and_store_patches(patch_size,level,overlap,limit_bounds=True)
#patches, coords, classes, labels = turtle.get_patches_from_file("01_01_0083.svs", verbose=True)
#tk.show_labeled_patches(patches, classes)
