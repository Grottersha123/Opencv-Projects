import os
import shutil as s
def copy(path_in,path_out):
    s.move(path_in, path_out)
def all_picture(path1,path2):
    a = set(os.listdir(path1)).difference(os.listdir(path2))
    return list(a)
if __name__ == '__main__':
    path_out = r'D:\Git_project\VKR\FALSE_DETEC_CARS'
    path_in = r'D:\Git_project\VKR\OUPUT_ANOTHER\ '
    all_files = all_picture(r'D:\Git_project\VKR\CARS_ANOTHER',r'D:\Git_project\VKR\OUPUT_ANOTHER')
    for i in all_files:
        path_in = 'D:\Git_project\VKR\CARS_ANOTHER\{0}'.format(i)
        copy(path_in,path_out)



