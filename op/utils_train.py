# -*- coding:utf-8 -*-
# 工具包,用于常用的函数的使用
import os
import random
import shutil


def listdir(path, list_name):  # 传入存储的list
    '''
    递归得获取对应文件夹下的所有文件名的全路径
    存在list_name 中
    :param path: input the dir which want to get the file list
    :param list_name:  the file list got from path
	no return
    '''
    list_dirs = os.listdir(path)
    list_dirs.sort()
    for file in list_dirs:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    list_name.sort()
    # print(list_name)


def write_in_File(target_file_path, list_file_path):
    '''
    用于将list写进目标文件中去,每写一次添加换行
    list_file_path = 目标文件的路径的list
    target_file_path = 目标写入的文件,一般为xxx.txt
    no return
    '''
    target_file = open(target_file_path, 'w')
    for file_path in list_file_path:
        print(file_path + "write in the %s" % (target_file_path))
        target_file.write(file_path + '\n')
    target_file.close();


def shuffle_list(file_list):
    '''
    对list进行打乱操作,随机打乱
    file_list = 传入的原始list
    return shuffle 后的list
    '''
    random.shuffle(file_list)
    return file_list


def getFathPath(path):
    """
    get the father path from path
    "aaa/bbb/ccc" == >  "aaa/bbb"
    """
    father_path = os.path.abspath(os.path.dirname(path) + os.path.sep + ".")

    return father_path


def readList(target_file, split_=""):
    '''
    从目标文件中读取文件中的内容,每次去掉回车和分隔符,并返回这个list
    target_file:目标文件 一般为xxx.txt
    return file_list  是一个按行读的list
    '''
    files = open(target_file)
    file_list = files.read().strip().split(split_)
    files.close()
    return file_list


def str_2_list(str_, split_=",", type=float):
    list_ = str_.strip().split(split_)
    list_ = [type(list_[i]) for i in range(len(list_))]
    return list_


def read_line_List(target_file):
    '''
    从目标文件中读取文件中的内容,每次去掉回车和分隔符,并返回这个list
    target_file:目标文件 一般为xxx.txt
    return file_list  是一个按行读的list
    '''

    files = open(target_file, "r")  # 设置文件对象
    file_list = files.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    files.close()  # 关闭文件

    return file_list


def getNameFromPath(file_path):
    '''
    从文件的全路径中,获取对应文件的文件名
    若果输入的是文件名,也可
    '''
    # 这里做的处理就是 我的当读取的文件里面是 全路径而不是文件名时
    # 做了一个鲁棒性处理,把对应的全路径处理成文件名
    # 以原来的方式为基准
    if file_path.find("/") >= 0:
        temp_list = file_path.strip().split('/')
    else:
        temp_list = file_path.strip().split('\\')
    temp_imId = temp_list[0]
    if (len(temp_list) > 1):
        temp_imId = temp_list[-1]

    temp_list = temp_imId.split('.')
    true_id = temp_list[0]
    for i in range(1, len(temp_list) - 1):
        true_id = true_id + "." + temp_list[i]

    # 鲁棒性处理结束 实质上就是把路径削成文件名
    return true_id


def match_string(str1, str2, postfix1="", postfix2=""):
    """
    match two string,
    fist_reduce the postfix
    if str1==str2 return True
    else false
    """

    str1 = str(str1).replace(postfix1, "")
    str2 = str(str2).replace(postfix2, "")

    if str1 == str2:
        return True
    else:
        print("match_string ==>not match!")
        return False


def match_list_str(list1, list2, postfix1="", postfix2=""):
    """
    match all strings in given 2 list
    """

    if not len(list1) == len(list2):
        return False

    for i in range(len(list1)):
        tag = match_string(list1[i], list2[i], postfix1, postfix2)
        if tag == False:
            return False

    print("match_list_str ==>all match!")

    return True


def getDirFromPath(file_path):
    '''
    从文件的全路径中,获取对应文件的文件夹名
    '''
    # 这里做的处理就是 我的当读取的文件里面是 全路径而不是文件名时
    # 做了一个鲁棒性处理,把对应的全路径处理成文件名
    # 以原来的方式为基准
    if file_path.find("/") >= 0:
        temp_list = file_path.strip().split('/')
    else:
        temp_list = file_path.strip().split('\\')
    temp_imId = temp_list[0]
    if (len(temp_list) > 1):
        temp_imId = temp_list[-1]

    temp_list = temp_imId.split('.')
    true_id = temp_list[0]
    if (len(temp_list) > 1):
        true_id = temp_list[0]
    # 鲁棒性处理结束 实质上就是把路径削成文件名
    return true_id


def getTypeFromPath(file_path):
    '''
    :param file_path: filepath or file name
    :return:  the type by string   etc  jpg \ xml ...
    '''
    # 这里做的处理就是 我的当读取的文件里面是 全路径而不是文件名时
    # 做了一个鲁棒性处理,把对应的全路径处理成文件名
    # 以原来的方式为基准
    if file_path.find("/") >= 0:
        temp_list = file_path.strip().split('/')
    else:
        temp_list = file_path.strip().split('\\')
    temp_imId = temp_list[0]
    if (len(temp_list) > 1):
        temp_imId = temp_list[-1]

    temp_list = temp_imId.split('.')
    true_id = temp_list[-1]
    if (len(temp_list) > 1):
        true_id = temp_list[-1]
    # 鲁棒性处理结束 实质上就是把路径削成文件名
    return true_id


def mkdir(path):
    """
    check the path and mkdir for given path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    print("make dir for %s!" % path)


def movefile2Dir(srcfile, dstDir):
    '''
    move file into target dir and save the original file name

    srcfile = 目标文件
    dstDir = 目标文件夹
    将目标文件移动至目标文件夹
    '''
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        # fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)  # 创建路径
        shutil.move(srcfile, dstDir)  # 移动文件
        print("move %s -> %s" % (srcfile, dstDir))


def copyfile2Dir(srcfile, dstDir):
    '''
    copy file into target dir and save the original file name
    srcfile = 目标文件
    dstDir = 目标文件夹
    将目标文件拷贝至目标文件夹
    '''
    if not os.path.exists(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)  # 创建路径
        shutil.copyfile(srcfile, os.path.join(dstDir, fname))  # 拷贝文件
        print("copy %s -> %s" % (srcfile, os.path.join(dstDir, fname)))


def copy_Dir2Dir(srcDir, dstDir,print_flag=False):
    '''
    copy dir files into target dir and save the original file name
    srcfile = 目标文件夹
    dstDir = 目标文件夹
    将目标文件拷贝至目标文件夹
    '''
    if not os.path.exists(srcDir):
        print("%s not exist!" % (srcDir))
    else:
        file_list = []
        listdir(srcDir, file_list)
        if not os.path.exists(dstDir):
            os.makedirs(dstDir)  # 创建路径
        for file in file_list:
            fpath, fname = os.path.split(file)  # 分离文件名和路径
            shutil.copyfile(file, os.path.join(dstDir, fname))  # 拷贝文件
            if print_flag == True: print("copy %s -> %s" % (file, os.path.join(dstDir, fname)))
