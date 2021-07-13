from tkinter import *
import time
import pyperclip # 复制功能
from PIL import Image
import matplotlib.pyplot as plt
LOG_LINE_NUM = 0

def callback():
    print("~被调用了~")

moshi = -1
leibie = -1


class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name


    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("代码生成工具")           #窗口名
        #self.init_window_name.geometry('320x160+10+10')                         #290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        #self.init_window_name.geometry('1068x681+50+50')
        self.init_window_name.geometry('1068x681+50+50')
        #self.init_window_name["bg"] = "pink"                                    #窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        #self.init_window_name.attributes("-alpha",0.9)                          #虚化，值越小虚化程度越高
        #标签
        menubar = Menu(self.init_window_name)

        filemenu = Menu(menubar, tearoff=False)
        filemenu.add_command(label="识别", command=self.shibie)
        filemenu.add_command(label="训练", command=self.xvnlian)
        menubar.add_cascade(label="模式", menu=filemenu)

        # 创建另一个下拉菜单“编辑”，然后将它添加到顶级菜单中
        editmenu = Menu(menubar, tearoff=False)
        editmenu.add_command(label="通用", command=self.wuping)
        editmenu.add_command(label="动漫", command=self.dongman)
        editmenu.add_command(label="物品", command=self.wuping)
        editmenu.add_command(label="Logo", command=self.Logo)
        editmenu.add_command(label="车辆", command=self.cheliang)
        menubar.add_cascade(label="类别", menu=editmenu)
        # 显示菜单
        self.init_window_name.config(menu=menubar)
        self.init_data_label = Label(self.init_window_name, text="图片名称")
        self.init_data_label.grid(row=0, column=0)
        self.result_data_label = Label(self.init_window_name, text="输出结果")
        self.result_data_label.grid(row=0, column=12)
        self.log_label = Label(self.init_window_name, text="日志")
        self.log_label.grid(row=12, column=0)
        #文本框
        self.init_data_Text = Text(self.init_window_name, width=67, height=35)  #原始数据录入框
        self.init_data_Text.grid(row=1, column=0, rowspan=10, columnspan=10)
        self.result_data_Text = Text(self.init_window_name, width=70, height=49)  #处理结果展示
        self.result_data_Text.grid(row=1, column=12, rowspan=15, columnspan=10)
        self.log_data_Text = Text(self.init_window_name, width=66, height=9)  # 日志框
        self.log_data_Text.grid(row=13, column=0, columnspan=10)
        #按钮
        self.str_trans_to_md5_button = Button(self.init_window_name, text="生成代码", bg="lightblue", width=10,command=self.str_trans_to_md5)  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=1, column=11)
        self.str_trans_to_md5_button = Button(self.init_window_name, text="复制代码", bg="lightblue", width=10,command=self.Copy_password)  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=2, column=11)
        self.str_trans_to_md5_button = Button(self.init_window_name, text="查看结果", bg="lightblue", width=10,command=self.showpicture)  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=3, column=11)
    def Copy_password(self):
        pyperclip.copy(myMd5_Digest)

    def showpicture(self):
        src = self.init_data_Text.get(1.0, END)
        src = src.replace('\n', '').replace('\r', '')
        path = "C:/Users/asus/project/PaddleClas/PaddleClas/deploy/output/"
        picturepath = path +src
        img = Image.open(picturepath)
        plt.figure(src)
        plt.imshow(img)
        plt.show()
    #功能函数
    def str_trans_to_md5(self):
        src = self.init_data_Text.get(1.0,END)
        # print("src =",src)
        global myMd5_Digest
        two = '-o Global.infer_imgs="./recognition_demo_data_v1.0/test/' + src + '"'
        two = two.replace('\n', '').replace('\r', '')
        if (moshi == 0 and leibie == 0):
            one = 'python python/predict_system.py -c configs/inference_cartoon.yaml '
            three = ' -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_cartoon/index_update"'
            myMd5_Digest = one + two +three
        if (moshi == 0 and leibie == 1):
            one = 'python python/predict_system.py -c configs/inference_logo.yaml '
            three = ' -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_logo/index_update"'
            myMd5_Digest = one + two +three
        if (moshi == 0 and leibie == 2):
            one = 'python python/predict_system.py -c configs/inference_vehicle.yaml '
            three = ' -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_vehicle/index_update"'
            myMd5_Digest = one + two +three
        if (moshi == 0 and leibie == 3):
            one = 'python python/predict_system.py -c configs/inference_product.yaml '
            three = ' -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_product/index_update"'
            myMd5_Digest = one + two +three
        if (moshi == 1 and leibie == 0):
            myMd5_Digest = 'python python/build_gallery.py -c configs/build_cartoon.yaml -o IndexProcess.data_file="./recognition_demo_data_v1.0/gallery_cartoon/data_file.txt" -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_cartoon/index_update"'

        if (moshi == 1 and leibie == 1):
            myMd5_Digest = 'python python/build_gallery.py -c configs/build_logo.yaml -o IndexProcess.data_file="./recognition_demo_data_v1.0/gallery_logo/data_file.txt" -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_logo/index_update"'
        if (moshi == 1 and leibie == 2):
            myMd5_Digest = 'python python/build_gallery.py -c configs/build_vehicle.yaml -o IndexProcess.data_file="./recognition_demo_data_v1.0/gallery_vehicle/data_file.txt" -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_vehicle/index_update"'
        if (moshi == 1 and leibie == 3):
            myMd5_Digest = 'python python/build_gallery.py -c configs/build_product.yaml -o IndexProcess.data_file="./recognition_demo_data_v1.0/gallery_product/data_file.txt" -o IndexProcess.index_path="./recognition_demo_data_v1.0/gallery_product/index_update"'


        if src:
            try:
                # myMd5 = hashlib.md5()
                # myMd5.update(src)
                # myMd5_Digest = myMd5.hexdigest()
                #print(myMd5_Digest)
                #输出到界面
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,myMd5_Digest)
                self.write_log_to_Text("INFO:success")
            except:
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,"失败")
        else:
            self.write_log_to_Text("ERROR: failed")


    #获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        return current_time

    def shibie(self):
        global moshi
        moshi = 0
    def xvnlian(self):
        global moshi
        moshi = 1

    def dongman(self):
        global leibie
        leibie = 0

    def Logo(self):
        global leibie
        leibie = 1

    def cheliang(self):
        global leibie
        leibie = 2

    def wuping(self):
        global leibie
        leibie = 3

    #日志动态打印
    def write_log_to_Text(self,logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) +" " + str(logmsg) + "\n"      #换行
        if LOG_LINE_NUM <= 7:
            self.log_data_Text.insert(END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0,2.0)
            self.log_data_Text.insert(END, logmsg_in)


def gui_start():
    init_window = Tk()              #实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示


gui_start()
