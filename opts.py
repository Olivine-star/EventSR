import argparse


"""
parser = argparse.ArgumentParser()
就是创建了一个 命令行参数解析器对象，叫做 parser.可以理解为它是一个“参数收集器”
.add_argument(...) 是在注册你程序“支持接收哪些参数”，以及这些参数的数据类型、默认值、说明等。
例如：parser.add_argument('--bs', type=int, default=32)
接受参数名 --bs
其值必须是整数
如果用户不传，就用默认值 32

"""
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--savepath', type=str, default='/repository/admin/DVS/Classification/N-MNIST/ckpt_convSNN/full')
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--showFreq', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--cuda', type=str, default='0,1')
parser.add_argument('--add', type=str, default=None)
parser.add_argument('--j', type=int, default=16)
