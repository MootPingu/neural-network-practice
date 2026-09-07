import matplotlib.pyplot as plt

# ================= 配置区 =================
# 你可以在这里随意修改你想看哪个文件，以及文件的第几行
FILE_PATH = "data/mnist_test_10.csv"
LINE_NUMBER = 6  # 程序员的计数从 0 开始哦！0 代表第 1 行，1 代表第 2 行...
# ==========================================

print(f"正在打开文件: {FILE_PATH}")
print(f"准备抽取第 {LINE_NUMBER + 1} 行数据...")

# 1. 打开你本地的数据文件，并读取所有的行
with open(FILE_PATH, 'r') as f:
    data_lines = f.readlines()

# 2. 拿到你指定的那一行，并用逗号切分
record = data_lines[LINE_NUMBER].split(',')

# 3. 第一个是标签（答案），后面是 784 个像素点
label = record[0]
# 把后面的文本数字全部转换成浮点数
pixels = [float(x) for x in record[1:]]

# 4. 把一维的长长队伍，重新折叠成 28x28 的二维网格（供画图库使用）
image_2d = []
for i in range(28):
    # 每次切片拿出 28 个数字，作为新的一行
    row = pixels[i * 28 : (i + 1) * 28]
    image_2d.append(row)

# 5. 召唤 matplotlib 弹窗作画！
print(f"抽取成功！这张图的标准答案是：{label}，即将弹出窗口展示...")

# 设置图像的标题
plt.title(f"Label: {label}", fontsize=20)
# cmap='Greys' 表示用灰度模式画图（数值越大越黑）
plt.imshow(image_2d, cmap='Greys')
# 显示弹窗（代码会在这里暂停，直到你手动关闭那个弹窗）
plt.show()

print("弹窗已关闭！")