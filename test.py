import urllib.request

print("正在从互联网拉取 10 张微型 MNIST 真实数据，请稍等...")
# 这是一个极其可靠的开源微型数据集地址（仅仅 10 行文本）
url = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv"
response = urllib.request.urlopen(url)
data_lines = response.readlines()

print("拉取成功！我们来查看第一张真实图片。\n")

# 1. 取出第一行数据，并用逗号把它切分成 785 个独立的值
first_record = data_lines[0].decode('utf-8').strip().split(',')

# 2. 第 1 个值是标准答案（标签）
label = first_record[0]

# 3. 后面的 784 个值是像素特征
pixels = first_record[1:]

print(f"--- 这张图片的标准答案是: 【 {label} 】 ---")
print("--- 计算机眼里的 28x28 像素长这样： ---")

# 4. 把一维的 784 个数字，重新折叠成 28 行，打印给人眼看
for row_index in range(28):
    row_text = ""
    for col_index in range(28):
        # 算出在一维列表里的位置
        pixel_index = row_index * 28 + col_index
        # 将文本转换为整数
        pixel_value = int(pixels[pixel_index])
        
        # 如果像素值大于 100（有较深的笔画），我们就画个方块，否则留空
        if pixel_value > 100:
            row_text += "■ "
        else:
            row_text += "  "
            
    print(row_text)