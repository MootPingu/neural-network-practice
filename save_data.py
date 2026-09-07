import urllib.request

print("正在去网上拿数据...")
url = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv"
response = urllib.request.urlopen(url)

# 直接把网页上所有的内容一口气读下来（字节形式）
raw_data = response.read()

print("拿到了！正在存入本地 data 文件夹...")

# 以“写入（w）”模式打开（或者新建）一个文件，把数据塞进去
# 注意：前提是你已经在外面手动建好了 data 文件夹！
with open("data/mnist_test_10.csv", "wb") as f:
    f.write(raw_data)

print("搞定！现在就算断网，我们也有数据可以训练了！")