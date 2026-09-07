import os
import urllib.request

# 确保 data 文件夹存在
os.makedirs("data", exist_ok=True)

url = "https://pjreddie.com/media/files/mnist_test.csv"
filepath = "data/mnist_test.csv"

if os.path.exists(filepath):
    print("✨ 本地已经有这个文件啦，不用重复下载！")
else:
    print("⏳ 正在下载 MNIST 数据集（约 17MB），请稍候...")
    urllib.request.urlretrieve(url, filepath)
    print("🎉 下载完成！")