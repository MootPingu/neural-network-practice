import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.inodes = input_nodes   # 输入层：28*28所有像素
        self.hnodes = hidden_nodes  # 隐藏层：
        self.onodes = output_nodes  # 输出层：预测数字0-9

        # 生成输入层 -> 隐藏层的权重表格 (二维矩阵)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        
        # 生成隐藏层 -> 输出层的权重表格 (二维矩阵)
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid 激活函数

    def forward(self, inputs_list):
        # 将输入列表转升为二维数组，并转置
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)  # 计算隐藏层的输入
        hidden_outputs = self.activation_function(hidden_inputs)  # 计算隐藏层的输出

        final_inputs = np.dot(self.who, hidden_outputs)  # 计算输出层的输入
        final_outputs = self.activation_function(final_inputs)  # 计算输出层的输出

        return final_outputs



if __name__ == "__main__":
    # 初始化网络：784输入，200隐藏神经元，10个输出分类
    net = NeuralNetwork(784, 200, 10)
    
    # 读取真实的 MNIST 测试数据
    with open("data/mnist_test_10.csv", "r") as f:
        data_list = f.readlines()
        
    # 取第一行数据
    all_values = data_list[0].split(',')
    real_label = all_values[0]
    
    # 归一化处理像素值 (0-255 -> 0.01-0.99)
    inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
    
    # 运行前向传播
    result = net.forward(inputs)
    
    print(f"👉 这张图片的真实标签是: {real_label}")
    print("--- 前向传播运行成功 ---")
    print("输出矩阵形状:", result.shape)
    print("神经网络随机猜的 10 个分类得分（当前未训练，所以是乱猜的）:\n", result)