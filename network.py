import numpy as np

class NeuralNetwork:
    def __init__(self, 
            input_nodes, hidden_nodes, output_nodes,
            learning_rate):
        
        self.inodes = input_nodes   # 输入层：28*28所有像素
        self.hnodes = hidden_nodes  # 隐藏层：
        self.onodes = output_nodes  # 输出层：预测数字0-9
        self.lr = learning_rate     # 学习率

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

    def train(self, inputs_list, targets_list):
        # 前向传播
        inputs = np.array(inputs_list, dtype=float, ndmin=2).T
        targets = np.array(targets_list, dtype=float, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)
        
        # 更新第二张权重表格 (who)
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)), 
            hidden_outputs.T
        )
        
        # 更新第一张权重表格 (wih)
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
            inputs.T
        )

    def query(self, inputs_list):
        final_outputs = self.forward(inputs_list)
        return np.argmax(final_outputs)


if __name__ == "__main__":
    # 初始化网络：784输入，200隐藏神经元，10个输出分类
    net = NeuralNetwork(784, 200, 10, 0.1)
    
    # 读取真实的 MNIST 测试数据
    with open("data/mnist_test.csv", "r") as f:
        data_list = f.readlines()

    batch_size = 100
    print(f"神经网络开始刷前 {batch_size} 个样本...")

    # 1. 训练部分（只训100个）
    for i in range(batch_size):
        # 准备数据
        all_values = data_list[i].split(',')
        real_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        targets = np.zeros(10) + 0.01
        targets[real_label] = 0.99
        
        # 训练一次
        net.train(inputs, targets)

    # 2. 考试部分（用接下来的 10 个从未见过的样本，避开前面的 100 个）
    test_start = 100
    test_size = 10
    print(f"开始测试第 {test_start + 1} 到第 {test_start + test_size} 个样本...")

    scorecard = []
    for i in range(test_start, test_start + test_size):
        all_values = data_list[i].split(',')
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        
        # 考试盲猜
        ans = net.query(inputs)
        
        if ans == correct_label:
            scorecard.append(1)
            print(f"样本{i - test_start + 1}：输出 {ans}，答案 {correct_label} -> 🟢")
        else:
            scorecard.append(0)
            print(f"样本{i - test_start + 1}：输出 {ans}，答案 {correct_label} -> 🔴")
            
    # 计算最终准确率
    scorecard_array = np.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    print(f"这 {test_size} 个全新样本的准确率: {performance * 100:.2f}%")