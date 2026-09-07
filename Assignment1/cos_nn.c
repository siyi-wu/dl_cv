#include <stdio.h>   // 提供 printf 函数
#include <stdlib.h>  // 提供 rand 和 srand 函数
#include <math.h>    // 提供 cos、tanh、sqrt 和 acos 函数

#define N 16         // 每个隐藏层包含 16 个神经元

double w1[N], b1[N];     // 输入层到第一隐藏层的权重和偏置
double w2[N][N], b2[N];  // 第一隐藏层到第二隐藏层的权重和偏置
double w3[N], b3;        // 第二隐藏层到输出层的权重和偏置

double R(void) {                         // 生成一个 [-1,1] 范围内的随机数
    return 2.0 * rand() / RAND_MAX - 1;  // 将 rand 的结果缩放到 [-1,1]
}

double net(double x, double *a, double *h) {  // 执行一次神经网络前向传播
    int i, j;                                 // 隐藏层神经元的循环下标

    for (i = 0; i < N; i++)                   // 遍历第一隐藏层
        a[i] = tanh(w1[i] * x + b1[i]);       // 计算第一隐藏层的激活值

    for (i = 0; i < N; i++) {                 // 遍历第二隐藏层
        h[i] = b2[i];                         // 从当前神经元的偏置开始累加
        for (j = 0; j < N; j++)               // 遍历第一隐藏层的所有输出
            h[i] += w2[i][j] * a[j];          // 累加加权输入
        h[i] = tanh(h[i]);                    // 使用 tanh 激活函数
    }

    x = b3;                                   // 用 x 临时保存输出层结果
    for (i = 0; i < N; i++)                   // 遍历第二隐藏层的所有输出
        x += w3[i] * h[i];                    // 计算输出层的加权和
    return x;                                 // 返回线性输出层的预测结果
}

int main(void) {                              // 程序入口
    int i, j, k, q, correct;                  // 定义循环下标和正确预测数量
    double x, t, y, e, loss;                  // 输入、目标、预测、误差和损失
    double a[N], h[N];                        // 两个隐藏层的激活值
    double d1[N], d2[N], d3;                  // 三层参数对应的误差梯度
    double lr = .01;                          // 学习率
    double pi = acos(-1.0);                   // 使用反余弦计算圆周率 π

    srand(1);                                 // 固定随机种子，使结果可重复

    for (i = 0; i < N; i++) {                 // 初始化网络权重
        w1[i] = R();                          // 初始化输入层到第一隐藏层的权重
        w3[i] = R() / sqrt(N);                // 初始化第二隐藏层到输出层的权重
        for (j = 0; j < N; j++)               // 遍历隐藏层之间的所有连接
            w2[i][j] = R() / sqrt(N);         // 初始化两个隐藏层之间的权重
    }

    for (k = 0; k < 200000; k++) {            // 使用 20 万个随机样本进行训练
        x = R();                              // 随机生成归一化输入 x∈[-1,1]
        t = cos(pi * x);                      // 计算目标值 cos(πx)
        y = net(x, a, h);                     // 前向传播并取得预测值
        d3 = y - t;                           // 计算均方误差对线性输出的梯度

        for (i = 0; i < N; i++)               // 计算第二隐藏层的误差梯度
            d2[i] = w3[i] * d3 * (1 - h[i] * h[i]); // 使用 tanh'(z)=1-tanh²(z)

        for (j = 0; j < N; j++) {             // 计算第一隐藏层的误差梯度
            d1[j] = 0;                        // 清空当前神经元的梯度累加值
            for (i = 0; i < N; i++)           // 汇总来自第二隐藏层的梯度
                d1[j] += w2[i][j] * d2[i];    // 将误差沿连接权重反向传播
            d1[j] *= 1 - a[j] * a[j];         // 乘以第一隐藏层 tanh 的导数
        }

        for (i = 0; i < N; i++) {             // 使用梯度下降更新所有参数
            w3[i] -= lr * d3 * h[i];          // 更新第二隐藏层到输出层的权重
            for (j = 0; j < N; j++)           // 遍历两个隐藏层之间的连接
                w2[i][j] -= lr * d2[i] * a[j];// 更新隐藏层之间的权重
            b2[i] -= lr * d2[i];              // 更新第二隐藏层的偏置
            w1[i] -= lr * d1[i] * x;          // 更新输入层到第一隐藏层的权重
            b1[i] -= lr * d1[i];              // 更新第一隐藏层的偏置
        }
        b3 -= lr * d3;                        // 更新输出层偏置

        if ((k + 1) % 10000 == 0) {           // 每训练 10000 次进行一次评估
            loss = 0;                         // 清空累计平方误差
            correct = 0;                      // 清空正确预测数量
            for (q = 0; q < 1001; q++) {      // 在 [-1,1] 上取 1001 个均匀测试点
                x = -1.0 + 2.0 * q / 1000;    // 计算当前归一化测试输入
                t = cos(pi * x);              // 计算当前测试点的真实值
                y = net(x, a, h);             // 计算神经网络预测值
                e = y - t;                    // 计算预测误差
                loss += e * e;                // 累加平方误差
                if (fabs(e) < .05) correct++; // 绝对误差小于 0.05 视为预测正确
            }
            printf("step=%6d  loss=%.8f  accuracy=%6.2f%%\n",
                   k + 1, loss / 1001, 100.0 * correct / 1001);
        }
    }

    for (x = -pi; x <= pi + .001; x += pi / 8) // 在 [-π,π] 上测试网络
        printf("% .3f  cos=% .6f  nn=% .6f\n", // 输出真实值和预测值
               x, cos(x), net(x / pi, a, h));   // 将测试输入归一化后送入网络

    return 0;                                 // 程序正常结束
}
