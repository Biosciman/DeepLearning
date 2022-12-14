% 相比于T6SS_V3,
% 增加convolution1dLayer、reluLayer、maxPooling1dLayer
clear;clc;close all;

data = readmatrix('T6SS_Positive.txt');
data2 = readmatrix('T6SS_Negative.txt');
TotalData_X = [data; data2];
a = linspace(1,1,414);
b = linspace(2,2,1111);
TotalData_Y = [a,b];
TotalData_Y = categorical(TotalData_Y)';

%% 切分训练集和测试集，70%为训练集
dataNumber = size(TotalData_X,1); %%样本个数

randIndex = randperm(dataNumber);  %%打乱数组
new_data_X = TotalData_X(randIndex,:);   
new_data_Y = TotalData_Y(randIndex,:);   

% ceil将 X 的每个元素四舍五入到大于或等于该元素的最接近整数。
testindex = ceil(dataNumber * 0.3); %% 获得分界下标,

XTest = new_data_X(1:testindex,:);
XTrain = new_data_X(testindex+1:end,:);
YTest = new_data_Y(1:testindex,:);
YTrain = new_data_Y(testindex+1:end,:);



XTrain = mat2cell(XTrain,linspace(1,1,size(TotalData_X,1)-testindex));
XTest = mat2cell(XTest,linspace(1,1,testindex));


miniBatchSize = 32;


%% 定义 LSTM 网络架构
% 将输入大小指定为序列大小 1（输入数据的维度）。
% 指定具有100个隐含单元的LSTM层，并输出序列的最后一个元素。
% 最后，通过包含大小为2的全连接层，后跟 softmax 层和分类层，来指定2个类。
% 增加wordEmbeddingLayer，将2000维的one-hot vector降维
% 增加convolution1dLayer
% 增加reluLayer，ReLU 层对输入的每个元素执行阈值运算，其中任何小于零的值都设置为零。
% 增加maxPooling1dLayer

inputSize = 1;
numHiddenUnits = 100;
numClasses = 2;
embeddingDimension = 21;
numWords = 2000;
filterSize = 16;
numFilters= 64;
poolSize = 5;

layers = [ ...
    sequenceInputLayer(inputSize,MinLength=2000)
    wordEmbeddingLayer(embeddingDimension,numWords)
    convolution1dLayer(filterSize,numFilters)
    reluLayer
    maxPooling1dLayer(poolSize, Stride=5)
    lstmLayer(numHiddenUnits,OutputMode="last")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% 指定训练选项
% 指定求解器为 "adam"，梯度阈值为 1，最大轮数为 50。optimizer
% 要填充数据以使长度与最长序列相同，请将序列长度指定为 "longest"。
% 要确保数据保持按序列长度排序的状态，请指定从不打乱数据。

options = trainingOptions("adam", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=1, ...
    MaxEpochs=50, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest", ...
    Shuffle="every-epoch", ...
    Verbose=0, ...
    Plots="training-progress");

%% 训练 LSTM 网络
% 使用trainNetwork以指定的训练选项训练 LSTM 网络。
net = trainNetwork(XTrain,YTrain,layers,options);

% 对测试数据进行分类。
% 要减少分类过程中引入的填充量，请指定使用相同的小批量大小进行训练。
% 要应用与训练数据相同的填充，请将序列长度指定为 "longest"。
YPred = classify(net,XTest, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest");

% 计算预测值的分类准确度。
acc = sum(YPred == YTest)./numel(YTest);
acc;

save(sprintf('T6SS_LSTM_Model_V4_acc_%.2f.mat',acc)); 
