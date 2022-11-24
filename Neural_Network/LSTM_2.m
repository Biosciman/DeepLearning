% 此示例说明如何使用长短期记忆 (LSTM) 网络对序列数据进行分类。
% 要训练深度神经网络以对序列数据进行分类，可以使用 LSTM 网络。
% LSTM 网络允许您将序列数据输入网络，并根据序列数据的各个时间步进行预测。
% 此示例使用 日语元音数据集。
% 此示例训练一个 LSTM 网络，旨在根据表示连续说出的两个日语元音的时间序列数据来识别说话者。
% 训练数据包含九个说话者的时间序列数据。
% 每个序列有 12 个特征，且长度不同。
% 该数据集包含 270 个训练观测值和 370 个测试观测值。

%% 加载序列数据
[XTrain,YTrain] = japaneseVowelsTrainData;
% XTrain 是包含 270 个不同长度的 12 维序列的元胞数组。
% Y是对应于九个说话者的标签 "1"、"2"、...、"9" 的分类向量。
% XTrain 中的条目是具有 12 行（每个特征一行）和不同列数（每个时间步一列）的矩阵。
XTrain(1:5)

% 视化第一个时间序列。每行对应一个特征。
figure
plot(XTrain{1}') % '为复共轭转置
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),Location="northeastoutside")

%% 准备要填充的数据
% 在训练过程中，默认情况下，软件将训练数据拆分成小批量并填充序列，使它们具有相同的长度。

% 获取每个观测值的序列长度。
numObservations = numel(XTrain); % numel获取矩阵的元素个数
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2); % size返回sequence中维度2的长度
end

% 按序列长度对数据进行排序。
[sequenceLengths,idx] = sort(sequenceLengths); % 返回长度和index
XTrain = XTrain(idx);
YTrain = YTrain(idx);

% 在条形图中查看排序的序列长度。
figure
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

% 选择小批量大小 27 以均匀划分训练数据，并减少小批量中的填充量。
miniBatchSize = 27;


%% 定义 LSTM 网络架构
% 将输入大小指定为序列大小 12（输入数据的维度）。
% 指定具有 100 个隐含单元的双向 LSTM 层，并输出序列的最后一个元素。
% 最后，通过包含大小为 9 的全连接层，后跟 softmax 层和分类层，来指定九个类。

inputSize = 12;
numHiddenUnits = 100;
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,OutputMode="last")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

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
    Shuffle="never", ...
    Verbose=0, ...
    Plots="training-progress");

%% 训练 LSTM 网络
% 使用trainNetwork以指定的训练选项训练 LSTM 网络。
net = trainNetwork(XTrain,YTrain,layers,options);

%% 测试 LSTM 网络
% 加载测试集并将序列分类到不同的说话者。
% 加载日语元音测试数据。
% XTest 是包含 370 个不同长度的 12 维序列的元胞数组。
% YTest 是由对应于九个说话者的标签 "1"、"2"、...、"9" 组成的分类向量。
[XTest,YTest] = japaneseVowelsTestData;
XTest(1:3)

% LSTM 网络 net 已使用相似长度的小批量序列进行训练。
% 确保以相同的方式组织测试数据。
% 按序列长度对测试数据进行排序。
numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end

[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);

% 对测试数据进行分类。
% 要减少分类过程中引入的填充量，请指定使用相同的小批量大小进行训练。
% 要应用与训练数据相同的填充，请将序列长度指定为 "longest"。
YPred = classify(net,XTest, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest");

% 计算预测值的分类准确度。
acc = sum(YPred == YTest)./numel(YTest)
acc









