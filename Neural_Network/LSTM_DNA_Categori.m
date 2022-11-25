%% 尝试用文本分词对dna数据训练。
%% 利用LSTM对不同的DNA序列进行分类
%% 清理环境空间
clc
clear 
close all
%% 读取DNA数据与预处理
% a=importdata('sequence.fasta');
% c=a.Sequence;
% % c={a.Sequence};

%% 随机字符串生成
symbols = ['A' 'T' 'C' 'G'];
% MAX_ST_LENGTH = 50;
stLength = 10000;
nums = randi(numel(symbols),[1 stLength]);
st = symbols (nums);

n_num=50;
for i=1:n_num
    c_1(i,:)=st(i*20:(i+1)*20); %序列的长度
    cc(i,:)= strtrim(regexprep(c_1(i,:), '.{3}', '$0 ')); %每三个字符加一个空格
    hh(i,1)= ceil(rand(1)*4+0);%随机录入标签
end
cc_t=table(cc(1:n_num-5,:),hh(1:n_num-5,:),'VariableNames',{'Description','Category'});
%% 数据处理
data=cc_t;
data.Category = categorical(cc_t.Category);

% figure
% histogram(data.Category);
% xlabel("Class")
% ylabel("Frequency")
% title("Class Distribution")
cvp = cvpartition(data.Category,'Holdout',0.2); %选择20%作为测试集合
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);
textDataTrain = dataTrain.Description;
textDataValidation = dataValidation.Description;
YTrain = dataTrain.Category;
YValidation = dataValidation.Category;
% figure
% wordcloud(textDataTrain);
% title("Training Data")
%% 对序列进行分词 此处需要更改，
documentsTrain = cellstr(textDataTrain); %调整数据类型
documentsTrain=tokenizedDocument(documentsTrain);% 进行分词统计 官网使用的是其他函数
documentsValidation = cellstr(textDataValidation);
documentsValidation=tokenizedDocument(documentsValidation);
% documentsTrain(1:100,:) 数据展示
enc = wordEncoding(documentsTrain);% 自带分词库 需要根据需求调整
documentLengths = doclength(documentsTrain);
% figure
% histogram(documentLengths)
% title("Document Lengths")
% xlabel("Length")
% ylabel("Number of Documents")
% %% 设置训练的参数
sequenceLength = 3; %词的截断长度
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XTrain(1:5)
XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);
inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 80;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));
% 层结构
layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
%超参
options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);
%训练
net = trainNetwork(XTrain,YTrain,layers,options);
%预测
reportsNew =cc(n_num-5:end,:);
documentsNew = cellstr(reportsNew);
documentsNew=tokenizedDocument(documentsNew);
XNew = doc2sequence(enc,documentsNew,'Length',sequenceLength);
%结果输出
labelsNew = classify(net,XNew)
