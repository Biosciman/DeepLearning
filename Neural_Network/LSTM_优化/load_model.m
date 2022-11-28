load("T6SS_LSTM_Model_V11_acc_0.8231.mat")

recall = numel(intersect(find(YTest=='1'),find(YPred=='1')))/numel(find(YTest=='1'));
recall