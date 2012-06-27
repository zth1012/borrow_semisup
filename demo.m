%% Data setup
object = 'car';
dataset = 'PASCAL2007';
featureFolder = './';   % folder for the feature file
datasetMetaFolder = './'; % folder for the dataset meta-data file
addpath('./liblinear-1.8/matlab/');
datasetMetaFile = fullfile(datasetMetaFolder, [dataset, '.mat']);
featureFile = fullfile(featureFolder, [dataset, '_llc_2f_' object '.mat']);

load(featureFile);
train_features = [train_features, ones(size(train_features,1),1)];
test_features = [test_features, ones(size(test_features,1),1)];

train_labels = train_labels';

% Split the training data into two sets: labeled and unlabeled
pos_idx = find(train_labels == 1);
neg_idx = find(train_labels == -1);
num_label_pos = 100;
num_label_neg = 100;

pm = randperm(numel(pos_idx));
l_pos_idx = pos_idx(pm(1:num_label_pos));
u_pos_idx = pos_idx(pm(num_label_pos+1:end));
pm = randperm(numel(neg_idx));
l_neg_idx = neg_idx(pm(1:num_label_neg));
u_neg_idx = neg_idx(pm(num_label_neg+1:end));

tr_label_copy = train_labels;

% Unlabeled samples are assigned 0 as training labels
train_labels(u_pos_idx) = 0;
train_labels(u_neg_idx) = 0;

% Get ground-truth labels for unlabeled set (for evaluation)
u_idx = find(train_labels == 0);
gt_u_label = tr_label_copy(u_idx);  

fprintf('#l_pos = %d, #l_neg = %d, #u_pos = %d, #u_neg = %d\n', numel(l_pos_idx), numel(l_neg_idx), numel(u_pos_idx), numel(u_neg_idx));

%% Parameter setup
C_l = [1000];
C_up=[1000];
C_un=[500];

cnt = 0;
for i = 1 : numel(C_l)
    for j = 1 : numel(C_up)
        for k = 1 :numel(C_un)
            cnt = cnt + 1;
            opts{cnt}.C_l = C_l(i);
            opts{cnt}.C_up = C_up(j);
            opts{cnt}.C_un = C_un(k);
            opts{cnt}.global_max_iter = 10;
            opts{cnt}.w_max_iter = 100;
            opts{cnt}.w_learn_rate = 0.0000001;
            opts{cnt}.dataset_meta_file = datasetMetaFile;
        end
    end
end
model = cell(cnt,1);
w = cell(cnt,1);
beta = cell(cnt,1);
base_ap = zeros(cnt,1);
semi_dec_val = cell(cnt,1);
base_dec_val = cell(cnt,1);
semi_ap = zeros(cnt,1);
for i = 1 : cnt
    [w{i}, beta{i}] = semi_train_sgd(train_features, train_labels, opts{i});
    semi_dec_val{i} = test_features * w{i};
    semi_ap(i) = myAP(semi_dec_val{i}, test_labels', 1);
    model{i} = train(double(train_labels([l_pos_idx; l_neg_idx])), sparse(train_features([l_pos_idx; l_neg_idx],:)), ['-s 0 -c ' num2str(opts{i}.C_l) ' -q 1']);
    [predict_label, accuracy, base_dec_val{i}] = predict(double(test_labels'), sparse(test_features), model{i});
    base_ap(i) = myAP(base_dec_val{i}, test_labels',1);
end