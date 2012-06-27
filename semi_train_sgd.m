function [w, beta] = semi_train_sgd(train_feat, train_label, opt)

C_l = opt.C_l;
C_up = opt.C_up;
C_un = opt.C_un;
global_max_iter = opt.global_max_iter;
w_max_iter = opt.w_max_iter;
w_learn_rate = opt.w_learn_rate;
sgd_batch = 100;

l_idx = find(train_label);
lp_idx = find(train_label == 1);
ln_idx = find(train_label == -1);
u_idx = find(train_label == 0);

% Initialize 'w' using labeled data only (with liblinear)
% (need to make sure that the first example is positive)
w = init_w(train_feat([lp_idx;ln_idx],:), train_label([lp_idx;ln_idx]), C_l);

% Initialized to borrow none for positives
beta = zeros(numel(u_idx),1);

% Global objective function at initialization
prev_obj = compute_global_obj(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un);

iter = 1;
while 1
    % Phase 1: update 'beta' by fixing 'w' as constant
    beta = update_beta(w, beta, train_feat(u_idx,:), C_up, C_un);
    
    % Phase 2: update 'w' by fixing 'beta' as constant
    w = update_w(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un, w_learn_rate, w_max_iter, sgd_batch);
    
    % Global objective function after each iteration
    curr_obj = compute_global_obj(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un);
    
    obj_dec = prev_obj - curr_obj;
    fprintf('Big loop: iter = %d, curr_obj = %f, obj_dec = %08d, norm(w) = %f, norm(beta,1) = %f\n', iter, curr_obj, obj_dec, norm(w), norm(beta,1));
    if obj_dec == 0
        fprintf('Converged!\n');
        break;
    end
    iter = iter + 1;
    if iter > global_max_iter
        break;
    end
    
    prev_obj = curr_obj;
end

function best_w = update_w(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un, learn_rate, max_iter, sgd_batch)
iter = 1;
prev_obj = compute_global_obj(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un);
best_obj = prev_obj;
best_w = w;
while 1
    grad = w;
    % Permutation for SGD
    pm = randperm(n);
    for i = 1 : sgd_batch
        idx = pm(i);
        xi = train_feat(idx,:);
        yi = train_label(idx);
        if yi ~= 0
            if 1 - yi * xi * w > 0
                grad = grad - C_l * yi * xi';
            end
        else
            % Find the corresponding index in 'beta'
            bidx = find(u_idx == idx);
            if 1 - xi * w > 0
                grad = grad - C_up * beta(bidx) * xi';
            end
            if 1 + xi * w > 0
                grad = grad + C_un * (1 - beta(bidx)) * xi';
            end
        end
    end
    
    w = w - learn_rate * grad;
    
    curr_obj = compute_global_obj(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un);
    obj_dec = prev_obj - curr_obj;
    
    % Update only if the objective function is less than the current best
    if curr_obj < best_obj
        best_obj = curr_obj;
        best_w = w;
    end
    
    fprintf('update_w: iter = %d, curr_obj = %f, obj_dec = %08f, norm(w) = %f\n', iter, curr_obj, obj_dec, norm(w));
    prev_obj = curr_obj;
    
    iter = iter + 1;    
    if iter > max_iter
        break;
    end
end


function beta = update_beta(w, beta, u_feat, C_up, C_un)

for i = 1 : size(u_feat, 1)
    if C_up * max([0, 1 - u_feat(i,:) * w]) - C_un * max([0, 1 + u_feat(i,:) * w]) >= 0
        beta(i) = 0;
    else 
        beta(i) = 1;
    end
end

function best_w = init_w(l_feat, l_label, C_l)

model = train(double(l_label), sparse(l_feat), ['-s 0 -c ' num2str(C_l) ' -q 1']);
best_w = model.w';

function obj = compute_global_obj(w, beta, train_feat, train_label, l_idx, u_idx, C_l, C_up, C_un)

obj = 0.5 * norm(w,2);

train_score = train_feat * w;

temp_loss = max([zeros(length(l_idx),1), 1 - train_score(l_idx).*train_label(l_idx)], [], 2);
obj = obj + C_l * sum(temp_loss);

temp_loss = beta .* max([zeros(length(u_idx),1), 1 - train_score(u_idx)], [], 2);
obj = obj + C_up * sum(temp_loss);

temp_loss = (1 - beta) .* max([zeros(length(u_idx),1), 1 + train_score(u_idx)], [], 2);
obj = obj + C_un * sum(temp_loss);
