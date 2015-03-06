function [size_vec, error_train, error_val] = ...
    bonus_question(X, y, Xval, yval, lambda)

size_vec = 1:12;
iterations = 50;

n_train = size(X,1);
n_val = size(Xval,1);

% You need to return these variables correctly.
error_train = zeros(length(size_vec), 1);
error_val = zeros(length(size_vec), 1);

for i = 1:length(size_vec)
	sum_error_train = 0;
	sum_error_val = 0;
	size = size_vec(i);
	for j = 1:iterations
		perm = randperm(n_train);
		Xtrain = X(perm(1:size),:);
		ytrain = y(perm(1:size));
		perm = randperm(n_val);
		Xval_sub = Xval(perm(1:size),:);
		yval_sub = yval(perm(1:size));
		theta = trainLinearReg(Xtrain, ytrain, lambda);
		sum_error_train = sum_error_train + linearRegCostFunction(Xtrain, ytrain, theta, 0);
		sum_error_val = sum_error_val + linearRegCostFunction(Xval_sub, yval_sub, theta, 0);
	end
	error_train(i) = sum_error_train/iterations;
	error_val(i) = sum_error_val/iterations;
end



plot(size_vec, error_train, size_vec, error_val);




% =========================================================================

end
