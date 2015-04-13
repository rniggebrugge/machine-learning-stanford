function [trainDS Theta1 Theta2 Xmean Xrange self_accuracy accuracy] = kaggle_run(m, hidden_layer_size, lambda)

	if ~exist('lambda', 'var') || isempty(lambda)
	    lambda = 0.0;
	end

	if ~exist('hidden_layer_size', 'var') || isempty(hidden_layer_size)
	    hidden_layer_size = 25;
	end

	[trainDS testDS] = take_random_parts(m, m);
	[X Y] = add_features(trainDS);
	% [X Xmax]= normalize(X);

	Xmean = mean(X,1);
	Xrange = range(X);
	X = (X.-Xmean)./Xrange;



	[Xtest Ytest] = add_features(testDS);
	Xtest = (Xtest.-Xmean)./Xrange;

	size(Xtest)
	size(X)

	input_layer_size = size(X,2); 
	num_labels = 9; 

	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

	options = optimset('MaxIter', 400);
	
	costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, Y, lambda);

	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


	pred_self = predict(Theta1, Theta2, X);

	self_accuracy = mean(pred_self==Y);

	% fprintf(['\nAccuracy in training set:  %f ' ...
	% 		'\n\n'], self_accuracy);

	pred = predict(Theta1, Theta2, Xtest);

	accuracy = mean(pred==Ytest);

	% fprintf(['\nAccuracy on test set:  %f ' ...
	% 		'\n\n'], accuracy);


end
