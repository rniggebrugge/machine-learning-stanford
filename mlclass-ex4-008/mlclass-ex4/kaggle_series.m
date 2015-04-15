function [theta1_return theta2_return Xmean Xrange] = kaggle_series(iterations, m, lambda_vector, hl_vector, newfs, th1, th2, xm, xr)


	if ~exist('lambda_vector', 'var') || isempty(lambda_vector)
		lambda_vector = [0; 0.01; 0.05; .1; .15; .2 ];
	end

	if ~exist('hl_vector', 'var') || isempty(hl_vector)
		hl_vector = 9:60;
	end


	j_max = -10;

	[trainDS testDS] = take_random_parts(m, 7000);

	if newfs
		[X Y] = add_features_new(trainDS);
		[Xtest Ytest] = add_features_new(testDS);
	else
		[X Y] = add_features(trainDS);
		[Xtest Ytest] = add_features(testDS);
	end
	
	if ~exist('xm', 'var') || isempty(xm)
		Xmean = mean(X,1);
		Xrange = range(X);
	else
		Xmean = xm;
		Xrange = xr;
	end

	% X = (X.-Xmean)./Xrange;
	% Xtest = (Xtest.-Xmean)./Xrange;

	size(X)
	size(Xtest)

	input_layer_size = size(X,2); 
	num_labels = 9; 


	for j=1:length(hl_vector)

		hidden_layer_size = hl_vector(j);


		if ~exist('th1', 'var') || isempty(th1)
			initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
			initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
			fprintf('Random initialization thetas\n\n');
		else
			initial_Theta1 = th1;
			initial_Theta2 = th2;
			fprintf('Continuing with given thetas\n\n');
		end

		initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
		options = optimset('MaxIter', iterations);

		for i=1:length(lambda_vector)
			lambda = lambda_vector(i);
		
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
			pred = predict(Theta1, Theta2, Xtest);
			accuracy = mean(pred==Ytest);

			if accuracy>j_max
				j_max = accuracy;
				lambda_optimal = lambda;
				hl_optimal = hidden_layer_size;
				theta1_return = Theta1;
				theta2_return = Theta2;
			end



			fprintf(['Lambda: %f '...
	         ''], lambda);
			fprintf(['Hidden layer: %f '...
	         ''], hidden_layer_size);
			fprintf(['J_train: %f '...
	         ''], self_accuracy);
			fprintf(['J_test: %f '...
	         ''], accuracy);
			fprintf('\n');
		end

	end

	fprintf(['\n\n Best result: %f' ...
			'\n Lambda: %f' ...
			'\n Hiddenlayer: %f'] , [j_max lambda_optimal hl_optimal]);
	fprintf('\n\n');





end