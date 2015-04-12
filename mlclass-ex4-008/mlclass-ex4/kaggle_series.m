function kaggle_series(m)


	lambda_vector = 0:0.01:.5;
	hl_vector = [20 25 27 30 34 36 40 45 60 70 80 90];

	j_max = 0;

	[trainDS testDS] = take_random_parts("train.csv", m, m);
	[X Y] = add_features(trainDS);
	X = normalize(X);

	[Xtest Ytest] = add_features(testDS);
	Xtest = normalize(Xtest);

	input_layer_size = size(X,2); 
	num_labels = 9; 


	for j=1:length(hl_vector)

		hidden_layer_size = hl_vector(j);

		initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
		initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
		initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
		options = optimset('MaxIter', 400);

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
			end



			fprintf(['Lambda: %f '...
	         '\n'], lambda);
			fprintf(['Hidden layer: %f '...
	         '\n'], hidden_layer_size);
			fprintf(['J_train: %f '...
	         '\n'], self_accuracy);
			fprintf(['J_test: %f '...
	         '\n'], accuracy);
			fprintf('\n');
		end

	end

	fprintf(['\n\n Best result: %f' ...
			'\n Lambda: %f' ...
			'\n Hiddenlayer: %f'] , [j_max lambda_optimal hl_optimal]);
	fprintf('\n\n');





end