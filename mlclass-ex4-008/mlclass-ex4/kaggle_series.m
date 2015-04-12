function kaggle_series(m, hidden_layer_size)

	lambda_vector = 0:0.05:1;

	for i=1:length(lambda_vector)
		lambda = lambda_vector(i);
		[Theta1 Theta2 self_accuracy accuracy] = kaggle_run(m, hidden_layer_size, lambda);
		fprintf(['Lambda: %f '...
         '\n'], lambda);
		fprintf(['J_train: %f '...
         '\n'], self_accuracy);
		fprintf(['J_test: %f '...
         '\n'], accuracy);
		fprintf('\n');
	end





end