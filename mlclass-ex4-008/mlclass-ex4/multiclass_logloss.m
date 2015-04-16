function logloss = multiclass_logloss(y, Theta1, Theta2, X)

	m = size(X, 1);
	num_labels = size(Theta2, 1);

	h1 = sigmoid([ones(m, 1) X] * Theta1');
	p = sigmoid([ones(m, 1) h1] * Theta2');

	p = p./sum(p,2);
	p(p<1e-15) = 1e-15;
	p(p>(1-e-15)) = 1 - 1e-15;

	y = eye(n)(y,:);

	temp = log(p).*y;

	logloss = -1/m*sum(temp(:));

end




