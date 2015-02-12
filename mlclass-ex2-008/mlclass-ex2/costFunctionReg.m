function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g=@(z) 1/(1+e^(-z));

f1=@(val) log(g(val));

f2=@(val) log(1-g(val));

temp = X*theta;

J = (-y'*arrayfun(f1, temp) -(1-y)'*arrayfun(f2, temp))/m + lambda/(2*m)*(theta'*theta-theta(1)^2);

additional_term = lambda/m * theta;
additional_term(1) = 0;
grad = (X'* (arrayfun(g, temp) - y))/m + additional_term;





% =============================================================

end
