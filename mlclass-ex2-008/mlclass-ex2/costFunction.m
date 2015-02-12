function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
g=@(z) 1/(1+e^(-z));

f1=@(val) log(g(val));

f2=@(val) log(1-g(val));

temp = X*theta;

J = (-y'*arrayfun(f1, temp) -(1-y)'*arrayfun(f2, temp))/m;

grad = (X'* (arrayfun(g, temp) - y))/m;

% =============================================================

end
