function [Xreturn Y] = add_features(X)

	Y = X(:,end);	%% Y := class
	X(:,end) = []; 	%% remove last colum (class)
	X = log(X+1); 	%% see how this works

	[m n] = size(X);

	%% squared values for all features
	X2 = X.^2;

	%% get the number of non-zero values, and sort to be able
	%% to create new features based on the most non-zero features.
	%% my idea is that these probably have the highest predictive
	%% power

	v = sum(X>0);
	[i ix] = sort(v, "descend");

	%% power2-products (except quadratic)

	% Xpower2 = zeros(m, 90);
	% for row=1:m
	% 	column = 1;
	% 	for i=1:9
	% 		for j=(i+1):10
	% 			product = X(row,ix(i))*X(row,ix(j));
	% 			Xpower2(row, column) = product;
	% 			column = column+1;
	% 			quotient = 0;
	% 			if product>0
	% 				quotient = X(row,ix(i))/X(row,ix(j));
	% 			end
	% 			Xpower2(row, column) = quotient;
	% 			column = column+1;
	% 		end
	% 	end
	% end

	% for i=1:9
	% 	for j=(i+1):10
	% 		X = [X X(:,ix(i)).*X(:,ix(j))];
	% 		X = [X X(:,ix(i))./(1+X(:,ix(j)))];
	% 	end
	% end

	%% power3-products

	% Xpower3 = zeros(m, 64);
	% for row=1:m
	% 	column = 1;
	% 	for i=1:4
	% 		for j=1:4
	% 			for k=1:4
	% 				Xpower3(row, column) = X(row,ix(i))*X(row,ix(j))*X(row,ix(k));
	% 				column = column+1;
	% 			end
	% 		end
	% 	end
	% end

	% for i=1:10
	% 	for j=i:10
	% 		for k=j:10	
	% 			X = [X X(:,ix(i)).*X(:,ix(j)).*X(:,ix(k))];
	% 		end
	% 	end
	% end

	% for i=1:4
	% 	for j=i:4
	% 		for k=j:4
	% 			for l=k:4	
	% 				X = [X X(:,ix(i)).*X(:,ix(j)).*X(:,ix(k)).*X(:,ix(l))];
	% 		end
	% 	end
	% end

	Xreturn = X; %[X X2];

end