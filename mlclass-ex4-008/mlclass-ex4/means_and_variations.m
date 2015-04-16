function [means vars] = means_and_variations(X, Y)

	means = [];
	vars = [];
	for c=1:9
		subset = X(Y==c,:);
		ms = bsxfun(@rdivide, sum(subset), sum(subset~=0));
		means = [means; ms ];
		devs = bsxfun(@minus, subset, ms);
		devs = devs.^2;
		devs = devs.*(subset~=0);
		devs = sum(devs);
		vals = sum(subset~=0);
		devs = devs./vals;
		vars = [vars; devs];
	end

end

