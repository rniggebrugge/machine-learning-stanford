function [nearest_class nearest_distance] = nearest_in_group(x, y, nn)

	m = size(x,1);
	nearest_class = zeros(m,nn+1);
	nearest_distance = zeros(m,nn+1);

	for i=1:m
		vector = x(i,:);
		tmp = bsxfun(@minus, x, vector);
		tmp = tmp.^2;
		tmp = sum(tmp,2);
		[distance ix] = sort(tmp);
		nearest_class(i,1) = y(i);
		for j=1:nn
			nearest_class(i,j+1) = y(ix(j+1));
			nearest_distance(i, j) = distance(j+1);
		end
	end

end
