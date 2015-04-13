function [prediction cl] = classify_by_distance(train, Ytrain, test)

	m = size(test,1);
	prediction = zeros(m,9);

	Xmax = max(train);
	train = train./Xmax;
	test = test./Xmax;

	for i=1:m
		fprintf('%f', i);
		vector = test(i,:);
		dt = train.-vector;
		dt = dt.^2;
		dt = sum(dt,2);
		[dummy idx] = sort(dt);
		for j=1:100
			prediction(i,Ytrain(idx(j))) += 1; %dummy(1)/dummy(j);
		end
	end

	[dummy cl] = max(prediction, [], 2);

end