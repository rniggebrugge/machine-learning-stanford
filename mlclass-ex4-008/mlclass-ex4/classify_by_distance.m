function [prediction cl] = classify_by_distance(train, Ytrain, test, Ytest)

	mtrain = size(train,1);
	m = size(test,1);
	prediction = zeros(m,9);

	Xmax = max(train);
	train = train./Xmax;
	test = test./Xmax;

	non_zero = sum(train~=0,2)/93;

	for i=1:m
		fprintf('%f', i);
		vector = test(i,:);

		% distance calculations
		dt = train.-vector;
		dt = dt.^2;
		dt = sum(dt,2);
		% max_distance = min(dt)*1.6;
		[distances idx] = sort(dt);
		j=1;
		while(j<100)
			prediction(i,Ytrain(idx(j))) += (1/j * non_zero(idx(j))) ; %+ distances(1)/distances(j) ; 
			% prediction(i,Ytrain(idx(mtrain-j+1))) -= 1/j;
			j++;
		end
	end

%	prediction = prediction./count;

	[dummy cl] = max(prediction, [], 2);

	accuracy = mean(cl==Ytest);

	fprintf(['This got an accuracy of %f' ...
		'\n'], accuracy );

end