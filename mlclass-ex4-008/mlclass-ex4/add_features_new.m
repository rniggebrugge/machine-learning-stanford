function [X Y] = add_features_new(X_in)

	X = X_in;
	Y = X(:,end);
	X(:, end) = [];
	initX = X;
	X = log(X+1);

	v = sum(X>0);
	[i ix] = sort(v, "descend");

	X2 = X.^2;

	avg_features = csvread("non-zero-average-table.csv");
	avg_features(:,1)=[];
	avg_features(1,:)=[];
	avg_features = avg_features';

	% The method below takes the absolute distance for each feature with the
	% expected (average) value for that feature for each class.

	% for idx=1:9
	% 	avg_feature_vector = avg_features(idx,:);
	% 	dt = abs(initX.-avg_feature_vector).*(initX~=0);
	% 	X = [X dt];
	% end

	% The method below calculates a distance between the vector formed by all
	% expected (average) feature values. Taking into account only those features for
	% which the record has non-zero value features.

	for idx=1:9
		avg_feature_vector = avg_features(idx,:);
		dt = bsxfun(@minus,initX, avg_feature_vector);
		dt = (dt.^2).*(initX~=0);
		dt = exp(-sum(dt,2)/10);
		X = [X dt];
	end

	for i=1:9
		for j=(i+1):10
			X = [X X(:,ix(i)).*X(:,ix(j))];
			X = [X X(:,ix(i))./(1+X(:,ix(j)))];
		end
	end

	%% power3-products

	for i=1:10
		for j=i:10
			for k=j:10	
				X = [X X(:,ix(i)).*X(:,ix(j)).*X(:,ix(k))];
			end
		end
	end

	%% power4-products

	for i=1:4
		for j=i:4
			for k=j:4
				for l=k:4	
					X = [X X(:,ix(i)).*X(:,ix(j)).*X(:,ix(k)).*X(:,ix(l))];
			end
		end
	end


	X = [X X2];
end





