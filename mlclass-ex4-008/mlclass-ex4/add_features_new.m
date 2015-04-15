function [X Y] = add_features_new(X_in)

	X = X_in;
	Y = X(:,end);
	X(:, end) = [];
	initX = X;

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
		dt = initX.-avg_feature_vector;
		dt = (dt.^2).*(initX~=0);
		dt = exp(-sum(dt,2)/10);
		X = [X dt];
	end



end





