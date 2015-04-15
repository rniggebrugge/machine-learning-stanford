function [X Y] = add_features_new(X_in)

	X = X_in;
	Y = X(:,end);
	X(:, end) = [];

	avg_features = csvread("non-zero-average-table.csv");
	avg_features(:,1)=[];
	avg_features(1,:)=[];
	avg_features = avg_features';

	for idx=1:9
		avg_feature_vector = avg_features(idx,:);
		dt = abs(X(:,1:93).-avg_feature_vector).*(X(:,1:93)~=0);
		X = [X dt];
	end

end





