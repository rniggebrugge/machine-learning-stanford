function [Xtrain Xtest] = take_random_parts(mtrain, mtest)

	filename = "train.csv";

	X = csvread(filename);
	X(1,:) = []; %% remove first line (labels)
	X(:,1) = []; %% remove first column (ids)
	m = size(X,1);

	if mtrain >= m
		mtrain = m;
		mtest = 0;
	end

	if (mtrain+mtest)>m
		mtest = m - mtrain;
	end

	if mtrain == 0
		Xtrain = X;
		Xtest = [];
	else
		v = randperm(size(X,1));
		X = X(v,:);
		Xtrain = X(1:mtrain,:);
		Xtest = X((mtrain+1):(mtrain+mtest),:);
	end
end