function [x1 x2 x3 x4 x5 y1 y2 y3 y4 y5] = take5parts(sizes)

	filename = "train.csv";

	X = csvread(filename);
	X(1,:) = []; %% remove first line (labels)
	X(:,1) = []; %% remove first column (ids)

	v = randperm(size(X,1));
	X = X(v,:);

	first = 1;
	last = first + sizes(1)-1;
	x1=X(first:last,:);

	first=last+1;
	last = first + sizes(2)-1;
	x2=X(first:last,:);

	first=last+1;
	last = first + sizes(3)-1;
	x3=X(first:last,:);

	first=last+1;
	last = first + sizes(4)-1;
	x4=X(first:last,:);

	first=last+1;
	last = first + sizes(5)-1;
	x5=X(first:last,:);

	y1=x1(:,end);
	y2=x2(:,end);
	y3=x3(:,end);
	y4=x4(:,end);
	y5=x5(:,end);

	x1(:,end)=[];
	x2(:,end)=[];
	x3(:,end)=[];
	x4(:,end)=[];
	x5(:,end)=[];

end