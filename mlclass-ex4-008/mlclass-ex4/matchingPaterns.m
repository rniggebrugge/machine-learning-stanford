function matchingPatterns(Theta1, Theta2)

	show_max = [];
	show_min = [];
	for i = 1:10
		T = Theta2(i,2:end)';
		[val, idx] = min(T);
		show_min = [show_min idx];
		[val, idx] = max(T);
		show_max = [show_max idx];
	end

figure 1;
displayData(Theta1(show_max, 2:end));

figure 2;
displayData(Theta1(show_min, 2:end));


end
