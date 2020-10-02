close all

ffCap = 5;
I = 1;
Tmax = 2.5*I;
t = 0.02;
ffT = 0.1;
loads = [4, 7, 10, 15, 30];
epsilon = 1./(loads-ffCap);
epsilon(epsilon < 0) = 1;

T = 0:t:Tmax;
Ti = (T./I)';
Ti = repmat(Ti, 1, 5);

A = I*(floor(Ti) + (Ti - floor(Ti)).^epsilon) + ffT;

h1 = plot(T, A(:,1), 'LineWidth', 2, 'DisplayName', 'Load = 4', 'LineStyle', '-');
hold on
h2 = plot(T, A(:,2), 'LineWidth', 2, 'DisplayName', 'Load = 7', 'LineStyle', '--');
h3 = plot(T, A(:,3), 'LineWidth', 2, 'DisplayName', 'Load = 10', 'LineStyle', ':');
h5 = plot(T, A(:,5), 'LineWidth', 2, 'DisplayName', 'Load = 30', 'LineStyle', '-.');

set(gca, 'FontSize', 18)
xlabel('Arrival Time at v_{i}', 'FontSize', 18)
ylabel('Arrival Time at v_j', 'FontSize', 18)
xlim([0 Tmax])

legend([h1, h2, h3, h5], 'Location', 'SouthEast')