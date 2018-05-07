    N = 100;
    landmarks = zeros(2, N);
    landmarks(:, 5) = [1 0]';
    lids = [];
    for i = 1:N
        if (landmarks(1,i) ~= 0 || landmarks(2,i) ~= 0)
            lids = [lids, i];
        end
    end