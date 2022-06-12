function shift = generate_random_shift(sz)
    shift = zeros(numel(sz), 1);
    for i = 1 : numel(sz)
        shift(i) = randi(sz(i));
    end
end