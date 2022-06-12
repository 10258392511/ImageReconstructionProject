function Xs = apply_shifts(X, shift)
    % shifts X according to shift

    Xs = X;
    for i = 1 : numel(shift)
        Xs = circshift(Xs, shift(i), i);
    end

%     if  isa(X, 'gpuArray') %(strcmp(class(Xs), 'gpuArray')
%         Xs = shift_faster_1(X, shift(1));
%         Xs = shift_faster_2(Xs, shift(2));
%         if numel(shift) >= 3
%             Xs = shift_faster_3(Xs, shift(3));
%         end
%     else
%         Xs = X;
%         for i = 1 : numel(shift)
%             Xs = circshift(Xs, shift(i), i);
%         end
%     end
    
end

function Xs = shift_faster_1(X, s)
    if s == 0
        Xs = X;
        return;
    end
    Xs = zeros(size(X), 'like', X);
    if s > 0
        Xs(s+1:end, :,:,:,:) = X(1:end-s, :,:,:,:);
        Xs(1:s, :,:,:,:) = X(end-s+1:end, :,:,:,:);
    else
        s = -s;
        Xs(1:end-s,     :,:,:,:) = X(s+1:end, :,:,:,:);
        Xs(end-s+1:end, :,:,:,:) = X(1:s, :,:,:,:);
    end
end

function Xs = shift_faster_2(X, s)
    if s == 0
        Xs = X;
        return;
    end
    Xs = zeros(size(X), 'like', X);
    if s > 0
        Xs(:, s+1:end, :,:,:,:) = X(:, 1:end-s, :,:,:,:);
        Xs(:, 1:s, :,:,:,:) = X(:, end-s+1:end, :,:,:,:);
    else
        s = -s;
        Xs(:, 1:end-s,     :,:,:,:) = X(:, s+1:end, :,:,:,:);
        Xs(:, end-s+1:end, :,:,:,:) = X(:, 1:s, :,:,:,:);
    end
end

function Xs = shift_faster_3(X, s)
    if s == 0
        Xs = X;
        return;
    end
    Xs = zeros(size(X), 'like', X);
    if s > 0
        Xs(:, :, s+1:end, :,:,:,:) = X(:, :, 1:end-s, :,:,:,:);
        Xs(:, :, 1:s, :,:,:,:) = X(:, :, end-s+1:end, :,:,:,:);
    else
        s = -s;
        Xs(:, :, 1:end-s,     :,:,:,:) = X(:, :, s+1:end, :,:,:,:);
        Xs(:, :, end-s+1:end, :,:,:,:) = X(:, :, 1:s, :,:,:,:);
    end
end