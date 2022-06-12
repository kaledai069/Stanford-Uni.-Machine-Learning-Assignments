function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

X1 = X(:, 1);
X2 = X(:, 2);
X3 = X(:, 3);
summed_vector = zeros(3, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    vectorized_value = (X*theta - y)';
    summed_vector(1,1) = vectorized_value * X1;
    summed_vector(2,1) = vectorized_value * X2;
    summed_vector(3,1) = vectorized_value * X3;
    summed_vector = summed_vector * alpha / m;

    for i = 1:3
      theta(i, 1) = theta(i, 1) - summed_vector(i, 1);
    endfor

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
