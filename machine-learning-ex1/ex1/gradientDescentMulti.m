function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it c+an be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
  %  temp = zeros(3,1);
    hypothesis = X*theta;
    error = hypothesis-y;
  %  for i = 1:3
   %   temp(i,1)=theta(i,1)-(alpha/m)*sum(error.*X(:,i));
   % end
   % theta(1,1)=temp(1,1);
    % theta(2,1)=temp(2,1);
    % theta(3,1)=temp(3,1);

      temp = theta-(alpha/m)*X'*(error);
      theta = temp;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
