function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_range = [0.01; 0.03 ; 0.1; 0.3; 1; 3; 10; 30];

sigma_range = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error = zeros(64,3);
count=1; 
for i = 1:length(C_range)
  for j = 0:length(sigma_range)-1
    model = svmTrain(X,y,C_range(i),@(x1,x2)gaussianKernel(x1,x2,sigma_range(j+1)));
    predictions = svmPredict(model, Xval);
    error(count,1) = C_range(i);
    error(count,2) = sigma_range(j+1);
    error(count,3) = mean(double(predictions~=yval));
    count = count+1;
  end
end

[s,i] = min(error);
k = i(1,3);
C = error(k,1);
sigma = error(k,2);



% =========================================================================

end
