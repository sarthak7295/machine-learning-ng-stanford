function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1=[ones(m,1) X];
z1=a1*Theta1';
a2=sigmoid(z1);
a2 = [ones(m,1) a2]; 
z2=a2*Theta2';
a3=sigmoid(z2);
%we also need to convert y to matrixs with 010000..00 ,00100..00 etc
y = repmat([1:num_labels], size(y), 1) == repmat(y, 1, num_labels);

% repmat(A,M,N) = repeat A with M times row, N times col

J=(-1/m)*sum(sum((y.*log(a3))+((1-y).*log(1-a3))));

regularizationTheta1=Theta1(:,2:end);    %we dont use the bias theta for reularization
regularizationTheta2=Theta2(:,2:end);
reularizationTerm=(lambda/(2*m))*((sum(sum(regularizationTheta1.^2)))+...
          (sum(sum(regularizationTheta2.^2))));  %regularization term is sum of all thetas

J=J+reularizationTerm;

del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));

for i=1:m
  ai1=a1(i,:);
  ai2=a2(i,:);
  ai3=a3(i,:);
  z2temp=Theta1*ai1';
  z2temp=[1;z2temp];   %adding the bias ....we can keep the bias as anything
  d3=ai3-y(i,:);        %the mchine will figure out its weight thus creating the
                        %perfect bias
  d2=(Theta2'*d3').*sigmoidGradient(z2temp);
  del1=del1+d2(2:end)*ai1 ;   %since d2(1) is of bias unit and is not connected
                              % backwards
  del2=del2+d3'*ai2;                              
endfor
%adding zeros in starting to maintain matrix multiplication
%zero because first term is not considered in reularization
Theta1_grad=(1/m)*(del1+(lambda*[zeros(size(Theta1,1),1) ...
              regularizationTheta1]));
Theta2_grad=(1/m)*(del2+(lambda*[zeros(size(Theta2,1),1) ...
              regularizationTheta2]));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
