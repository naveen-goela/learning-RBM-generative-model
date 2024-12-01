%
% train_RBM.m
%
% Description:
% Train a restricted Bolzmann machine (RBM) given
% a set of training images. 
%
% USAGE:
% [bRBM] = train_RBM(training_images)
%
% INPUTS:
% training_images = set of training samples.
% training_images(:,k) specifies the k-th training vector sample.
% training_images(:,k) for MNIST data is 784 x 60000. 
%
% OUTPUTS: 
% bRBM     = binary RBM trained from images.
%   bRBM.W   = weight connections matrix of RBM.
%   bRBM.b_v = bias vector for visible units of RBM.
%   bRBM.b_h = bias vector for hidden units of RBM.
%
% Author: N. Goela
% Date: January 31, 2015

function [bRBM] = train_RBM(training_images)

% Initialize RBM parameters.
conf = get_config_MNIST_binary_RBM(); 
bRBM = init_binary_RBM(conf); 

% Number of epochs across training set.
total_epochs = 20; 

% Gradient descent epsilon step size. 
eps = 0.001; 

% Process all data samples per epoch of training. 
for epoch=1:total_epochs
    
    % Display epoch number. 
    fprintf('Training epoch %d of %d ... \n', epoch, total_epochs);
    
    % Random order of sampling from training set.
    order_data = randperm(length(training_images));  
    for k=1:length(order_data)
        
        if (mod(k, 10000) == 0)
            disp('Completed 10000 training samples')
        end
        
        % Obtain binary vector sample from training set. 
        v_data = training_images(:,order_data(k));  
    
        % Compute constrastive divergence (CD) update. 
        [gradient] = compute_CD_update(bRBM, v_data); 

        % Update generative model. 
        bRBM.W   = bRBM.W   + eps*gradient.W; 
        bRBM.b_v = bRBM.b_v + eps*gradient.b_v;
        bRBM.b_h = bRBM.b_h + eps*gradient.b_h;                 
    end
    
    % Save RBM every epoch of training.
    fprintf('Saving to file ... \n \n');
    save_file_name = sprintf('bRBM_epoch_%d.mat', epoch); 
    save(save_file_name, 'bRBM'); 
end
