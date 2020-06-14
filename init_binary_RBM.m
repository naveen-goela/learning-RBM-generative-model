%
% init_binary_RBM.m
%
% Description:
% Initialize a binary Restricted Boltzmann Machine 
% (RBM) using configuration parameters. An RBM 
% consists of a weight matrix and bias vectors.
%
% USAGE:
% [binary_RBM] = init_binary_RBM(binary_RBM_config)
%
% INPUTS: 
% binary_RBM_config  =  configuration of RBM.
%   binary_RBM_config.n_v = number of visible units.
%   binary_RBM_config.n_h = number of hidden units.
%   binary_RBM_config.m_w = mean of individual weights.
%   binary_RBM_config.m_v = mean of visible units.
%   binary_RBM_config.m_h = mean of hidden units.
%   binary_RBM_config.stdev_w = standard deviation of weights.
%   binary_RBM_config.stdev_v = standard deviation of visible units.
%   binary_RBM_config.stdev_h = standard deviation of hidden units.
% 
% 
% OUTPUTS:
% binary_RBM   = binary RBM with configured entries.
%   binary_RBM.W   = weight connections matrix of RBM.
%   binary_RBM.b_v = bias vector for visible units of RBM.
%   binary_RBM.b_h = bias vector for hidden units of RBM.
%
% Author: N. Goela
% Date: January 31, 2015

function [binary_RBM] = init_binary_RBM(binary_RBM_config)
     
% Initialize binary RBM.
binary_RBM.W = binary_RBM_config.m_w + ...
               (binary_RBM_config.stdev_w* ...
                       randn(binary_RBM_config.n_v, ...
                       binary_RBM_config.n_h));

% Initialize binary RBM bias vector for visible units.
binary_RBM.b_v = binary_RBM_config.m_v + ...
                 (binary_RBM_config.stdev_v* ...
                       randn(binary_RBM_config.n_v, 1)); 
                   
% Initialize binary RBM bias vector for hidden units.
binary_RBM.b_h = binary_RBM_config.m_h + ...
                 (binary_RBM_config.stdev_h* ...
                       randn(binary_RBM_config.n_h, 1));  
                   