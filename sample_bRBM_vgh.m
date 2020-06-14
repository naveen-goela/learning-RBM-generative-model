%
% sample_bRBM_vgh.m
%
% Description:
% Produce a sample of the visible binary units  
% of a binary RBM given a sample of the hidden 
% binary units. 
%
% USAGE:
% [v_sample] = sample_bRBM_hgv(bRBM, h_sample)
%
% INPUTS: 
% h_sample = sample of binary hidden units.
% bRBM     = binary RBM with configured entries.
%   bRBM.W   = weight connections matrix of RBM.
%   bRBM.b_v = bias vector for visible units of RBM.
%   bRBM.b_h = bias vector for hidden units of RBM.
%
% OUTPUTS:
% v_sample = sample of binary visible units.
%
% Author: N. Goela
% Date: January 31, 2015

function [v_sample] = sample_bRBM_vgh(bRBM, h_sample)

% Compute terms prior to logistic activation function.
pre_logistic = bRBM.W * h_sample + bRBM.b_v; 

% Compute probabilities after logistic function.
after_logistic_prob = 1.0 ./ (1.0 + exp(-1.0 * pre_logistic));  

% Obtain samples based on probabilities. 
uniform_samples = rand(size(bRBM.b_v)); 
v_sample = 1.0 * (uniform_samples < after_logistic_prob); 

