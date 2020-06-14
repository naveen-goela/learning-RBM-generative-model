%
% sample_bRBM_hgv.m
%
% Description:
% Produce a sample of the hidden binary units  
% of a binary RBM given a sample of the visible 
% binary units. 
%
% USAGE:
% [h_sample] = sample_bRBM_hgv(bRBM, v_sample)
%
% INPUTS: 
% v_sample = sample of binary visible units.
% bRBM     = binary RBM with configured entries.
%   bRBM.W   = weight connections matrix of RBM.
%   bRBM.b_v = bias vector for visible units of RBM.
%   bRBM.b_h = bias vector for hidden units of RBM.
%
% OUTPUTS:
% h_sample = sample of binary hidden units.
%
% Author: N. Goela
% Date: January 31, 2015

function [h_sample] = sample_bRBM_hgv(bRBM, v_sample)

% Compute terms prior to logistic activation function.
pre_logistic = bRBM.W' * v_sample + bRBM.b_h; 

% Compute probabilities after logistic function.
after_logistic_prob = 1.0 ./ (1.0 + exp(-1.0 * pre_logistic)); 

% Obtain binary samples based on binomial probabilities. 
uniform_samples = rand(size(bRBM.b_h)); 
h_sample = 1.0 * (uniform_samples < after_logistic_prob); 
