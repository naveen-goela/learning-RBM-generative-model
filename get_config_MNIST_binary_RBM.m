%
% get_config_MNIST_binary_RBM.m
%
% Description:
% Obtain initial configuration of a binary RBM 
% suitable for the MNIST public dataset.  
%
% USAGE:
% [binary_RBM_config] = get_config_MNIST_binary_RBM()
%
% INPUTS:
% None
%
% OUTPUTS: 
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
% Author: N. Goela
% Date: January 31, 2015

function [binary_RBM_config] = get_config_MNIST_binary_RBM()

binary_RBM_config.n_v = 28*28;
binary_RBM_config.n_h = 1000;
binary_RBM_config.m_w = 0.0;
binary_RBM_config.m_h = -0.2;
binary_RBM_config.m_v = -0.5;
binary_RBM_config.stdev_w = 0.05;
binary_RBM_config.stdev_v = 0.05;
binary_RBM_config.stdev_h = 0.05;
