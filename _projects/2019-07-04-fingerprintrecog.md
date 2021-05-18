---
layout: project
title: Pore-based Fingerprint Recognition System
description:
summary: A customized deep learning based fingerprint recognition system has been developed using the multitask deep convolutional neural network architecture to extract the fixed length representation of level 1, 2 & 3 features from a high resolution fingerprint image. The recognition system calculates the cosine similarity score between the two representations to generate a distribution of scores (genuine and imposter score distributions) and hence, plot a probability vs matching score graph to study the tradeoff between false match rate (FMR) and false non-match rate (FNMR).
category: Pattern Recognition, Convolutional Networks, Residual Connections, Image Analysis, Unsupervised Learning, CVPR
organization: Pattern Recognition and Image Analysis Lab, Indian Institute of Technology Indore
github: fingerprint-recognition
guide1: Dr. Vivek Kanhangad
guide2:
gurl1: http://www.iiti.ac.in/people/~kvivek/
gurl2:
highlight:
---

A customized deep learning based fingerprint recognition system has been developed using the multitask deep convolutional neural network architecture to extract the fixed length representation of level 1, 2 & 3 features from a high resolution fingerprint image. The recognition system calculates the cosine similarity score between the two representations to generate a distribution of scores (genuine and imposter score distributions) and hence, plot a probability $$p$$ vs matching score $$s$$ graph to study the tradeoff between false match rate (FMR) and false non-match rate (FNMR). The implementation of the deep learning architecture for this recognition system is done using PyTorch framework.