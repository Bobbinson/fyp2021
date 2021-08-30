# Deep Inconsistency

Project repository for getting a single set of weights which, loaded into the same architecture, on two different Deep Learning frameworks, have different behaviours, in an interesting way.

## Abstract

- There are many deep learning frameworks (pytorch, tensorflow, JAX, etc) and platforms (CPUs, GPUs, TPUs, embedded DSPs). Standard interchange facilities exist to allow the same architecture, with the same weights, to be run on various systems. It turns out that seemingly minor implementation details (order of summation, numeric precision, and the like) can cause these to differ slightly in their I/O behaviour. The goal of this project is to try to magnify this effect: to create networks which exhibit radically different behaviour on standard datasets when run on different platforms. Perhaps innocuous behaviour on one, and malicious behaviour on another.

- Pre-trained deep networks are routinely published and used. Due to the development of system-agnostic formats, these networks can be developed and saved using one framework, then loaded and run on different frameworks, using the same architecture and weights despite the different underlying numeric routines. As it happens, small differences (order of addition of lists of numbers, bits of precision, etc) can cause slightly different results when the same network is run on CPU vs GPU, Pytorch vs Keras/Tensorflow, etc. The objective of this project is to train networks (by calculating and following gradients using different objective functions on multiple frameworks simultaneously), which exhibit radically different behaviour on different frameworks. This could be used to produce networks which exhibit good test performance on, say, a cloud-based evaluation system while exhibiting pathological behaviour (confusing dogs and cats, failing to notice people who are wearing black cowboy hats, etc) on the target deployment platform (security cameras or phones, say). The possibilities for mischief are endless!

- Preliminary work using Pytorch vs Keras/Tensorflow switching 1-vs-7 on the MNIST dataset appear to show the viability of this idea.

The goal here is to (a) check the previous work, (b) extend it to much more malicious examples than just confusing 1s vs 7s in deployment but not in the lab.
