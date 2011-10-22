# Summary

This is a simple implementation of discrete hidden markov models for Python > 2.5. It is currently written in pure python but will eventually be cythonized for cutting edge performances.

# Basic usage

	>>> from hmmpy import hmm
	>>> m=hmm(2, 2)
	>>> observations=[[0,0,0,0,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1],[0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0]]
	>>> ground_truths=[[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0]]
	>>> m.explicit_learn(observations, ground_truths)
	>>> m.viterbi(observations[1])
	([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], -21.944735122525493)
