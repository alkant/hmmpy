from __future__ import division
from math import log
from copy import deepcopy

def fromtables(pi, t, e):
    """Constructs a HMM from probability tables.

    Parameters
    ----------
    pi: list of floats between 0 and 1
        Initial state probabilities p(s_0)=pi[s_0].
    t: list of list of floats between 0 and 1
        Transition matrix p(s_j|s_i)=t[s_i][s_j].
    e: list of list of floats between 0 and 1
        Emission matrix p(o_i|s_i)=e[s_i][o_i]."""

    #sanity checks
    nStates=len(pi)
    assert(nStates==len(t) and nStates==len(e) and nStates>0)
    nObs=len(e[0])
    for i in range(nStates):
        assert(len(t[i])==nStates and len(e[i])==nObs)

    m=hmm(nStates, nObs)
    m.pi=deepcopy(pi)
    m.t=deepcopy(t)
    m.e=deepcopy(e)

    return m

class hmm:
    def __init__(self, nStates, nObs):
        """HMM constructor.
        
        Parameters
        ----------
        nStates: non-negative integer
            Number of hidden states.
        nObs: non-negative integer
            Number of possible observed values."""
        self.pi=[0]*nStates
        self.t=[[0]*nStates for i in range(nStates)]
        self.e=[[0]*nObs for i in range(nStates)]
        self.nStates=nStates
        self.nObs=nObs
        self.logdomain=False

    def learn(self, observations, ground_truths):
        """Learns from a list of observation sequences and their associated ground truth.

        Parameters
        ----------
        observations: list of list of integers in {0, ..., nObs-1}
            List of observed sequences.
        ground_truths: list of list of integers in {0, ..., nStates-1}
            Associated list of ground truths.""" 
        assert(len(observations)==len(ground_truths))
        self.__init__(self.nStates, self.nObs)
        N=len(observations)
        for i in range(N):
            o=observations[i]
            t=ground_truths[i]
            assert(len(o)==len(t))

            self.pi[t[0]]+=1
            for j in range(len(t)-1):
                self.t[t[j]][t[j+1]]+=1
                self.e[t[j]][o[j]]+=1
            j+=1
            self.e[t[j]][o[j]]+=1

        for i in range(self.nStates):
            self.pi[i]/=N
            Zt=sum(self.t[i])
            Ze=sum(self.e[i])
            for j in range(self.nStates):
                self.t[i][j]/=max(1, Zt)
            for j in range(self.nObs):
                self.e[i][j]/=max(1, Ze)

    def __convert_to_log(self):
        """Convers the internal probability tables in the log domain."""
        for i in range(self.nStates):
            if self.pi[i]>0:
                self.pi[i]=log(self.pi[i])
            else:
                self.pi[i]=float('-inf')
            for j in range(self.nStates):
                if self.t[i][j]>0:
                    self.t[i][j]=log(self.t[i][j])
                else:
                    self.t[i][j]=float('-inf')
            for j in range(self.nObs):
                if self.e[i][j]>0:
                    self.e[i][j]=log(self.e[i][j])
                else:
                    self.e[i][j]=float('-inf')
        self.logdomain=True

    def viterbi(self, observation):
        """Viterbi inference of the highest likelihood hidden states sequence given the observations. Time complexity is O(|observation|*nStates^2).
        
        Parameters
        ----------
        observation: List of integers in {0, ..., nObs-1}.

        Returns
        -------
        seq: List of integers in {0, ..., nStates-1}
            Highest likelihood infered sequence of hidden states.
        loglike: negative float
            Loglikelihood of the model for that sequence."""
        N=len(observation)
        tab=[[0]*self.nStates for i in range(N)]
        backtrack=[[-1]*self.nStates for i in range(N)]
        if not self.logdomain:
            self.__convert_to_log()

        for i in range(self.nStates):
            tab[0][i]=self.e[i][observation[0]]+self.pi[i]
        
        for i in range(1,N):
            for j in range(self.nStates):
                smax=-1
                maxval=float('-inf')
                for s in range(self.nStates):
                    cs=tab[i-1][s]+self.t[s][j]
                    if cs>maxval:
                        smax=s
                        maxval=cs
                assert(smax>-1 and smax<self.nStates)
                tab[i][j]=self.e[j][observation[i]]+maxval
                backtrack[i][j]=smax

        smax=-1
        llike=float('-inf')
        for s in range(self.nStates):
            if llike<tab[N-1][s]:
                llike=tab[N-1][s]
                smax=s

        best=[-1]*N
        best[-1]=smax
        for i in range(N-2, -1, -1):
            best[i]=backtrack[i+1][best[i+1]]

        return best, llike
