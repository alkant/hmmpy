from cpython cimport bool
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log

cdef float MIN_FLOAT = float('-inf')

cdef float csum(float* tab, int n):
    cdef int i=0
    cdef float res=0
    while i<n:
        res+=tab[i]
        i+=1
    return res

def fromtables(list pi, list t, list e):
    """Constructs a HMM from probability tables.

    Parameters
    ----------
    pi: list of floats between 0 and 1
        Initial state probabilities p(s_0)=pi[s_0].
    t: list of list of floats between 0 and 1
        Transition matrix p(s_j|s_i)=t[s_i][s_j].
    e: list of list of floats between 0 and 1
        Emission matrix p(o_i|s_i)=e[s_i][o_i]."""

    cdef int i, j
        
    #sanity checks
    nStates=len(pi)
    assert(nStates==len(t) and nStates==len(e) and nStates>0)
    nObs=len(e[0])
    for i in range(nStates):
        assert(len(t[i])==nStates and len(e[i])==nObs)

    m=hmm(nStates, nObs)
    for i in range(nStates):
        m.pi[i]=pi[i]
        for j in range(nStates):
            m.t[i][j]=t[i][j]
        for j in range(nObs):
            m.e[i][j]=e[i][j]
    return m

cdef class hmm:
    cdef public int nStates, nObs
    cdef bool logdomain
    cdef float* pi #initial state probas
    cdef float** t #transition probas
    cdef float** e #emission probas
    
    cdef float** tab #cached viterbi table
    cdef int** backtrack #cached viterbi backtrack table
    cdef int seqSize #size of the cached viterbi tables

    def __init__(self, int nStates, int nObs):
        """HMM constructor.
        
        Parameters
        ----------
        nStates: non-negative integer
            Number of hidden states.
        nObs: non-negative integer
            Number of possible observed values."""
        self.pi= <float*> calloc(nStates, sizeof(float))
        self.t= <float**> malloc(nStates*sizeof(float*))
        self.e= <float**> malloc(nObs*sizeof(float*))
        cdef int i=0
        for i in range(nStates):
                self.t[i]= <float*> calloc(nStates, sizeof(float))
                self.e[i]= <float*> calloc(nObs, sizeof(float))
                i+=1
        
        self.nStates=nStates
        self.nObs=nObs
        self.logdomain=False
        self.seqSize=-1

    def learn(self, list observations, list ground_truths):
        """Learns from a list of observation sequences and their associated ground truth.

        Parameters
        ----------
        observations: list of list of integers in {0, ..., nObs-1}
            List of observed sequences.
        ground_truths: list of list of integers in {0, ..., nStates-1}
            Associated list of ground truths.""" 
        self.__init__(self.nStates, self.nObs)
        N=len(observations)
        cdef int i, j
        for i in xrange(N):
            o=observations[i]
            t=ground_truths[i]

            self.pi[t[0]]+=1
            for j in range(len(t)-1):
                self.t[t[j]][t[j+1]]+=1 #possible loss of precision
                self.e[t[j]][o[j]]+=1   #rewrite with int** types
            j+=1
            self.e[t[j]][o[j]]+=1

        cdef float Zt, Ze
        for i in range(self.nStates):
            self.pi[i]/=N
            Zt= csum(self.t[i], self.nStates)
            Ze= csum(self.e[i], self.nObs)
            for j in range(self.nStates):
                self.t[i][j]/=max(1., Zt)
            for j in range(self.nObs):
                self.e[i][j]/=max(1., Ze)

    def __convert_to_log(self):
        """Convers the internal probability tables in the log domain."""
        cdef int i, j
        for i in range(self.nStates):
            if self.pi[i]>0:
                self.pi[i]=log(self.pi[i])
            else:
                self.pi[i]=MIN_FLOAT
            for j in range(self.nStates):
                if self.t[i][j]>0:
                    self.t[i][j]=log(self.t[i][j])
                else:
                    self.t[i][j]=MIN_FLOAT
            for j in range(self.nObs):
                if self.e[i][j]>0:
                    self.e[i][j]=log(self.e[i][j])
                else:
                    self.e[i][j]=MIN_FLOAT
        self.logdomain=True

    cdef void __init_viterbi_tables(self, int n):
        """Initialize viterbi cache."""
        self.flush()
        self.tab = <float**> malloc(n*sizeof(float*))
        self.backtrack = <int**> malloc(n*sizeof(int*))
        cdef int i
        for i in xrange(n):
            self.tab[i] = <float*> malloc(self.nStates*sizeof(float))
            self.backtrack[i] = <int*> malloc(self.nStates*sizeof(int))
        self.seqSize=n

    def flush(self):
        """Free cached memory of viterbi inference."""
        if self.seqSize<1: return
        cdef int i
        for i in xrange(self.seqSize):
            free(self.tab[i])
            free(self.backtrack[i])
        free(self.tab)
        free(self.backtrack)

    def viterbi(self, list obs):
        """Viterbi inference of the highest likelihood hidden states sequence given the observations. Time complexity is O(|observation|*nStates^2).
        
        Parameters
        ----------
        obs: List of integers in {0, ..., nObs-1}
            The observations.

        Returns
        -------
        seq: List of integers in {0, ..., nStates-1}
            Highest likelihood infered sequence of hidden states.
        loglike: negative float
            Loglikelihood of the model for that sequence."""
        cdef int n=len(obs)
        cdef int i, j, s, smax
        cdef float maxval, llike, cs

        cdef int* observation= <int*> malloc(n*sizeof(int))
        i=0
        for val in obs:
            observation[i]=val
            i+=1

        if n>self.seqSize:
            self.__init_viterbi_tables(n)

        if not self.logdomain:
            self.__convert_to_log()

        for i in range(self.nStates):
            self.tab[0][i]=self.e[i][observation[0]]+self.pi[i]
        
        for i in xrange(1,n):
            for j in range(self.nStates):
                smax=-1
                maxval=MIN_FLOAT
                for s in range(self.nStates):
                    cs=self.tab[i-1][s]+self.t[s][j]
                    if cs>maxval:
                        smax=s
                        maxval=cs
                self.tab[i][j]=self.e[j][observation[i]]+maxval
                self.backtrack[i][j]=smax

        smax=-1
        llike=MIN_FLOAT
        for s in range(self.nStates):
            if llike<self.tab[n-1][s]:
                llike=self.tab[n-1][s]
                smax=s

        cdef int* best= <int*> malloc(n*sizeof(int))
        best[n-1]=smax
        for i in xrange(n-2, -1, -1):
            best[i]=self.backtrack[i+1][best[i+1]]
        
        lbest=[0]*n
        for i in xrange(n):
            lbest[i]=best[i]

        free(observation)
        free(best)
        return lbest, llike

    def __del__(self):
        self.flush()
        cdef int i
        for i in range(self.nStates):
            free(self.t[i])
            free(self.e[i])
        free(self.t)
        free(self.e)
        free(self.pi)
