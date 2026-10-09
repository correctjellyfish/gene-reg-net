# Gene Regulatory Networks in Python

Welcome to gene-reg-net! This project aims to provide a simple inferface for
working with gene regulatory networks in python, by integrating NetworkX, SciPy
and more to enable easy investigation of GRN topology.

## Roadmap

- Implement Ising approach
  - Each gene (or other entity) is assigned and activity that is one of 2 values
    (i.e. [0,1], or [-1,1])
  - At each timestep, the incoming regulation for each gene (i.e. 1 for
    activation, -1 for repression) is multiplied by the regulating nodes current
    activity. The sum of these values is taken, and the gene's activity is
    updated according to
    - If the sum is greater than 0, the gene's activity is set to 1
    - If the sum is less than 0, the gene's activity is set to 0 (or -1
      depending on user input)
    - If the sum is 0, the gene keeps its previous activity level
  - The Updates are repeated to create a time course. Which genes are updated at
    each step is determined by a schedule. For example synchronous
    simultaneously updates all genes, while asynchronous updates a single
    randomly selected gene at a time.
  - To implement:
    - Function which returns full time course
    - Function which returns steady state
      - For the synchronous updates, this can check two consecutive states are
        the same, for other schedules it can just run for a certain number of
        steps or have a longer window of no activity changes.
    - Functions for generating initial states
      - Uniform random states
      - Uniform random proportion, randomly selected
        - Generate p~[0,1], and randomly select that proportion of genes to be
          active
      - Sobol sampling/Latin hypercubes/Halton/Poisson disc/... (Not sure if
        useful in this case)
