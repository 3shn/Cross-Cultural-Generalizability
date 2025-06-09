data {
  int N;
  array[N] int outcome;
}

parameters {
  real p;
}

model {
 outcome ~ bernoulli_logit(p);
}
