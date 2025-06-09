data {
  int N;
  int N_pop;
  int MA;
  array[N] int age;
  array[N] int pop_id;
  array[N] int condition;
  array[N] int outcome;
  array[N] int gender;
  array[N_pop, MA] int Demo;
  int Ref;
  }

parameters {
  array[N_pop] real alpha;
  array[N_pop] real b_prime;
}

model {
  vector[N] p;
  alpha ~ normal(0, 2);
  b_prime ~ normal(0, 2);
  for ( i in 1:N ) {
   p[i] = alpha[pop_id[i]] + b_prime[pop_id[i]] * condition[i];
  }
 outcome ~ binomial_logit(1, p);
}

generated quantities{

array[N_pop] real empirical_p;
 for (h in 1:N_pop){
   empirical_p[h] =  inv_logit(alpha[h]) - inv_logit(alpha[h] + b_prime[h]);
 }
}
