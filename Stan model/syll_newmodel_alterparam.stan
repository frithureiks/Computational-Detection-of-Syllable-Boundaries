data{
    int N;
    int NL;
    int NP;
    array[N] int S;
    array[N] int L;
    matrix[N, NP] dataMatrix;


}
parameters{

    matrix[1, 8] mus;
    matrix<lower=0>[1, 8] sigmas;
    matrix[NL, 8] zs; 
  
}
transformed parameters{

    matrix[NL, 8] effectsMatrix;
    for(i in 1:NL){
        effectsMatrix[i,1:8] = mus[1,1:8] + zs[i,1:8] .* sigmas[1,1:8] ;
    }

}

model{

    vector[N] p;
    for (i in 1:N){
        p[i] = effectsMatrix[L[i], 1] + sum(effectsMatrix[L[i], 2:8] .* dataMatrix[i, 1:7]);
    }

    S ~ bernoulli_logit(p);

    to_vector(mus) ~ std_normal();
    to_vector(zs) ~ std_normal();
    to_vector(sigmas) ~ exponential(1);
    
}

generated quantities {

}
