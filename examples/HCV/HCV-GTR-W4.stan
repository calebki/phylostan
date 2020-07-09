functions{
	// transform node heights to proportion, except for the root
	real[] transform(real[] p, real rootHeight, int[,] map, real[] lowers){
		int S = size(p)+2;
		int nodeCount = S*2-1;
		
		real heights[S-1];
		int j = 1;
		
		heights[map[1,1]-S] = rootHeight;
		for( i in 2:nodeCount ){
			// internal node: transform
			if(map[i,1] > S){
				heights[map[i,1]-S] = lowers[map[i,1]] + (heights[map[i,2]-S] - lowers[map[i,1]])*p[j];
				j += 1;
			}
		}
		return heights;
	}
	

	real skyride_coalescent_log(real[] heights, vector pop, int[,] map, real[] lowers){
		int S = size(heights)+1; // number of leaves from the number of internal nodes
		int nodeCount = size(heights) + S;

		real logP = 0.0;
		int index = 1;
		real lineageCount = 0.0;

		int indices[nodeCount];
		int childCounts[nodeCount];
		real times[nodeCount];
		real start;
		real finish;
		real interval;

		for( i in 1:nodeCount ){
			// internal node: transform
			if(map[i,1] > S){
				times[i] = heights[map[i,1]-S];
				childCounts[i] = 2;
			}
			else{
				times[i] = lowers[map[i,1]];
				childCounts[i] = 0;
			}
		}

		// calculate intervals
		indices = sort_indices_asc(times);

		// first tip
		start = times[indices[1]];

		for (i in 1:nodeCount) {
			finish = times[indices[i]];

			interval = finish - start;
			// consecutive sampling events
			if(interval != 0.0){
				logP -= interval*((lineageCount*(lineageCount-1.0))/2.0)/exp(pop[index]);
				if (childCounts[indices[i]] != 0) {
					logP -= pop[index];
					index += 1;
				}
			}

			// sampling event
			if (childCounts[indices[i]] == 0) {
				lineageCount += 1.0;
			}
			// coalescent event
			else {
				lineageCount -= 1.0;
			}

			start = finish;
		}

		return logP;
	}


	// Not time aware
	real gmrf_log(vector logPopSize, real precision){
		int N = rows(logPopSize);
		real realN = N;
		real s = 0;
		for (i in 2:N){
			s += pow(logPopSize[i]-logPopSize[i-1], 2.0);
		}
		return log(precision)*(realN - 1.0)/2.0 - s*precision/2.0 - (realN - 1.0)/2.0 * log(2.0*pi());
	}


	matrix[] calculate_gtr_p_matrices(vector freqs, vector rates, vector blens, vector rs){
		int C = rows(rs);
		int bcount = rows(blens);
		matrix[4,4] pmats[bcount*C]; // probability matrices
		
		matrix[4,4] Q; // rate matrix
		matrix[4,4] P2 = diag_matrix(sqrt(freqs));        // diagonal sqrt frequencies
		matrix[4,4] P2inv = diag_matrix(1.0 ./ sqrt(freqs)); // diagonal inverse sqrt frequencies
		matrix[4,4] A; // another symmetric matrix
		vector[4] eigenvalues;
		matrix[4,4] eigenvectors;
		matrix[4,4] m1;
		matrix[4,4] m2;
		// symmetric rate matrix
		matrix[4,4] R = [[0.0, rates[1], rates[2], rates[3]],
						 [rates[1], 0.0, rates[4], rates[5]],
						 [rates[2], rates[4], 0.0, rates[6]],
						 [rates[3], rates[5], rates[6], 0.0]];
		real s = 0.0;
		int index = 1;

		Q = R * diag_matrix(freqs);
		for (i in 1:4) {
			Q[i,i] = 0.0;
			Q[i,i] = -sum(Q[i,1:4]);
			s -= Q[i,i] * freqs[i];
		}
		Q /= s;

		A = P2 * Q * P2inv;

		eigenvalues = eigenvalues_sym(A);
		eigenvectors = eigenvectors_sym(A);

		m1 = P2inv * eigenvectors;
		m2 = eigenvectors' * P2;

		for(c in 1:C){
			for( b in 1:bcount ){
				pmats[index] = m1 * diag_matrix(exp(eigenvalues*blens[b]*rs[c])) * m2;
				index += 1;
			}
		}

		return pmats;
	}
	
}

data{
	int <lower=0> L;                      // alignment length
	int <lower=0> S;                      // number of tips
	real<lower=0,upper=1> tipdata[S,L,4]; // alignment as partials
	int <lower=0,upper=2*S> peel[S-1,3];  // list of nodes for peeling
	real weights[L];
	int map[2*S-1,2];                     // list of node in preorder [node,parent]
	int C;
	real lower_root;
	real lowers[2*S-1]; // list of lower bounds for each internal node (for reparametrization)
	int I; // number of intervals
	vector<lower=0>[4] frequencies_alpha; // parameters of the prior on frequencies
	vector<lower=0>[6] rates_alpha;       // parameters of the prior on rates
}

transformed data{
	int bcount = 2*S-2; // number of branches
	int nodeCount = 2*S-1; // number of nodes
	int pCount = S-2; // number of proportions
}

parameters{
	real<lower=0.1> wshape;
	real <lower=0,upper=1> props[pCount]; // proportions
	real <lower=0> rate;
	real <lower=lower_root> height; // root height
	vector[I] thetas; // log space
	real<lower=0> tau;
	simplex[6] rates;
	simplex[4] freqs;
}

transformed parameters{
	vector[C] ps = rep_vector(1.0/C, C);
	vector[C] rs;
	real <lower=0> heights[S-1];

	
		{
			real m = 0;
			for(i in 1:C){
				rs[i] = pow(-log(1.0 - (2.0*(i-1)+1.0)/(2.0*C)), 1.0/wshape);
			}
			m = sum(rs)/C;
			for(i in 1:C){
				rs[i] /= m;		
			}
		}

	heights = transform(props, height, map, lowers);
}

model{
	real probs[C];
	vector[4] partials[C,2*S,L];   // partial probabilities for the S tips and S-1 internal nodes
	matrix[4,4] pmats[bcount*C]; // finite-time transition matrices for each branch
	vector [bcount] blens; // branch lengths

	wshape ~ exponential(1.0);
	rate ~ exponential(1000);
	heights ~ skyride_coalescent(thetas, map, lowers);
	thetas ~ gmrf(tau);
	tau ~ gamma(0.001, 0.001);
	rates ~ dirichlet(rates_alpha);
	freqs ~ dirichlet(frequencies_alpha);

	
	// populate blens from heights array in preorder
	for( j in 2:nodeCount ){
		// internal node
		if(map[j,1] > S){
			blens[map[j,1]] = rate*(heights[map[j,2]-S] - heights[map[j,1]-S]);
		}
		else{
			blens[map[j,1]] = rate*(heights[map[j,2]-S] - lowers[map[j,1]]);
		}
	}

	pmats = calculate_gtr_p_matrices(freqs, rates, blens, rs);
	
	// copy tip data into node probability vector
	for( n in 1:S ) {
		for( i in 1:L ) {
			for( a in 1:4 ) {
				for(c in 1:C){
					partials[c,n,i][a] = tipdata[n,i,a];
				}
			}
		}
	}


	// calculate tree likelihood
	for( i in 1:L ) {
		for( n in 1:(S-1) ) {
			for(c in 1:C){
				partials[c,peel[n,3],i] = (pmats[peel[n,1]+(c-1)*bcount]*partials[c,peel[n,1],i]) .* (pmats[peel[n,2]+(c-1)*bcount]*partials[c,peel[n,2],i]);
			}
		}
		for(c in 1:C){
			probs[c] = ps[c] * sum(partials[c,peel[S-1,3],i] .* freqs);
		}
		// add the site log likelihood
		target += log(sum(probs))*weights[i];
	}

	
	// add log det jacobian
	for( i in 2:nodeCount ){
		// skip leaves
		if(map[i,1] > S ){
			target += log(heights[map[i,2]-S] - lowers[map[i,1]]);
		}
	}

}

