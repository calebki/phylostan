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


	matrix[] calculate_jc69_p_matrices(vector blens){
		
		int bcount = rows(blens);
		matrix[4,4] pmats[bcount]; // probability matrices
		int index = 1;
		real d;
		
			for( b in 1:bcount ) {
				pmats[index] = rep_matrix(0.25 - 0.25*exp(-blens[b]/0.75), 4, 4);
				d = 0.25 + 0.75*exp(-blens[b]/0.75);
				for( i in 1:4 ) {
					pmats[index][i,i] = d;
				}
				index += 1;
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
	real lower_root;
	real lowers[2*S-1]; // list of lower bounds for each internal node (for reparametrization)
	int I; // number of intervals
}

transformed data{
	int bcount = 2*S-2; // number of branches
	int nodeCount = 2*S-1; // number of nodes
	int pCount = S-2; // number of proportions
	vector[4] freqs = rep_vector(0.25,4);
}

parameters{
	real <lower=0,upper=1> props[pCount]; // proportions
	real <lower=0> substrates[bcount];
	real <lower=0> ucln_mean;
	real <lower=0> ucln_stdev;
	real <lower=lower_root> height; // root height
	vector[I] thetas; // log space
	real<lower=0> tau;
}

transformed parameters{
	real <lower=0> heights[S-1];

	heights = transform(props, height, map, lowers);
}

model{
	vector[4] partials[2*S,L];   // partial probabilities for the S tips and S-1 internal nodes
	matrix[4,4] pmats[bcount]; // finite-time transition matrices for each branch
	vector [bcount] blens; // branch lengths

	substrates ~ lognormal(log(ucln_mean)-ucln_stdev*ucln_stdev*0.5, ucln_stdev);
	ucln_mean ~ exponential(1000);
	ucln_stdev ~ gamma(0.5396, 2.6184);
	heights ~ skyride_coalescent(thetas, map, lowers);
	thetas ~ gmrf(tau);
	tau ~ gamma(0.001, 0.001);

	
	// populate blens from heights array in preorder
	for( j in 2:nodeCount ){
		// internal node
		if(map[j,1] > S){
			blens[map[j,1]] = substrates[map[j,1]]*(heights[map[j,2]-S] - heights[map[j,1]-S]);
		}
		else{
			blens[map[j,1]] = substrates[map[j,1]]*(heights[map[j,2]-S] - lowers[map[j,1]]);
		}
	}

	pmats = calculate_jc69_p_matrices(blens);
	
	// copy tip data into node probability vector
	for( n in 1:S ) {
		for( i in 1:L ) {
			for( a in 1:4 ) {
				partials[n,i][a] = tipdata[n,i,a];
			}
		}
	}


	// calculate tree likelihood
	for( i in 1:L ) {
		for( n in 1:(S-1) ) {
			partials[peel[n,3],i] = (pmats[peel[n,1]]*partials[peel[n,1],i]) .* (pmats[peel[n,2]]*partials[peel[n,2],i]);
		}

		for(j in 1:4){
			partials[2*S,i][j] = partials[peel[S-1,3],i][j] * freqs[j];
		}
		// add the site log likelihood
		target += log(sum(partials[2*S,i]))*weights[i];
	}

	
	// add log det jacobian
	for( i in 2:nodeCount ){
		// skip leaves
		if(map[i,1] > S ){
			target += log(heights[map[i,2]-S] - lowers[map[i,1]]);
		}
	}

}

