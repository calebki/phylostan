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

	// calculate q
	real q_helper(vector A, vector B, vector t, int i, real s){
		if (t[i] == s){
			return 1;
		}
		else{
			real f;
			real g;
			f = exp(-A[i]*(s-t[i]));
			g = square(f * (1 + B[i]) + (1 - B[i]));
			return 4 * f / g;
		}
		
	}

	// calculate I
	int[] I(vector t, real[] x){
		int m = num_elements(t);
		int n = size(x);
		int i = 1;
		int j = 1;
		int Ix[n];

		while (x[i] > t[j]){
			j += 1;
		}

		while (i <= n){
			while (t[j] < x[i]){
				j += 1;
			}	
			Ix[i] = j;
			i += 1;
		}
		return Ix;
	}

	// Version without rho. No extra sampling effort.
	real bdsky_log(real[] heights, vector R, vector delta, vector s, //vector rho, 
				   real x1, int m, int[,] map, //vector N, 
				   real[] lowers, real[] sample_times){

		int S = size(heights)+1; // number of leaves from the number of internal nodes
		int nodeCount = size(heights) + S;

		real heights_sorted[S-1] = sort_desc(heights);
		real height = heights_sorted[1];
		real sample_times_sorted[S] = sort_desc(sample_times);

		//parameter transform
		vector[m] lambda = R .* delta;
		vector[m] psi = s .* delta;
		vector[m] mu = delta - psi;

		//recursion parameters
		vector[m] A = sqrt(square(lambda - mu - psi) + 4 * lambda .* psi);
		vector[m] B;
		vector[m+1] p;
		vector[2*S - 2 + m] q;
		real q0;
		
		// helper parameters to calculate likelihood
		real increment = (height + x1)/m;
		vector[m] t;
		int vCount[m-1];
		real l;
		real u;
		real x[S-1]; 
		int Ix[S-1];
		real y[S];
		int Iy[S];

		// helper parameters for B and p recursion loop
		real first;
		real second;
		real t_z_1;
		int z;

		// helper parameters for q recursion loop
		real f;
		real g;

		// log likelihood
		real logP;
		vector[4*S + 2*m - 2] summands;

		for (k in 1:m){
			t[k] = k * increment;
		}

		// Times is different here than in in skyride

		// Define vCount. Number of 2 degree vertices in at time t_i
		// Defined as n_i in Stadler 2013

		vCount = rep_array(0, m-1);
		for ( k in 2:nodeCount ){
			if(map[k,1] > S){
				l = t[m] - heights[map[k,2]-S];
				u = t[m] - heights[map[k,1]-S];
			}
			else{
				l = t[m] - heights[map[k,2]-S];
				u = t[m] - lowers[k];
			}
			for ( w in 1:m ){
				if ((u > t[w]) && (t[w] > l)){
					vCount[w] += 1;
					if (vCount[w] > S){
						reject("n_i out of bounds. n_i=", vCount[w])
					}
				}
			}
		}

		for ( k in 1:S-1 ){
			x[k] = height - heights_sorted[k];
			y[k] = height - sample_times_sorted[k];
		}
		y[S] = height - sample_times_sorted[S];
		
		// Define I(x_i) and I(y_i)
		Ix = I(t, x);
		Iy = I(t, y);

		// print("Height: ", height+x1);
		// print("heights: ", heights_sorted);
		// print("t: ", t);
		// print("x: ", x);
		// print("Ix: ", Ix);
		// print("Iy: ", Iy);

		// print("A: ", A)
		// Recursion for B and p
		p[m+1] = 1;

		for( k in 0:m-1 ){
			z = m-k;
			B[z] = ((1-2*p[z+1]) * lambda[z] + mu[z] + psi[z])/A[z];
			if (z > 1){
				first = exp(A[z] * (t[z]-t[z-1])) * (1 + B[z]);
			}
			else{
				first = exp(A[z] * (t[z]-0)) * (1 + B[z]);
			}
			second = 1 - B[z];
			p[z] = (lambda[z] + mu[z] + psi[z] - A[z] * (first - second) / (first + second))/ (2 * lambda[z]);
		}
		// print("B: ", B);

		for( k in 1:S-1 ){
			q[k] = q_helper(A, B, t, Ix[k], x[k]);
		}

		for( k in S:2*S-1 ){
			z = k - (S-1);
			q[k] = q_helper(A, B, t, Iy[z], y[z]);
		}

		for( k in 2*S:2*S+m-2){
			z = k - (2*S-1);
			q[k] = q_helper(A, B, t, z+1, t[z]);
		}

		q0 = q_helper(A, B, t, 1, 0);
		// print("q: ", q)
		// print("p:", p)

		summands[1] = log(q0 / (1 - p[1]));
		// print("Outer left term: ", summands[0]);
		for ( k in 1:S-1 ){
			summands[k+1] = log(lambda[Ix[k]]);
			summands[k+S] = log(q[k]);
		}
		// print("x prod: ", summands[1:2*S-1]);

		for ( k in 1:S ){
			summands[2*S-1+k] = log(psi[Iy[k]]);
			summands[3*S-1+k] = -log(q[k+(S-1)]);
		}
		// print("y prod: ", summands[2*S:4*S-1]);

		for ( k in 1:m-1){
			//logP += N[i] * log(rho[i]);
			summands[4*S-1+k] = vCount[k] * log(q[k+(2*S-1)]);
		}
		// print("n_i: ", vCount)
		// print("t prod: ", summands[4*S:4*S+m-2]);

		//Add Jacobian term for transform
		for ( k in 1:m ){
			summands[4*S+m-2+k] = log(square(delta[k]));
		}
		// print("jacobian: ", summands[4*S+m-1:]);
		
		logP = sum(summands);
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
	real lowers[2*S-1]; // list of lower bos_sorted[1]unds for each internal node (for reparametrization)
	int m; // number of intervals
	// vector<lower=0>[m] t; // time discretization
	// int<lower=0>[m] N // number of tips sampled at t_i
	// int n; // number of sequentially sampled tips
	real<lower=0> sample_times[S]; // sampling time of sequentially sampled tips
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
	real <lower=0> x1; //length of edge above root
	real<lower=0> tau;
	vector<lower=0>[m] R; // effective reproductive number
	vector<lower=0>[m] delta; // become uninfectious rate
	vector<lower=0.000001,upper=0.999999>[m] s; // probability of being sampled
	//vector<lower=0>[m] rho; // sampling rate at each t_i
	simplex[6] rates;
	simplex[4] freqs;
}

transformed parameters{
	vector[C] ps = rep_vector(1.0/C, C);
	vector[C] rs;
	real <lower=0> heights[S-1];

	
		{
			real h = 0;
			for(i in 1:C){
				rs[i] = pow(-log(1.0 - (2.0*(i-1)+1.0)/(2.0*C)), 1.0/wshape);
			}
			h = sum(rs)/C;
			for(i in 1:C){
				rs[i] /= h;		
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
	heights ~ bdsky(R, delta, s, x1, m, map, lowers, sample_times);
	R ~ lognormal(0.5, 1);
	delta ~ lognormal(-1,1);
	s ~ beta(1,1);
	//rho ~ beta(1,99);
	x1 ~ lognormal(1, 1.25);
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


