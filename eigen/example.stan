functions{
	// transform node heights to proportion, except for the root
	real loglik(vector blens);
	real[] transform(real[] p, real rootHeight, int[,] map){
		int S = size(p)+2;
		int nodeCount = S*2-1;
		
		real heights[S-1];
		int j = 1;
		
		heights[map[1,1]-S] = rootHeight;
		for( i in 2:nodeCount ){
			// internal node: transform
			if(map[i,1] > S){
				heights[map[i,1]-S] = (heights[map[i,2]-S])*p[j];
				j += 1;
			}
		}
		return heights;
	}
	

	real oneOnX_log(real x){
		return -log(x);
	}


	real constant_coalescent_log(real[] heights, real popSize, int[,] map){
		int S = size(heights)+1; // number of leaves from the number of internal nodes
		int nodeCount = size(heights) + S;

		real logP = 0.0;
		real lineageCount = 0.0; // first 2 intervals are sampling events

		int indices[nodeCount];
		int childCounts[nodeCount];
		real times[nodeCount];

		real start;
		real finish;
		real interval;
		real logPopSize = log(popSize);

		for( i in 1:nodeCount ){
			// internal node: transform
			if(map[i,1] > S){
				times[map[i,1]] = heights[map[i,1]-S];
				childCounts[map[i,1]] = 2;
			}
			else{
				times[map[i,1]] = 0;
				childCounts[map[i,1]] = 0;
			}
		}

		// calculate intervals
		indices = sort_indices_asc(times);

		// first tip
		start = times[indices[1]];

		for (i in 1:nodeCount) {
			finish = times[indices[i]];
			
			interval = finish - start;
			if(interval != 0.0){
				logP -= interval*((lineageCount*(lineageCount-1.0))/2.0)/popSize;
			}
			
			// sampling event
			if (childCounts[indices[i]] == 0) {
				lineageCount += 1.0;
			}
			// coalescent event
			else {
				lineageCount -= 1.0;
				logP -= logPopSize;
			}
			
			start = finish;
		}

		return logP;
	}
	
}

data{
	int <lower=0> L;                      // alignment length
	int <lower=0> S;                      // number of tips
	int map[2*S-1,2];                     // list of node in preorder [node,parent]
	real <lower=0> rate;
	real lower_root;
}

transformed data{
	int bcount = 2*S-2; // number of branches
	int nodeCount = 2*S-1; // number of nodes
	int pCount = S-2; // number of proportions
}

parameters{
	real <lower=0,upper=1> props[pCount]; // proportions
	real<lower=lower_root> height; // root height
	real <lower=0> theta;
}

transformed parameters{
	real <lower=0> heights[S-1];

	heights = transform(props, height, map);
}

model{
	vector[4] partials[2*S,L];   // partial probabilities for the S tips and S-1 internal nodes
	matrix[4,4] pmats[bcount]; // finite-time transition matrices for each branch
	vector [bcount] blens; // branch lengths

	theta ~ oneOnX();
	heights ~ constant_coalescent(theta, map);

	
	// populate blens from heights array in preorder
	for( j in 2:nodeCount ){
		// internal node
		if(map[j,1] > S){
			blens[map[j,1]] = rate*(heights[map[j,2]-S] - heights[map[j,1]-S]);
		}
		else{
			blens[map[j,1]] = rate*(heights[map[j,2]-S]);
		}
	}

	target += loglik(blens);
	
	// add log det jacobian
	for( i in 2:nodeCount ){
		// skip leaves
		if(map[i,1] > S ){
			target += log(heights[map[i,2]-S]);
		}
	}

}

