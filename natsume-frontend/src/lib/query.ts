export type Contribution = {
	corpus: string;
	normalizedFrequency: number;
	rawFrequency: number;
};

export type Collocate = {
	n: string;
	p: string;
	v: string;
	contributions: Contribution[];
};

export type Result = {
	n: string;
	v: string;
	frequency: number;
	corpus: string;
	p: string;
	contributions: Contribution[];
	mode: 'n-pv' | 'v-np';
};

export type CombinedResult = {
	n: string;
	p: string;
	v: string;
	frequency: number;
	contributions: Contribution[];
};
