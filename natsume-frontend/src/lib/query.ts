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
