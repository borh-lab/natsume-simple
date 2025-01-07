export type Result = {
	n: string;
	v: string;
	frequency: number;
	corpus: string;
	p: string;
	contributions?: { corpus: string; frequency: number }[];
	mode: "n-pv" | "v-np";
};

export type CombinedResult = {
	n: string;
	p: string;
	v: string;
	frequency: number;
	contributions: { corpus: string; frequency: number }[];
};
