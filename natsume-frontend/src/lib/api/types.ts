export type SearchPosition = 'noun' | 'verb';
export type RankBy = 'raw' | 'meanPerMillion';

export type Corpus = {
	id: string;
	label: string;
	collocationCount: number;
	sentenceCount: number;
};

export type CorporaResponse = {
	corpora: Corpus[];
	databaseBuildId: string;
};

export type Suggestion = {
	lemma: string;
	pos: SearchPosition;
	occurrenceCount: number;
};

export type SuggestionsResponse = { suggestions: Suggestion[] };

export type CorpusContribution = {
	corpusId: string;
	rawFrequency: number;
};

export type CorpusDistribution = CorpusContribution & {
	frequencyPerMillion: number;
};

export type CollocationItem = {
	noun: string;
	particle: string;
	verb: string;
	totalRawFrequency: number;
	meanFrequencyPerMillion: number;
	contributions: CorpusContribution[];
};

export type ParticleGroup = {
	particle: string;
	totalMatchingCollocations: number;
	returnedCount: number;
	items: CollocationItem[];
	corpusDistribution: CorpusDistribution[];
};

export type CollocationsResponse = {
	particleGroups: ParticleGroup[];
	selectedCorpusIds: string[];
	rankBy: RankBy;
	databaseBuildId: string;
};

export type TextSpan = { start: number; end: number };

export type Example = {
	corpusId: string;
	sourceId: number;
	sourceTitle: string;
	sentenceId: number;
	text: string;
	nounSpan: TextSpan;
	particleSpan: TextSpan;
	verbSpan: TextSpan;
};

export type ExamplesResponse = {
	examples: Example[];
	selectedCorpusIds: string[];
	databaseBuildId: string;
};

export type PublicErrorEnvelope = {
	error: { code: string; message: string; requestId: string };
};
