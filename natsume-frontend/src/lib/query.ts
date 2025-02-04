declare global {
    interface Window {
        NATSUME_API_URL?: string;
    }
}

function getApiUrl() {
    // First try runtime environment variable
    const runtimeUrl = window.NATSUME_API_URL;
    if (runtimeUrl) return runtimeUrl;
    
    // Then try build-time environment variable
    const buildTimeUrl = import.meta.env.VITE_API_URL;
    if (buildTimeUrl) return buildTimeUrl;
    
    // Finally fall back to default
    return "http://localhost:8000";
}

export const API_URL = getApiUrl();

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
