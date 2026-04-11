export const enum RetrievalMode {
    Hybrid = "hybrid",
    Vectors = "vectors",
    Text = "text"
}

export type ChatAppRequestOverrides = {
    use_advanced_flow?: boolean;
    retrieval_mode?: RetrievalMode;
    top?: number;
    temperature?: number;
    prompt_template?: string;
};

export type ChatAppRequestContext = {
    overrides: ChatAppRequestOverrides;
};

export type ChatAppRequestOptions = {
    context: ChatAppRequestContext;
};

export type ChatAppRequest = {
    input: { content: string; role: string }[];
    context: ChatAppRequestContext;
};

export type Thoughts = {
    title: string;
    description: any; // It can be any output from the api
    props?: { [key: string]: string };
};

export type RAGContext = {
    data_points: { [key: string]: any };
    thoughts: Thoughts[];
};

export type RAGChatCompletion = {
    output_text: string;
    context: RAGContext;
};

export type RAGChatCompletionDelta = {
    type: string;
    delta?: string;
    context?: RAGContext;
    error?: string;
};
