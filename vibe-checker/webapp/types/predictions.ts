export interface PredictionMetadata {
  document_id: string;
  document_name: string;
  document_slug: string;
  translated: string;
  publication_ts: string;
  "document_metadata.corpus_type_name": string;
  world_bank_region: string;
  similarity: string;
}

export interface Prediction {
  text: string;
  marked_up_text?: string;
  spans?: Array<{ start: number; end: number; label: string }>;
  metadata: PredictionMetadata;
}
