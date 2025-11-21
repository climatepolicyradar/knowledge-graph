export interface PredictionMetadata {
  "text_block.text_block_id": string;
  "text_block.page_number": string;
  document_id: string;
  translated: string;
  "document_metadata.publication_ts": string;
  "document_metadata.corpus_type_name": string;
  "document_metadata.slug"?: string;
  "document_metadata.family_slug"?: string;
  world_bank_region: string;
  similarity: string;
}

export interface Prediction {
  text: string;
  marked_up_text?: string;
  spans?: Array<{ start: number; end: number; label: string }>;
  metadata: PredictionMetadata;
}
