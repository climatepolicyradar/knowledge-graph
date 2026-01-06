export interface FilterParams {
  translated?: boolean;
  corpus_type?: string;
  world_bank_region?: string;
  publication_year_start?: number;
  publication_year_end?: number;
  document_id?: string;
  search?: string;
  has_predictions?: boolean;
}

/** Alias for FilterParams - used in UI components for consistency with client-side filtering */
export type FilterState = FilterParams;
