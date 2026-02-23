export interface ConceptData {
  wikibase_id: string;
  preferred_label?: string;
  description?: string;
  n_classifiers?: number;
}

/** Type for enriched concept data returned from the API - all fields have defaults applied */
export type EnhancedConcept = Omit<
  ConceptData,
  "preferred_label" | "description" | "n_classifiers"
> & {
  preferred_label: string;
  description: string;
  n_classifiers: number;
};
