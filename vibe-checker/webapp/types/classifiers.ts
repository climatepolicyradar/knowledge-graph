export interface ClassifierData {
  id: string;
  name: string;
  date: string;
}

export interface ClassifierInfo {
  id: string;
  data?: ClassifierData;
  loading: boolean;
  error?: string;
}
